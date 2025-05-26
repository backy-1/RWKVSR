import torch
import torch.nn as nn
import pdb
from torch.nn import functional as F
from einops import rearrange
import math
T_MAX = 512 * 512

from torch.utils.cpp_extension import load

wkv_cuda = load(name="wkv", sources=["./cuda/wkv_op.cpp", "./cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class OmniShift(nn.Module):
    def __init__(self, wn, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = wn(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False))
        self.conv3x3 = wn(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False))
        self.conv5x5 = wn(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False))
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = wn(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim,
                                         bias=False))
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        # import pdb
        # pdb.set_trace()

        out = self.alpha[0] * x + self.alpha[1] * out1x1 + self.alpha[2] * out3x3 + self.alpha[3] * out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)).to(device='cuda')
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)).to(device='cuda')

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)).to(device='cuda')
        # print('5*5',self.conv5x5.weight.shape)
        # print('3',self.alpha[3].shape)
        weight = self.conv5x5.weight.to(padded_weight_3x3.device)

        combined_weight = self.alpha[0] * identity_weight + self.alpha[1] * padded_weight_1x1 + self.alpha[
            2] * padded_weight_3x3 + self.alpha[3] * weight

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight).cuda()

    def forward(self, x):

        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)

        return out

class FeedForward(nn.Module):
    def __init__(self, wn, dim, mult=4, dropout=0.01):
        super().__init__()
        self.net = nn.Sequential(
            wn(nn.Conv2d(dim, dim * mult, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            wn(nn.Conv2d(dim * mult, dim, 1)),
        )
    def forward(self, x):
        return self.net(x)
class VRWKV_SpatialMix(nn.Module):
    def __init__(self, wn, n_embd, n_layer, layer_id, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        self.dwconv = wn(nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False))

        self.recurrence = 2

        self.omni_shift = OmniShift(wn, dim=n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

        # self.gelu = nn.GELU()

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        k = self.key(x)

        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        # sr = self.gelu(r)

        return sr, k, v

    def forward(self, x, resolution):

        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, wn, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        self.omni_shift = OmniShift(wn, dim=n_embd)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        # self.gelu = nn.GELU()


    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        # x = self.gelu(self.receptance(x)) * kv

        return x


class rwkvblock(nn.Module):
    def __init__(self, wn, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(wn, n_embd, n_layer, layer_id, init_mode,
                                    key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(wn, n_embd, n_layer, layer_id, hidden_rate,
                                    init_mode, key_norm=key_norm)

        self.ffn_mlp = FeedForward(wn, n_embd)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True).cuda()
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True).cuda()

    def forward(self, x):

        b, c, h, w = x.shape
        residual = x

        resolution = (h, w)
        x1 = rearrange(x, 'b c h w -> b (h w) c')
        x1 = self.att(self.ln1(x1), resolution)
        x1 = rearrange(x1, 'b (h w) c -> b h w c', h=h, w=w)

        x2 = rearrange(x, 'b c h w -> b (h w) c')
        x2 = self.ffn(self.ln2(x2), resolution)
        x2 = rearrange(x2, 'b (h w) c -> b h w c', h=h, w=w)

        x3 = x.permute(0, 2, 3, 1) + self.gamma1 * x1 + self.gamma2 * x2
        out = x3.permute(0, 3, 1, 2) + self.ffn_mlp(self.ln3(x3).permute(0, 3, 1, 2))
        return out

class BasicConv3d(nn.Module):
    def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv = wn(nn.Conv3d(in_channel, out_channel,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# class S3Dblock(nn.Module):
#     def __init__(self, wn, n_feats):
#         super(S3Dblock, self).__init__()
#
#         self.conv = nn.Sequential(
#             BasicConv3d(wn, n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
#             BasicConv3d(wn, n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
#         )
#
#     def forward(self, x):
#         return self.conv(x)
class S3Dblock(nn.Module):
    def __init__(self, wn, n_feats):
        super(S3Dblock, self).__init__()

        # self.conv = nn.Sequential(
        #     BasicConv3d(wn, n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
        #     BasicConv3d(wn, n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        # )
        # self.conv1 = BasicConv3d(wn, n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.conv2 = BasicConv3d(wn, n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)))
        self.conv2 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)))
        self.conv3 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1)))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = torch.add(self.conv2(out), self.conv3(out))

        # out = self.conv1(x)
        # out = self.relu(out)
        # out = self.conv2(out)
        return out


def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x
# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         # print('k',k.shape)
#         # print('v', v.shape)
#         attn_weights = torch.matmul(q, k.transpose(2, 3))
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)
#         attended_values = torch.matmul(attn_weights, v)
#         return attended_values


class Block(nn.Module):
    def __init__(self, wn, n_feats, n_conv, num_blocks=[1, 2, 2, 4]):
        super(Block, self).__init__()

        self.relu = nn.ReLU(False)

        Block1 = []
        for i in range(n_conv):
            # Block1.append(S3Dblock(wn, n_feats))
            Block1.append(S3Dblock(wn, n_feats))
        self.Block1 = nn.Sequential(*Block1)

        Block2 = []
        for i in range(n_conv):
            Block2.append(S3Dblock(wn, n_feats))
        self.Block2 = nn.Sequential(*Block2)

        Block3 = []
        for i in range(n_conv):
            Block3.append(S3Dblock(wn, n_feats))
        self.Block3 = nn.Sequential(*Block3)

        self.reduceF = BasicConv3d(wn, n_feats * 3, n_feats, kernel_size=1, stride=1)
        self.Conv = S3Dblock(wn, n_feats)
        self.gamma = nn.Parameter(torch.ones(3))

        # self.conv1 = nn.Sequential(*[rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[0], layer_id=i) for i in range(num_blocks[0])])
        #
        # self.conv2 = nn.Sequential(*[rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[1], layer_id=i) for i in range(num_blocks[1])])
        #
        # self.conv3 = nn.Sequential(*[rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[2], layer_id=i) for i in range(num_blocks[2])])
        self.conv1 = nn.Sequential(
            *[rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[0], layer_id=i) for i in range(1)])

        self.conv2 = nn.Sequential(
            *[rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[1], layer_id=i) for i in range(1)])

        self.conv3 = nn.Sequential(
            *[rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[2], layer_id=i) for i in range(2)])
        # self.sa = SelfAttention(32)

    def forward(self, x):

        res = x
        x1 = self.Block1(x) + x
        x2 = self.Block2(x1) + x1
        x3 = self.Block3(x2) + x2

        x1, depth = _to_4d_tensor(x1, depth_stride=1)
        x1 = self.conv1(x1)
        # # print('x1', x1.shape)
        x1 = _to_5d_tensor(x1, depth)
        # # print(res.shape, x1.shape)

        x2, depth = _to_4d_tensor(x2, depth_stride=1)
        # x2, depth = _to_4d_tensor(x1, depth_stride=1)
        x2 = self.conv2(x2)
        x2 = _to_5d_tensor(x2, depth)
        #
        x3, depth = _to_4d_tensor(x3, depth_stride=1)
        x3 = self.conv3(x3)
        x3 = _to_5d_tensor(x3, depth)

        x = torch.cat([self.gamma[0] * x1, self.gamma[1] * x2, self.gamma[2] * x3], 1)
        # x = torch.cat([self.gamma[0] * x1, self.gamma[1] * x2,], 1)
        # print("x",x.shape)
        x = self.reduceF(x)
        x = self.relu(x)
        x = x + res

        x = self.Conv(x)
        return x


class RWKVNet(nn.Module):
    def __init__(self, args):
        super(RWKVNet, self).__init__()

        scale = args.upscale_factor
        n_colors = args.n_colors
        n_feats = args.n_feats
        n_conv = args.n_conv
        kernel_size = args.kernel_size

        # scale = 3
        # n_colors = 31
        # n_feats = 64
        # n_conv = 1
        # kernel_size = 3
        #
        # band_mean = (0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834,
        #              0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194,
        #              0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541)  # CAVE
        band_mean = (0.0100, 0.0137, 0.0219, 0.0285, 0.0376, 0.0424, 0.0512, 0.0651, 0.0694, 0.0723, 0.0816,
                    0.0950, 0.1338, 0.1525, 0.1217, 0.1187, 0.1337, 0.1481, 0.1601, 0.1817, 0.1752, 0.1445,
                    0.1450, 0.1378, 0.1343, 0.1328, 0.1303, 0.1299, 0.1456, 0.1433, 0.1303) #Hararvd

        #        band_mean = (0.0944, 0.1143, 0.1297, 0.1368, 0.1599, 0.1853, 0.2029, 0.2149, 0.2278, 0.2275, 0.2311,
        #                     0.2331, 0.2265, 0.2347, 0.2384, 0.1187, 0.2425, 0.2441, 0.2471, 0.2453, 0.2494, 0.2584,
        #                     0.2597, 0.2547, 0.2552, 0.2434, 0.2386, 0.2385, 0.2326, 0.2112, 0.2227) #ICVL

        # band_mean = (0.0483, 0.0400, 0.0363, 0.0373, 0.0425, 0.0520, 0.0559, 0.0539, 0.0568, 0.0564, 0.0591,
        #             0.0678, 0.0797, 0.0927, 0.0986, 0.1086, 0.1086, 0.1015, 0.0994, 0.0947, 0.0980, 0.0973,
        #             0.0925, 0.0873, 0.0887, 0.0854, 0.0844, 0.0833, 0.0823, 0.0866, 0.1171, 0.1538, 0.1535) #Foster

        #        band_mean = (0.0595,	0.0600,	0.0651,	0.0639,	0.0641,	0.0637,	0.0646,	0.0618,	0.0679,	0.0641,	0.0677,
        #        	           0.0650,	0.0671,	0.0687,	0.0693,	0.0687,	0.0688,	0.0677,	0.0689,	0.0736,	0.0735,	0.0728,	0.0713,	0.0734,
        #        	           0.0726,	0.0722,	0.074,	0.0742,	0.0794,	0.0892,	0.1005) #Foster2002
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, n_colors, 1, 1])

        self.head = wn(nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size // 2))

        self.SSRM1 = Block(wn, n_feats, n_conv)
        self.SSRM2 = Block(wn, n_feats, n_conv)
        self.SSRM3 = Block(wn, n_feats, n_conv)
        self.SSRM4 = Block(wn, n_feats, n_conv)

        tail = []
        tail.append(
            wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3, 2 + scale, 2 + scale), stride=(1, scale, scale),
                                  padding=(1, 1, 1))))
        tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size // 2)))
        self.tail = nn.Sequential(*tail)

        # self.var = nn.Parameter(torch.tensor(0.1, requires_grad=True))

    def forward(self, x):
        x = x - self.band_mean.cuda()
        x = x.unsqueeze(1)
        T = self.head(x)
        # x = self.dropout(x)

        x = self.SSRM1(T)
        # x = self.dropout(x)
        x = torch.add(x, T)


        x = self.SSRM2(x)
        x = torch.add(x, T)
        #
        x = self.SSRM3(x)
        x = torch.add(x, T)
        #
        # x = self.SSRM4(x)
        # x = torch.add(x, T)

        x = self.tail(x)
        # var1_clipped = torch.clamp(self.var, min=1e-7, max=0.999)
        # x = self.dropout(x)
        x = x.squeeze(1)
        # x = self.dropout(x)

        x = x + self.band_mean.cuda()
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import time
    from thop import profile, clever_format

    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''

    x = torch.zeros((1, 1, 32, 32)).type(torch.FloatTensor).cuda()
    model = RWKVNet()
    model.cuda()

    since = time.time()
    y = model(x)
    print("time", time.time() - since)

    flops, params = profile(model, inputs=(x,))
    # flops, params = clever_format([flops, params], '%.6f')
    print('flops', flops)
    print('params', params)
    print(count_parameters(model) / 1e6)