# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time


from loss import HFL
from torchnet import meter
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR

from torch.optim.lr_scheduler import MultiStepLR
from architecture.common import *
from RWKVSR import RWKVNet
import torch.nn.init as init
import scipy.io as scio


psnr = []
out_path = 'result/' + opt.datasetName + '/'

log_interval = 50
per_epoch_iteration = 10

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def sam_loss(x_gt, x_sr, eps=1e-8):
    # x_gt, x_sr: shape [B, C, H, W]
    x_gt_flat = x_gt.view(x_gt.shape[0], x_gt.shape[1], -1)
    x_sr_flat = x_sr.view(x_sr.shape[0], x_sr.shape[1], -1)

    # dot product
    dot = torch.sum(x_gt_flat * x_sr_flat, dim=1)

    # norms
    norm_gt = torch.norm(x_gt_flat, dim=1)
    norm_sr = torch.norm(x_sr_flat, dim=1)

    # avoid divide by zero
    cos_sim = dot / (norm_gt * norm_sr + eps)

    # clip to avoid NaNs
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # SAM loss: angle in radians
    loss = torch.acos(cos_sim)

    return torch.mean(loss)
def main():
    global best_psnr
    global writer
    if opt.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    train_set = TrainsetFromFolder(
        '/mnt/data/LSH/py_project/SRDNet-main/dataset/trains' + opt.datasetName + '/' + str(opt.upscale_factor) + '/')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_set = ValsetFromFolder(
        '/mnt/data/LSH/py_project/SRDNet-main/dataset/evals/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/')
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    model = RWKVNet(opt)

    criterion = nn.L1Loss()
    HFL_loss=HFL()


    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    # loss functions to choose
    # mse_loss = torch.nn.MSELoss()
    # criterion = HybridLoss(spatial_tv=True, spectral_tv=True)
    # hylap_loss = HyLapLoss(spatial_tv=False, spectral_tv=True)

    if opt.cuda:
        # model = nn.DataParallel(model).cuda()
        model = model.cuda()
        criterion = criterion.cuda()
        HFL_loss= HFL_loss.cuda()

    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-08)

    # optionally resuming from a checkpoints_32
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoints '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoints found at '{}'".format(opt.resume))

    # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105,140,175], gamma=0.5, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,opt.nEpochs,eta_min=2e-5)

    writer = SummaryWriter(log_dir='logs/' + opt.datasetName + opt.method+ '_' + str(time.ctime()))
    # Training

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(train_loader, optimizer, model, criterion, epoch,HFL_loss)
        # train(train_loader, optimizer, model, criterion, epoch, HFL_loss)
        val(val_loader, model, epoch)
        save_checkpoint(epoch, model, optimizer)
        scheduler.step()

    scio.savemat(out_path + 'LFF1GRL0.mat', {'psnr': psnr})  # , 'ssim':ssim, 'sam':sam})
    # scio.savemat(out_path + opt.datasetName + opt.method+'LFF1GRL0.txt', {'psnr': psnr})  # , 'ssim':ssim, 'sam':sam})
nearest = nn.Upsample(scale_factor=opt.upscale_factor, mode='nearest')
def train(train_loader, optimizer, model, criterion, epoch,HFL_loss):
# def train(train_loader, optimizer, model, criterion, epoch, HFL_loss):
    for iteration, batch in enumerate(train_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        lms = nearest(input)
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()
            lms =  lms.cuda()

        train_loss = []
        SR = model(input)
        # print('label', input.shape)
        # print('input', input.shape)

        _loss = HFL_loss(SR, label)
        loss = criterion(SR, label)
        S_loss = sam_loss(SR,label)
        loss = loss+0.1*_loss+0.2*S_loss

        optimizer.zero_grad()
        loss.backward()
        train_loss.append(loss)

        optimizer.step()
        if (iteration + log_interval) % log_interval == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.item()))
          
        writer.add_scalar('scalar/train_loass', loss,epoch)

def val(val_loader, model, epoch):

    model.eval()
    val_psnr = 0
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for iteration, batch in enumerate(val_loader, 1):
            with torch.no_grad():
                input = batch[0]  # 直接使用 Tensor
            HR = batch[1]

            lms=nearest(input)
            if opt.cuda:
                input = input.cuda()
                lms = lms.cuda()
            SR = model(input)
            val_psnr += PSNR(SR.cpu().detach().numpy(), HR.detach().numpy())

        val_psnr = val_psnr / len(val_loader)
        print("PSNR = {:.3f}".format(val_psnr))

    psnr.append(val_psnr)
    writer.add_scalar('Val/PSNR',  val_psnr, epoch)
def save_checkpoint(epoch, model, optimizer):
    model_out_path = "checkpoint/" + "{}_model_{}_epoch_{}.pth".format(opt.datasetName, opt.upscale_factor, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists("all_checkpoint/checkpoint/"):
        os.makedirs("all_checkpoint/checkpoint/")
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()
