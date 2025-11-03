% clc; clear; close all;
% 
% %% 参数设置
% scales = [2, 3, 4];   % 放大倍数
% dataset_name = 'PaviaC';
% src_file = '/mnt/data/LSH/py_project/deepx/Datasets/PaviaC/Pavia.mat';
% save_root = '/mnt/data/LSH/py_project/SRDNet-main/dataset/tests/';
% 
% %% Step 1: 加载数据
% data = load(src_file);
% if isfield(data,'paviaC')
%     img = data.paviaC;
% elseif isfield(data,'pavia')
%     img = data.pavia;
% else
%     fn = fieldnames(data);
%     img = data.(fn{1});
% end
% img = single(img);
% img = img ./ max(img(:));  % 归一化
% fprintf('原始图像尺寸: %d × %d × %d\n', size(img));
% 
% %% Step 1.5: 调整尺寸以保证能整除
% max_scale = max(scales);
% H = size(img, 1);
% W = size(img, 2);
% H_new = floor(H / max_scale) * max_scale;
% W_new = floor(W / max_scale) * max_scale;
% if H_new ~= H || W_new ~= W
%     fprintf('⚙️ 调整尺寸: (%d, %d) → (%d, %d)\n', H, W, H_new, W_new);
%     img = img(1:H_new, 1:W_new, :);
% end
% 
% %% Step 2: 为每个 scale 生成 HR/LR 数据
% for s = 1:numel(scales)
%     scale = scales(s);
%     factor = 1 / scale;
% 
%     % 生成保存目录
%     test_folder = fullfile(save_root, dataset_name, num2str(scale));
%     if ~exist(test_folder, 'dir'), mkdir(test_folder); end
% 
%     % HR / LR 数据生成
%     hr = img;
%     lr = imresize(hr, factor, 'bicubic');  % 降采样
%     hr = single(permute(hr, [3 1 2]));     % [C,H,W]
%     lr = single(permute(lr, [3 1 2]));
% 
%     % 保存
%     save(fullfile(test_folder, sprintf('PC_test_x%d.mat', scale)), 'hr', 'lr', '-v6');
%     fprintf('✅ 已生成 scale ×%d 的测试数据：%s\n', scale, test_folder);
% end
% 
% fprintf('🎯 所有测试集已生成完毕！\n📂 路径：%s\n', save_root);

clc; clear; close all;

%% 参数设置
scales = [2, 3, 4];   % 放大倍数
dataset_name = 'PaviaC';
save_root = '/mnt/data/LSH/py_project/SRDNet-main/dataset/tests1/';
src_file = '/mnt/data/LSH/py_project/deepx/Datasets/PaviaC/Pavia.mat'; % 修改为你的文件路径

%% Step 1: 加载原始数据
data = load(src_file);
if isfield(data, 'paviaC')
    img = data.paviaC;
elseif isfield(data, 'pavia')
    img = data.pavia;
else
    fn = fieldnames(data);
    img = data.(fn{1});
end
img = single(img);
img = img ./ max(img(:));  % 归一化
fprintf('原始图像尺寸: %d × %d × %d\n', size(img));

%% Step 2: 因为当前数据已为 1096×715×102，无需再次裁剪无效区
img_valid = img;
fprintf('有效区域尺寸: %d × %d × %d\n', size(img_valid));

%% Step 3: 从底部截取 128×715×102 子图
H = size(img_valid, 1);
sub_bottom = img_valid(H-128+1:H, :, :);
fprintf('底部子图尺寸: %d × %d × %d\n', size(sub_bottom));

%% Step 4: 将子图沿宽度方向裁成 4 个 128×128×102 不重叠块
patches = cell(1, 4);
x_positions = [1, 129, 257, 385];  % 每个128宽度，间隔紧凑
for i = 1:4
    x_start = x_positions(i);
    x_end = x_start + 127;
    if x_end > size(sub_bottom, 2)
        x_end = size(sub_bottom, 2);
    end
    patches{i} = sub_bottom(:, x_start:x_end, :);
    fprintf('Patch %d: [%d : %d]\n', i, x_start, x_end);
end

%% Step 5: 为每个 scale 生成 HR/LR 对
for s = 1:numel(scales)
    scale = scales(s);
    factor = 1 / scale;

    test_folder = fullfile(save_root, dataset_name, num2str(scale));
    if ~exist(test_folder, 'dir'), mkdir(test_folder); end

    for p = 1:4
        hr = patches{p};
        lr = imresize(hr, factor, 'bicubic');

        hr = single(permute(hr, [3 1 2])); % [C,H,W]
        lr = single(permute(lr, [3 1 2]));

        save(fullfile(test_folder, sprintf('PC_patch%d_x%d.mat', p, scale)), 'hr', 'lr', '-v6');
        fprintf('✅ 已生成 patch %d (scale×%d)\n', p, scale);
    end
end

fprintf('🎯 所有 Pavia Center 测试补丁生成完毕！\n📂 路径：%s\n', save_root);