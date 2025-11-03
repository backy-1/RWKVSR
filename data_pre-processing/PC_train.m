% preprocess_PU_dataset.m
% 数据预处理: Pavia University (PU) 数据集
% 生成 HR/LR patch，用于超分辨率训练和测试

clc; clear; close all;

%% 参数设置
patch_size = 64;        % patch 尺寸
stride     = 32;        % 滑动步长
scales     = [2,3,4];   % 放大倍数
train_ratio = 0.8;      % 训练集比例

% 数据路径
src_file   = '/mnt/data/LSH/py_project/deepx/Datasets/PaviaC/Pavia.mat';
save_root  = '/mnt/data/LSH/py_project/main/dataset/';  %  数据根目录

dataset_name = 'PaviaC';  %  统一命名

%% Step 1: 加载数据
data = load(src_file);
if isfield(data,'paviaC')
    img = data.paviaC;
elseif isfield(data,'pavia')
    img = data.pavia;
else
    fn = fieldnames(data);
    img = data.(fn{1});
end
img = single(img);
img = img ./ max(img(:));  % 归一化
[H,W,C] = size(img);

%% Step 2: 生成 HR patch
fprintf('生成 HR patch...\n');
patches = {}; cnt = 0;
for x = 1:stride:(H - patch_size + 1)
    for y = 1:stride:(W - patch_size + 1)
        cnt = cnt + 1;
        patch = img(x:x+patch_size-1, y:y+patch_size-1, :);
        patches{cnt} = patch;
    end
end
fprintf('共生成 %d 个 patch\n', cnt);

%% Step 3: 划分训练 / 测试
rand_idx = randperm(cnt);
train_num = floor(train_ratio * cnt);
train_idx = rand_idx(1:train_num);
test_idx  = rand_idx(train_num+1:end);

train_patches = patches(train_idx);
test_patches  = patches(test_idx);

%% Step 4: 生成 HR / LR 数据并保存
for s = 1:numel(scales)
    scale = scales(s);
    factor = 1 / scale;

    % ✅ 保存路径改结构
    train_folder = fullfile(save_root, 'trains', dataset_name, num2str(scale));
    val_folder   = fullfile(save_root, 'evals', dataset_name, num2str(scale));

    if ~exist(train_folder, 'dir'), mkdir(train_folder); end
    if ~exist(val_folder, 'dir'), mkdir(val_folder); end

    fprintf('处理 scale x%d ...\n', scale);

    % --- 训练集 ---
    for i = 1:numel(train_patches)
        hr = train_patches{i};
        lr = imresize(hr, factor, 'bicubic');
        hr = single(permute(hr, [3 1 2]));  % [C,H,W]
        lr = single(permute(lr, [3 1 2]));
        save(fullfile(train_folder, sprintf('PC_train_%05d.mat', i)), 'hr','lr','-v6');
    end

    % --- 验证集 ---
    for i = 1:numel(test_patches)
        hr = test_patches{i};
        lr = imresize(hr, factor, 'bicubic');
        hr = single(permute(hr, [3 1 2]));
        lr = single(permute(lr, [3 1 2]));
        save(fullfile(val_folder, sprintf('PC_val_%05d.mat', i)), 'hr','lr','-v6');
    end
end

fprintf(' PaviaC 数据集预处理完成！\n');
fprintf(' 数据存放在：%s\n', save_root);
