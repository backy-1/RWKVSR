import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
mat_data = loadmat("660_apples/rwkv.mat")
fake_data_np = mat_data['SR']  # (H, W, B)
real_data_np = mat_data['HR']  # (H/3, W/3, B)

# 转换为 PyTorch 张量并归一化到 [0, 1]
fake_data = torch.tensor(fake_data_np, dtype=torch.float32) / 255.0  # 假设数据是 uint8 类型
real_data = torch.tensor(real_data_np, dtype=torch.float32) / 255.0

# 选择第五波段
fake_band = fake_data[:, :, 4].squeeze()  # (H, W)
real_band = real_data[:, :, 4].squeeze()  # (H/3, W/3)


def compute_spectrum(image, db_scale=20):
    """
    计算 FFT 频谱并转换为 dB
    """
    # 输入数据归一化到 [0, 1]
    image_normalized = image.clamp(0, 1)

    # FFT 变换（无需 normalized 参数）
    fft = torch.fft.fft2(image_normalized)
    power = torch.abs(fft) ** 2

    # 归一化到 [0, 1]（避免 log10 溢出）
    power_normalized = power / power.max()

    # 转换为分贝（dB）
    db = db_scale * torch.log10(power_normalized + 1e-10)
    return db


# 计算频谱（动态范围调整）
fake_spectrum_db = compute_spectrum(fake_band)
real_spectrum_db = compute_spectrum(real_band)

# 动态调整显示范围（根据实际数据）
vmin = max(fake_spectrum_db.min(), real_spectrum_db.min()) - 10
vmax = min(fake_spectrum_db.max(), real_spectrum_db.max()) + 10

# 可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(fake_spectrum_db, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('Fake Spectrum (dB)')
plt.colorbar(label='dB')

plt.subplot(1, 2, 2)
plt.imshow(real_spectrum_db, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('Real Spectrum (dB)')
plt.colorbar(label='dB')

plt.tight_layout()
plt.show()