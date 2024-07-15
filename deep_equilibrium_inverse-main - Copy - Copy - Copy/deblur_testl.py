
import torch
import os
import random
import sys
import argparse
sys.path.append('/home-nfs/gilton/learned_iterative_solvers')
# sys.path.append('/Users/dgilton/PycharmProjects/learned_iterative_solvers')

# 添加以下两行代码
sys.path.append(os.path.abspath('/content/drive/My Drive/deblur'))
os.chdir('/content/drive/My Drive/deblur')

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import operators.blurs as blurs
from operators.operator import OperatorPlusNoise
from utils.celeba_dataloader import CelebaTrainingDatasetSubset, CelebaTestDataset
from networks.normalized_equilibrium_u_net import UnetModel, DnCNN
from solvers.equilibrium_solvers import EquilibriumProxGrad
from training import refactor_equilibrium_training
from solvers import new_equilibrium_utils as eq_utils

import torchvision.transforms.functional as TF
import numpy as np

# 设置路径
test_data_location = "/content/drive/My Drive/deblur/training data-CeleBA/img_align_celeba/"



output_folder = "/content/drive/My Drive/deblur/test_output"

# 定义数据转换
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# 测试集数据路径
celeba_test_size = 345  # 修改为您的测试集图片数量

# 创建测试集
test_dataset = CelebaTestDataset(test_data_location, transform=transform)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False,
)



# 加载已训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(model_path, map_location=device)

# 初始化模型相关参数
kernel_size = 5
kernel_sigma = 5.0
noise_sigma = 1e-2
n_channels = 3
initial_eta = 0.2

# 初始化模型
forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                      n_channels=n_channels, n_spatial_dimensions=2).to(device=device)
internal_forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                               n_channels=n_channels, n_spatial_dimensions=2).to(device=device)
learned_component = DnCNN(channels=n_channels)  # 这里的 learned_component 使用了 DnCNN

solver = EquilibriumProxGrad(linear_operator=internal_forward_operator, nonlinear_operator=learned_component,
                             eta=initial_eta, minval=-1, maxval=1).to(device=device)

# 加载模型参数
solver.load_state_dict(checkpoint['solver_state_dict'])
solver.eval()


# 计算PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# 确保输出文件夹存在，如果不存在则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"Output folder: {output_folder}")

# 测试并保存生成图像
for idx, (inputs, _) in enumerate(test_dataloader):
    print(f"Processing image {idx}")
    inputs = inputs.to(torch.device('cpu'))
    with torch.no_grad():
        outputs = solver(inputs)
    outputs = outputs.squeeze().cpu()
    inputs = inputs.squeeze().cpu()
    psnr_value = calculate_psnr(inputs.numpy(), outputs.numpy())
    output_img = TF.to_pil_image(outputs)
    output_path = os.path.join(output_folder, f"IMG_{idx}_{psnr_value:.2f}dB.png")
    output_img.save(output_path)
    print(f"Saved output image to: {output_path}")

print("Testing completed!")
