


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
import torch
import os
import sys
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from solvers.equilibrium_solvers import EquilibriumProxGrad



# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.abspath('/content/drive/My Drive/deblur'))

# 设定工作目录为项目根目录
os.chdir('/content/drive/My Drive/deblur')

# 设置路径
val_data_location = "/content/drive/My Drive/deblur/validate set/default"
save_location = "/content/drive/My Drive/deblur/saved_model.ckpt"

# 定义数据变换
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建验证数据集和数据加载器
val_dataset = ImageFolder(root=val_data_location, transform=val_transform)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

# 加载模型
forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                      n_channels=3, n_spatial_dimensions=2).to(device=device)
internal_forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                               n_channels=3, n_spatial_dimensions=2).to(device=device)
learned_component = DnCNN(channels=n_channels)  # 这里的 learned_component 使用了 DnCNN

solver = EquilibriumProxGrad(linear_operator=internal_forward_operator, nonlinear_operator=learned_component,
                             eta=initial_eta, minval=-1, maxval=1).to(device=device)

checkpoint = torch.load(save_location, map_location=torch.device('cpu'))
solver.load_state_dict(checkpoint['solver_state_dict'])
solver.eval()


# 验证模型
for idx, (inputs, _) in enumerate(val_dataloader):
    # 在此处添加验证逻辑
    pass

print("Validation completed!")
