from config import *
from torchvision import transforms as T
from tqdm import tqdm
import datasets
import torch

dataset = datasets.BreaKHis('binary', 'train', magnification=None, transform=T.ToTensor())
full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=8)

mean = torch.zeros(3)
std = torch.zeros(3)
print('==> Computing mean and std..')
for inputs, _labels in tqdm(full_loader, total=len(full_loader)):
    for i in range(N_CHANNELS):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)
