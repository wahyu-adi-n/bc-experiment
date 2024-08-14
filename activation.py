from config import SAVE_AFS_PLOTTING
import matplotlib.pyplot as plt
import os
import shutil
import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.name = 'ReLU'
        self.inplace = inplace
        
    def forward(self, x):
        if self.inplace:
            return x.clamp_(min=0)  # In-place operation
        else:
            return torch.maximum(torch.tensor(0.0), x)

class LeakyReLU(nn.Module):
    def __init__(self, a=0.01, inplace=False):
        super(LeakyReLU, self).__init__()
        self.name = 'LeakyReLU'
        self.a = a
        self.inplace = inplace
        
    def forward(self, x):
        if self.inplace:
            return x.mul_(self.a).clamp_(min=0).add_(torch.minimum(x, torch.tensor(0.0)))
        else:
            return torch.maximum(x * self.a, x)

class LessNegativeReLU(nn.Module):
    def __init__(self, a, inplace=False):
        super(LessNegativeReLU, self).__init__()
        self.name = f'LessNegativeReLU {a}'
        self.a = a
        self.inplace = inplace
        
    def forward(self, x):
        if self.inplace:
            x.mul_(self.a).add_(torch.maximum(torch.zeros_like(x), x)).sub_(torch.maximum(torch.zeros_like(x), x*self.a))
            return x
        else:
            return torch.maximum(x * self.a, x)
        
        
class ActivationFunction():
    def __init__(self, afs) -> None:
        self.activation_module =  afs
        self.x = torch.linspace(-10, 10, 100)
        self.y = self.activation_module(self.x)
    
    def replace_activation_function(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(model, name, self.activation_module)
            else:
                self.replace_activation_function(module)

    def plot_activaition_function(self):
        if not os.path.exists(SAVE_AFS_PLOTTING):
            os.makedirs(SAVE_AFS_PLOTTING)
        plt.figure(figsize=(12, 8))
        plt.plot(self.x, self.y)
        plt.grid(True)
        plt.title(f'{self.activation_module.name} Activation Function', fontsize=16)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.savefig(f'{SAVE_AFS_PLOTTING}/{self.activation_module.name} Activation Function Plotting.png', 
                    format='png')
        plt.close()

activation_dict = {    
                        'ReLU': ActivationFunction(ReLU(inplace=True)),
                        'LeakyReLU' : ActivationFunction(LeakyReLU(inplace=True)),
                        'LessNegativeReLU_0.03': ActivationFunction(LessNegativeReLU(0.03, inplace=True)),
                        'LessNegativeReLU_0.05': ActivationFunction(LessNegativeReLU(0.05, inplace=True)),
                        'LessNegativeReLU_0.07': ActivationFunction(LessNegativeReLU(0.07, inplace=True)),
                        'LessNegativeReLU_0.09': ActivationFunction(LessNegativeReLU(0.09, inplace=True)),
                  }
    
if __name__ == '__main__':
    if os.path.exists(SAVE_AFS_PLOTTING):
        shutil.rmtree(SAVE_AFS_PLOTTING)
    
    activation_funtions = [
                            ActivationFunction(ReLU()), 
                            ActivationFunction(LeakyReLU()),
                            ActivationFunction(LessNegativeReLU(0.03)),
                            ActivationFunction(LessNegativeReLU(0.05)),
                            ActivationFunction(LessNegativeReLU(0.07)),
                            ActivationFunction(LessNegativeReLU(0.09)),
                        ]
    for afs in activation_funtions:
        afs.plot_activaition_function()