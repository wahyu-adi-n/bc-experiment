from config import SAVE_AFS_PLOTTING
import matplotlib.pyplot as plt
import os
import shutil
import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.name = 'ReLU'
        
    def forward(self, x):
        return torch.maximum(torch.tensor(0.0), x)

class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()
        self.name = 'LeakyRELU'
        
    def forward(self, x, a = 0.01):
        return torch.maximum(x*a, x)

class ParametricRELU(nn.Module):
    def __init__(self, a):
        super(ParametricRELU, self).__init__()
        self.name = f'ParametricRELU {a}'
        self.a = a
        
    def forward(self, x):
        return torch.maximum(x*self.a, x)
        
        
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
        plt.title(f'{self.activation_module.name} Activation Function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{SAVE_AFS_PLOTTING}/{self.activation_module.name} Activation Function Plotting.png', 
                    format='png')
        plt.close()

activation_dict = {    
                        'ReLU': ActivationFunction(ReLU()),
                        'LeakyReLU' : ActivationFunction(LeakyReLU()),
                        'ParametricReLU_0.1': ActivationFunction(ParametricRELU(0.1)),
                        'ParametricReLU_0.3': ActivationFunction(ParametricRELU(0.2)), 
                        'ParametricReLU_0.5': ActivationFunction(ParametricRELU(0.3)), 
                        'ParametricReLU_0.7': ActivationFunction(ParametricRELU(0.4)),  
                        'ParametricReLU_0.9': ActivationFunction(ParametricRELU(0.5)), 
                    }
    
if __name__ == '__main__':
    if os.path.exists(SAVE_AFS_PLOTTING):
        shutil.rmtree(SAVE_AFS_PLOTTING)
    
    activation_funtions = [
                            ActivationFunction(ReLU()), 
                            ActivationFunction(LeakyReLU()), 
                            ActivationFunction(ParametricRELU(0.1)),
                            ActivationFunction(ParametricRELU(0.2)),
                            ActivationFunction(ParametricRELU(0.3)),
                            ActivationFunction(ParametricRELU(0.4)),
                            ActivationFunction(ParametricRELU(0.5)),
                        ]
    for afs in activation_funtions:
        afs.plot_activaition_function()