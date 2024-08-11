import torch.nn as nn
import torchvision.models as models
from activation import *


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        self.num_classes = num_classes
        
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.num_classes = num_classes
        
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.densenet(x)
        return x
    

class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        self.num_classes = num_classes
        
        self.densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.densenet(x)
        return x
    

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.num_classes = num_classes
        
        self.vgg11 = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        
        num_features = self.vgg11.classifier[6].in_features
        self.vgg11.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.vgg11(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.num_classes = num_classes
        
        self.vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.vgg19(x)
        return x
    

network_dict = {
    'ResNet50': ResNet50,
    'ResNet152': ResNet152,
    'DenseNet121': DenseNet121,
    'DenseNet201': DenseNet201,
    'VGG11': VGG11,
    'VGG19': VGG19,
}

if __name__ == '__main__':      
    model  = network_dict['ResNet50'](2)
    # Usage
    afs = ActivationFunction(LeakyReLU()).replace_activation_function(model)
    print(model)