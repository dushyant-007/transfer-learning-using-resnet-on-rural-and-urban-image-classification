import torch
import numpy as np
import torch.nn as nn
import torchsummary
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch.nn.functional as f
from PIL import  Image
from pathlib import Path
import json

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class resnet_model(nn.Module):

    def __init__(self, pre_trained = True, num_classes = 2):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT

        if pre_trained is True:
            self.resnet18 = models.resnet18(pretrained=True)
        else:
            self.resnet18 = models.resnet18()

        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features , num_classes)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.resnet18.fc.requires_grad_(True)

        self.transforms = weights.transforms()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return f.softmax(self.resnet18(self.transforms(x)).float(), dim=1).argmax(dim =1 )

if __name__ == '__main__':
    model = resnet_model(pre_trained=True)
    img = []
    for i in range(1, 6):
        img.append(read_image('imagenet/img_'+str(i)+'.png'))
    #show(img)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    result =[]
    for i in img:
        i = i.to(device)
        shape_o_i = i.shape
        result.append(model(i[:3, :, :].reshape(1, 3, shape_o_i[1] , shape_o_i[2])))
    '''for res in range(result):
        print(f'results are {result}')
        print(f'results should be ')'''
    print(result)






