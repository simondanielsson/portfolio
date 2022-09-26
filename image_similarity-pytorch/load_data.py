from typing import List, Tuple
import torch
import os
import natsort
import numpy as np

from PIL import Image 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 


def getStat(data):

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(data))
    std.div_(len(data))

    return list(mean.numpy()), list(std.numpy())


def load_images(root_folder: str) :
    
    """
    root_floder: path of the rootfolder. 
               In our case, under root_folder, there is only one folder called "food".
               E.g. 
               root_folder = os.path.join("drive", "MyDrive", "task3-iml", "data", "food.zip (Unzipped Files)")
    
    image_data: ImageFolder: List[(torch.Tensor, index)]
    image_list: List(tensor)
    
    """
    
    # 1. Get the mean, std after resize
    #transform = transforms.Compose([transforms.Resize([256, 256]),
    #              transforms.ToTensor(),])
    #data = ImageFolder(root_folder, transform=transform)
    
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Normalization
    transform = transforms.Compose([transforms.Resize([256, 256]),
                  transforms.ToTensor(),
                  transforms.Normalize(img_mean, img_std)])
    image_data = ImageFolder(root=root_folder, transform=transform)
    
    image_list = [img_tensor for (img_tensor,_) in image_data]

    return image_list

def load_triplets(path) -> List[Tuple[str, str, str]]:
    """
    Reads the txt-files of image-triplets and hands them over as a List of triplets.
    """

    triplets = []
    raw = np.genfromtxt(path)
    
    for i in range(raw.shape[0]):
        (a, p, n) = raw[i, :]
        triplets.append((a, p, n))

    return triplets

