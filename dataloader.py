import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import time
import pdb

def gaussian_filter(s, k):
    probs = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k, k+1)]
    return np.outer(probs, probs) 

class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, path_dir='./data'):
        self.images = []
        self.landmarks = []
        self.image_dir = os.path.join(path_dir, 'images')
        self.landmark_dir = os.path.join(path_dir, 'landmarks')
        
        # filter
        s, k = 2, 4
        self.filter = gaussian_filter(s, k)
        self.filter = self.filter / self.filter[k][k]  
        self.filter = torch.from_numpy(self.filter).repeat(1, 1, 1, 1)

        for path, _, fnames in os.walk(self.image_dir):
            for fname in fnames:
                name, ext = os.path.splitext(fname)
                self.images.append(fname)
                self.landmarks.append(name)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.images[idx])).convert('RGB')
        w, h = img.size
        img = np.array(img.resize((224, 224)))

        with open(os.path.join(self.landmark_dir, self.landmarks[idx])) as f:
            coords = f.readline().strip().split(' ')
            assert(len(coords) == 8)
            coords = np.array(coords).reshape(4, 2).astype(float)

        coords = coords / np.array([w, h]) * np.array([224, 224])
        coords = coords.astype(int)
        coords = coords[:, ::-1]
        # print(coords)

        coords_img = np.zeros((4, 224, 224))
        for i in range(0, 4):
            coords_img[i][coords[i][0]][coords[i][1]] = 1
            
        img = torch.from_numpy(img).permute((2, 0, 1)) / 255
        coords_img = torch.from_numpy(coords_img).unsqueeze(dim=1)
        # pdb.set_trace()
        
        coords_img = F.conv2d(coords_img, self.filter, stride=1, padding=4)
        # print(coords_img.shape)
        coords_img = coords_img.view((4, 224, 224))
        
        return img.float(), coords_img.float()
        

if __name__ == '__main__':
    dataset = LandmarkDataset()
    img, coords_img = dataset[0]

    plt.subplot(231)
    plt.imshow(img.numpy())

    plt.subplot(232)
    plt.imshow(coords_img.numpy()[0])
    plt.subplot(233)
    plt.imshow(coords_img.numpy()[1])
    plt.subplot(235)
    plt.imshow(coords_img.numpy()[2])
    plt.subplot(236)
    plt.imshow(coords_img.numpy()[3])
    plt.show()

    