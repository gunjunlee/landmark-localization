import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb
import os
import time
import argparse

from unet import Unet
from dataloader import LandmarkDataset

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg('--data', default='./data', help='data directory')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = LandmarkDataset(args.data)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8)
    net = Unet(n_classes=4).cuda()
    net = nn.DataParallel(net)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    print('training start')
    for epoch in range(50):
        print('epoch: {}'.format(epoch))
        train_running_loss = 0
        
        net.train()
        for batch_images, batch_masks in tqdm(dataloader):
            optimizer.zero_grad()
            batch_images = batch_images.cuda()
            batch_masks = batch_masks.cuda()
            # pdb.set_trace()
            with torch.set_grad_enabled(True):
                # pdb.set_trace()
                outputs = F.sigmoid(net(batch_images))
                # outputs = outputs / outputs.max()
                # batch_masks = batch_masks / batch_masks.max()
                loss = (outputs - batch_masks) *\
                        (outputs - batch_masks)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                
            train_running_loss += loss.item() * batch_images.size(0)
        
        print('loss: {}'.format(train_running_loss/len(dataset)))
        # if train_running_loss/len(dataset) < 0.1: pdb.set_trace()
            