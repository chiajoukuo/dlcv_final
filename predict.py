import SegNet
from Pavements import Pavements

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product

NUM_OF_CLASSES = 4
MODEL_NAME = './models/model_10.pth.tar'

def build_color_map():
    # assumes no. of classes to be <= 64
    color_map = torch.tensor(list(product([63, 127, 191, 255], repeat=3)))

    print()
    print("Map of class to color: ")
    for class_ind, color in enumerate(color_map):
        print("Class: {}, RGB Color: {}".format(class_ind + 1, color))

    print()

    return color_map

cuda_available = torch.cuda.is_available()
model = SegNet.SegNet(in_chn=3, out_chn=NUM_OF_CLASSES, BN_momentum=0.5)

if cuda_available:
  model.cuda()

model.eval()

checkpoint = torch.load(MODEL_NAME)
epoch = checkpoint['epoch']
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
print("Checkpoint is loaded at {} | Epochs: {}".format(MODEL_NAME, epoch))


dataset = Pavements('./CamVid/test', './CamVid/test_labels')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
color_map = build_color_map()

for i, data in enumerate(dataloader):
  images, labels = data
  labels = labels.type(torch.long)

  if cuda_available:
    images = images.cuda()
    labels = labels.cuda()
  
  result = model(images)
  result = torch.argmax(result, dim=1).type(torch.long)

  for j in range(result.shape[0]):
    input_image = images[j]
    label_image = color_map[labels[i]].permute(2, 0, 1).to(torch.float).dic(255.0)
    result_image = color_map[result[i]].permute(2, 0, 1).to(torch.float).dic(255.0)

    if cuda_available:
      input_image =input_image.cuda()
      label_image = label_image.cuda()
      result_image = result_image.cuda()
    
    folder = ['input', 'label', 'result']
    for k, img in enumerate([input_image, label_image, result_image]):
      path = './results/{}/{}_{}.png'.format(folder, i, j)
      save_image(img, path)
    