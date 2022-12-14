from SegNet import SegNet
from Pavements import Pavements
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

EPOCHS = 10
NUM_OF_CLASSES = 4
BATCH_SIZE = 4
LEARNING_RATE = 0.005

trainset = Pavements('./CamVid/train', './CamVid/train_labels')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True) #, num_workers=4)

weight = torch.tensor([1.0/NUM_OF_CLASSES]*NUM_OF_CLASSES)

model = SegNet(in_chn=3, out_chn=NUM_OF_CLASSES, BN_momentum=0.5)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


cuda_available = torch.cuda.is_available()
if cuda_available:
  model = model.cuda()
  weight = weight.cuda()
  loss_fn = nn.CrossEntropyLoss(weight=weight).cuda()
else:
  loss_fn = nn.CrossEntropyLoss(weight=weight)
numsss = len(trainset)

for epoch in range(1, EPOCHS+1):
  print('Epock {}'.format(epoch))
  loss_sum = 0.0
  for i, data in enumerate(trainloader, 1):
    images, labels = data
    if cuda_available:
      images = images.cuda()
      labels = labels.cuda()
    optimizer.zero_grad()
    output = model(images)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()

    loss_sum += loss.item()
    print('{} / {}'.format(i, numsss), '\r', end='')
  print('Epoch {} loss {}'.format(epoch, loss))
  torch.save({'epoch': EPOCHS, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, './models/model_{}.pth.tar'.format(epoch))

