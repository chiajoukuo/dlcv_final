from SegNet import SegNet
from Pavements import Pavements
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

EPOCHS = 1
NUM_OF_CLASSES = 2

trainset = Pavements('./CamVid/train', './CamVid/train_labels')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True) #, num_workers=4)

weight = [1.0/NUM_OF_CLASSES]*NUM_OF_CLASSES

model = SegNet(in_chn=3, out_chn=NUM_OF_CLASSES, BN_momentum=0.5)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight))

cuda_available = torch.cuda.is_available()
if cuda_available:
  model.cuda()
  loss_fn.cuda()


for epoch in range(1, EPOCHS+1):
  print('Epock {}'.format(epoch), '\r')
  loss_sum = 0.0
  for i, data in enumerate(trainloader, 1):
    images, labels = data
    print(images.size(), labels.size())
    if cuda_available:
      images.cuda()
      labels.cuda()
    optimizer.zero_grad()
    output = model(images)
    print(i, output.size(), labels.size())
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()

    loss_sum += loss.item()
  
  print('Epoch {} loss {}'.format(epoch, loss))
torch.save({'epoch': EPOCHS, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, './model.pth.tar')

