import torch.nn as nn
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torchvision
from torchvision import datasets, transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_data_mnist = datasets.MNIST('./datasets', train=True, download=True, transform=transforms.ToTensor())
test_data_mnist = datasets.MNIST('./datasets',train=True,download=True, transform=transforms.ToTensor())

# model declaration

learning_rate=1e-3
batch_size=16

train_loader=torch.utils.data.DataLoader(train_data_mnist,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data_mnist,batch_size=batch_size)


class RobustModel(nn.Module):
    """
    TODO: Implement your model
    """
    def __init__(self):
        super(RobustModel, self).__init__()
        self.keep_prob=0.75
        
        
        n_channels_1=10
        n_channels_2=20
        
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,n_channels_1,kernel_size=3,stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(n_channels_1,n_channels_2,kernel_size=5,stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.fc3=torch.nn.Linear(4*4*n_channels_2,120,bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        
        self.layer3=torch.nn.Sequential(
            self.fc3,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob)
        )
        
        self.fc4=torch.nn.Linear(120,80,bias=True)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
        self.layer4=torch.nn.Sequential(
            self.fc4,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob)
        )
        
        self.fc5=torch.nn.Linear(80,10,bias=True)
        torch.nn.init.xavier_uniform_(self.fc5.weight)


    def forward(self,x):
        if x.shape[1] == 28:
            x = np.transpose(x, (0, 3, 1, 2))
            x = x.mean(dim=1)
            x = x.unsqueeze(1)

        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0),-1)#fc들어가기전 Flatten
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.fc5(out)
        return out


model=RobustModel()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

def test(data_loader,model):
    model.eval()
    n_predict=0
    n_correct=0
    with torch.no_grad():
        for X,Y in tqdm.tqdm(data_loader,desc='data_loader'):
            y_hat=model(X)
            y_hat.argmax()
            
            _,predicted=torch.max(y_hat,1)
            
            n_predict += len(predicted)
            n_correct += (Y==predicted).sum()
            
    accuracy=int(n_correct)/n_predict #n_correct가 tensor형태로 읽어져서 accuracy계속0으로 출력되어 형변환
    print(f"Accuracy:{accuracy}()")

training_epoch=12

for epoch in range(training_epoch):
    #model.train()
    cost=0
    n_batches=0
    for X,Y in tqdm.tqdm(train_loader, desc='train_loader'):#module object is nor callable 오류나서 desc추가
        optimizer.zero_grad()
        y_hat=model(X)
        loss=criterion(y_hat,Y)
        loss.backward()
        optimizer.step()
        
        cost+=loss.item()
        n_batches+=1
    cost/=n_batches
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch+1,cost))
    print("Dev")


torch.save(model.state_dict(),'model.pt')





