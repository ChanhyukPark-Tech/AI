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


def mnist_image_augment(mnist, scale=True, rotate=True, shear=True, colour=True, gaussian=True, invert=True):
    l = len(mnist)

    SC = np.random.normal(1, 0.3, size=l) # scale
    SH = np.random.normal(0, 1, size=(l, 3, 2)) # shear
    R = np.random.normal(0, 20, size=l) # rotate
    C = np.random.randint(22, size=l) # colour
    G = np.random.randint(30, size=l) # noise
    I = np.random.randint(2, size=l) # invert

    augmented = []

    for i, t in enumerate(mnist):
        X, y = t[0], t[1]
        X = X.numpy()
        X = (np.reshape(X, (28, 28, 1)) * 255).astype(np.uint8)

        if scale or rotate:
            if scale:
                sc = SC[i] if SC[i] >= 0 else -SC[i]
            else:
                sc = 1
            r = R[i] if rotate else 0

            M = cv2.getRotationMatrix2D((14, 14), r, sc)
            X = cv2.warpAffine(X, M, (28, 28))
        
        if shear:
            pts1 = np.float32([[4, 4], [4, 24], [24, 4]])
            pts2 = np.float32([[4+SH[i][0][0], 4+SH[i][0][1]], [4+SH[i][1][0], 24+SH[i][1][1]], [24+SH[i][2][0], 4+SH[i][2][1]]])
            
            M = cv2.getAffineTransform(pts1, pts2)
            X = cv2.warpAffine(X, M, (28, 28))

        if colour:
            X = cv2.applyColorMap(X, C[i])
        
        if gaussian:
            g = G[i]/100 if G[i] > 0 else - G[i]/100
            gauss = np.random.normal(0, g**0.5, X.shape)
            X = (X + gauss).astype(np.uint8)

        if invert:
            X = cv2.bitwise_not(X)

        recover = (np.reshape(X, (3, 28, 28)) / 255).astype(np.float32)   
        X = torch.from_numpy(recover)
        augmented.append([X, y])
    
    return augmented

augmented = mnist_image_augment(train_data_mnist)
train_set,val_set=torch.utils.data.random_split(augmented,[50000,10000])



#preview result는 생략
def mnist_preview(mnist, augmented, n=5):
    for i in range(1000, 1000+n):
        origin = train_data_mnist[i][0].numpy()
        origin = (np.reshape(origin, (28, 28, 1)) * 255).astype(np.uint8)

        augment = augmented[i][0].numpy()
        augment = (np.reshape(augment, (28, 28, 3)) * 255).astype(np.uint8)

        plt.figure()
        f, axarr = plt.subplots(1,2) 
        print(axarr)
        axarr[0].imshow(origin, cmap='gray')
        axarr[1].imshow(augment)
        
mnist_preview(train_data_mnist,augmented)


# model declaration

learning_rate=1e-3
batch_size=64

train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
dev_loader=torch.utils.data.DataLoader(val_set,batch_size=batch_size)
test_loader=torch.utils.data.DataLoader(augmented,batch_size=batch_size)


class RobustModel(nn.Module):
    """
    TODO: Implement your model
    """
    def __init__(self):
        super(RobustModel, self).__init__()
        self.keep_prob=0.5
        
        
        n_channels_1=6
        n_channels_2=16
        
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(3,n_channels_1,kernel_size=3,stride=1),
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
            x = x.permute(0,3,2,1)

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

training_epoch=5

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
    test(dev_loader,model)


print("이제 저장할게")
torch.save(model.state_dict(),'model.pt')





def imshow(img):
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
dataiter=iter(test_loader)
images,labels=dataiter.next()

imshow(torchvision.utils.make_grid(images,nrow=batch_size))
print('GroudTruth')
print('   '+'  '.join('%3s'%label.item() for label in labels))

outputs=model(images)
_, predicted=torch.max(outputs,1)
print('prediction')
print('   '+'  '.join('%3s'%label.item() for label in predicted))

#accuracy측정
n_predict=0
n_correct=0

for data in test_loader:
    inputs,labels=data
    outputs=model(inputs)
    _,predicted=torch.max(outputs,1)
    
    n_predict+=len(predicted)
    n_correct+=(labels==predicted).sum()
    
print(f"{n_correct}/{n_predict}")
