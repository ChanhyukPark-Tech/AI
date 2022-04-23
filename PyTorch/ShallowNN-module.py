import numpy as np
import torch
from torch import nn
from torch import optim
x_seeds = np.array([(0,0),(1,0),(0,1),(1,1)],dtype=np.float32)
y_seeds = np.array([0,1,1,0])

N = 1000
idxs = np.random.randint(0,4,N)

X = x_seeds[idxs]
Y = y_seeds[idxs]

X += np.random.normal(scale= 0.25,size=X.shape)


class shallow_neural_network(nn.Module):
    def __init__(self,num_input_features,num_hiddens):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_hiddens = num_hiddens

        self.linear1 = nn.Linear(num_input_features,num_hiddens)
        self.linear2 = nn.Linear(num_hiddens,1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        z1 = self.linear1(x)
        a1 = self.tanh(z1)
        z2 = self.linear2(a1)
        a2 = self.sigmoid(z2)

        return a2



model = shallow_neural_network(2,3)
optimizer = optim.SGD(model.parameters(),lr=1.0)
loss = nn.BCELoss()


for epoch in range(100):
    optimizer.zero_grad()

    cost = 0.0
    for x,y in zip(X,Y):
        x_torch = torch.from_numpy(x)
        y_torch = torch.FloatTensor([y])

        y_hat = model(x_torch)

        loss_val = loss(y_hat,y_torch)
        cost += loss_val
    cost = cost / len(X)
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch,cost)

for x,y in zip(x_seeds,y_seeds):
    print(x)
    x_torch = torch.from_numpy(x)
    y_hat = model(x_torch)
    print(y,y_hat.item())

