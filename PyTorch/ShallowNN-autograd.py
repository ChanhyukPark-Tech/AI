import numpy as np
import torch
x_seeds = np.array([(0,0),(1,0),(0,1),(1,1)],dtype=np.float32)
y_seeds = np.array([0,1,1,0])

N = 1000
idxs = np.random.randint(0,4,N)

X = x_seeds[idxs]
Y = y_seeds[idxs]

X += np.random.normal(scale= 0.25,size=X.shape)


class shallow_neural_network():
    def __init__(self,num_input_features,num_hiddens):
        self.num_input_features = num_input_features
        self.num_hiddens = num_hiddens

        self.W1 = torch.randn((num_hiddens,num_input_features),requires_grad=True)
        self.b1 = torch.randn(num_hiddens,requires_grad=True)
        self.W2 = torch.randn(num_hiddens,requires_grad=True)
        self.b2 = torch.randn(1,requires_grad=True)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def predict(self,x):
        z1 = torch.matmul(self.W1,x) + self.b1
        a1 = self.tanh(z1)
        z2 = torch.matmul(self.W2,a1) + self.b2
        a2 = self.sigmoid(z2)

        return a2

model = shallow_neural_network(2,4)

def train(X,Y,model,lr = 0.1):
    m = len(X)

    cost = 0.0

    for x,y in zip(X,Y):
        x_torch = torch.from_numpy(x)
        a2 = model.predict(x_torch)

        if y==1:
            loss = -torch.log(a2+0.0001)
        else:
            loss = -torch.log(1.00001-a2)

        loss.backward()
        cost += loss.item()

    with torch.no_grad():
        model.W1 -= lr * model.W1.grad/m    
        model.b1 -= lr * model.b1.grad/m    
        model.W2 -= lr * model.W2.grad/m    
        model.b2 -= lr * model.b2.grad/m    

    model.W1.requires_grad = True
    model.b1.requires_grad = True
    model.W2.requires_grad = True
    model.b2.requires_grad = True

    return cost/m


for epoch in range(100):
    cost = train(X,Y,model,1.0)
    if epoch % 10 == 0:
        print(epoch,cost)


print(model.predict(torch.Tensor((0,0))))
print(model.predict(torch.Tensor((0,1))))
print(model.predict(torch.Tensor((1,0))))
print(model.predict(torch.Tensor((1,1))))