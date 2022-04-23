num_epochs = 100
lr = 1.0
num_hinddens  = 3

model = shallow_neural_network(2,num_hiddens)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss = nn.BCELoss()

for epoch in range(num_epochs):
  optimizer.zero_grad() #init

  
  cost = 0.0
  for x,y in zip(X,Y):
    x_torch = torch.from_numpy(x)
    y_torch = torch.FloatTensor([y])
    
    y_hat = model(x_torch)
    
    loss_val = loss(y_hat, y_torch)
    cost += loss_val
    
  cost = cost / len(X)
  cost.backward() # compute
  optimizer.step() # update
