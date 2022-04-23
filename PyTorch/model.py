
# without pytorch

class shallow_neural_network():
  def __init__(self, num_input_features,num_hiddens):
    self.num_input_features = num_input_features
    self.num_hiddens = num_hiddens
    
    self.W1 = np.random.normal(size = (num_hiddens,num_input_features))
    self.b1 = np.random.normal(size = num_hiddens)
    self.W2 = np.random.normal(size = num_hiddens)
    self.b2 = np.random.normal(size = 1)
    
  def sigmoid(self,z)
    return 1/(1 + np.exp(-z))
  
  def predict(self,x): # forward pass
    z1 = np.matmul(self.W1,x) + self.b1
    a1 = np.tanh(z1)
    z2 = np.matmul(self.W2,a1) + self.b2
    a2 = self.sigmoid(z2)
    return a2, (z1,a1,z2,a2)
  
  

# with pytorch

class shallow_neural_network(nn.Module): # should be inherit nn.Module
  def __init__(self,num_input_features, num_hiddens):
    super().__init__()
    self.num_input_features = num_input_features
    self.num_hiddens = num_hiddens
    
    self.linear1 = nn.Linear(num_input_features, num_hiddens)
    self.linear2 = nn.Linear(num_hiddens,1)
    
    self.tanh = torch.nn.Tanh()
    self.sigmoid = torch.nn.Sigmoid()
    
  def forward(self,x):
    z1 = self.linear1(x)
    a1 = self.tanh(z1)
    z2 = self.linear2(a1)
    a2 = self.sigmoid(z2)
    
    return a2
