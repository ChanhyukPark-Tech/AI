import torch

t = torch.Tensor([[1,2],[2,5],[5,10]])
y_0 = t.sum(0)
y_1 = t.sum(1)
y_2 = t.sum()
print(t + y_0)
print(t + y_2)


