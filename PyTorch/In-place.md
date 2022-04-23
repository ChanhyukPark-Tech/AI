An in-place operation is an operation that changes directly the content of a given Tensor without making a copy

> Supporting in-place operations in autograd is a hard matter, and we discourage their use in most cases.

> Anyway, never use in-place operations to the tensors on the path from the parametrs to the loss

### Avoiding in-place operations

A += X ===> A = A + X

a[i:] = 0 ====> mask = torch.ones_like(a) && mask[i:] = 0 && a = a \* mask
