import torch

x1 = torch.tensor([1.0],dtype=torch.float64,requires_grad=True)
x2 = torch.tensor([2.0],dtype=torch.float64,requires_grad=True)

c = x1 + x2

c.backward()
print(x1.grad)
print(x2.grad)
print(c)



