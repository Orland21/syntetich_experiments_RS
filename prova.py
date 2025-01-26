import itertools
import torch

e=list(itertools.permutations([0,1,2]))
print(e)
print(range(6))
z=torch.rand(3,3)
print(z)
print(z[:,e[1]])