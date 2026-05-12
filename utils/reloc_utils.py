from gaussian_hierarchy._C import compute_relocation
import torch
import math
N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n+1):
        binoms[n, k] = math.comb(n, k)
def compute_relocation_cuda(opacity_old, scale_old, N):
    return compute_relocation(opacity_old, scale_old, N.int(), binoms, N_max)