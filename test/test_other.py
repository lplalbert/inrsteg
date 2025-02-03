import torch
import torch.nn as nn
class FourierFeatMapping(nn.Module):
    def __init__(self, in_dim, map_scale=16, map_size=512, tunable=False):
        super().__init__()

        B = torch.normal(0., map_scale, size=(map_size//2, in_dim))

        if tunable:
            self.B = nn.Parameter(B, requires_grad=True)
        else:
            self.register_buffer('B', B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[0]

    @property
    def flops(self):
        return self.B.shape[0] * self.B.shape[1]

    def forward(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


model = FourierFeatMapping(3, map_scale=16, map_size=512, tunable=True)

x = torch.randn(10, 256, 3)
print(model(x).shape)