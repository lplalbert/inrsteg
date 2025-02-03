import torch
import torch.nn as nn

from FastTools.module.common import LinearBlock
def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'relu':
        return nn.ReLU(inplace=True)
    else:
        return nn.Identity()
    
def get_norm(norm, num_features):
    # https://blog.csdn.net/weixin_43120238/article/details/110525369
    if norm.lower() == 'bn':
        return nn.BatchNorm1d(num_features)
    elif norm.lower() == 'in':
        return nn.InstanceNorm1d(num_features)
    elif norm.lower() == 'ln':
        return nn.LayerNorm(num_features)
    else:
        return nn.Identity()
    pass
class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1, bias=True, activation='relu', norm='ln'):
        super().__init__()
        self.act = get_activation(activation)
        self.net = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, groups=groups)
        self.norm = get_norm(norm, out_channels)
        
        self.norm_name = norm
        

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        if self.norm_name == 'ln':
            x = x.permute(0, 2, 1)
        x = self.norm(x)
        if self.norm_name == 'ln':
            x = x.permute(0, 2, 1)
        return self.act(x)


class ResMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', norm='bn'):
        super().__init__()
        self.conv1 = LinearBlock(in_channels, out_channels, activation=activation, norm=norm)
        self.conv2 = LinearBlock(out_channels, out_channels, activation=activation, norm=norm)
        if in_channels != out_channels:
            self.proj = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.proj = nn.Identity()
        
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + self.proj(x)
        pass

class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=3, activation='relu', norm='ln', final_bias=False):
        super().__init__()
        self.pre = ResMLPBlock(in_dim, hidden_dim, activation=activation, norm=norm)
        self.resmlp_blocks = nn.Sequential(*[ResMLPBlock(hidden_dim, hidden_dim, activation=activation, norm=norm) for _ in range(num_layers-1)])
        self.post = nn.Linear(hidden_dim, out_dim, bias=final_bias)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.resmlp_blocks(x)
        x = self.post(x)
        return x
    pass

if __name__ == "__main__":
    model = ResMLP(2, 3, 256)
    x = torch.randn(10, 30, 2)
    print(model(x).size())
    pass