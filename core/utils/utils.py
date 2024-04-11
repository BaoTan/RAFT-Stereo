import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    # 首先从dims中获取图像的高度和宽度，然后计算需要填充的高度和宽度，使其能够被divis_by整除。
    # 根据mode的取值，如果是'sintel'模式，则计算四个方向上的填充值，分别为左、右、上、下；
    # 如果不是'sintel'模式，则只计算水平方向的填充值。最终将填充值存储在self._pad中以备后续使用。
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    # 在Python中，带有星号（*）前缀的参数（例如*inputs）表示接受任意数量的位置参数，并将它们作为一个元组传递给函数。
    def pad(self, *inputs):
        # 断言确保所有输入的张量维度都为4（通常表示为(batch_size, channels, height, width)）
        assert all((x.ndim == 4) for x in inputs)
        # 对每个输入张量进行填充操作。填充操作使用了PyTorch中的F.pad函数，通过传入输入张量和预先计算好的填充值self._pad，
        # 采用"replicate"模式进行填充（即复制边界像素进行填充）
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    if H > 1:
        ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1,1,N,N).to(input)
    output = F.conv2d(input.reshape(B*D,1,H,W), weights, padding=N//2)
    return output.view(B, D, H, W)