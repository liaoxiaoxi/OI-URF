import os

import cv2
import numpy as np
import torch
def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out
def check_dir(base):
    if os.path.isdir(base):
        pass
    else:
        os.makedirs(base)

def save_state_dir(network, save_model_dir):
    state_dict = network.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, save_model_dir)


def load_state_dir(network, ckpts, device):
    network.load_state_dict({k.replace('module.', ''): v for k, v in ckpts.items()})
    network.to(device)
    network.eval()


import torch
import numpy as np
import cv2

def save_img(x, save_dir):
    """
    支持 (H,W)、(C,H,W)、(B,C,H,W) 三种格式
    自动转为 uint8 并保存到 save_dir
    """
    with torch.no_grad():
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().clamp(0, 1)  # 保证数值范围在 [0,1]

        if x.ndim == 4:
            # (B,C,H,W) -> 取第一张
            x = x[0]

        if x.ndim == 3:
            # (C,H,W) -> (H,W,C)
            x = x.permute(1, 2, 0).numpy()

        elif x.ndim == 2:
            x = x.numpy()

        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        x = (x * 255.0).astype(np.uint8)

        # 灰度图保存时去掉多余维度
        if x.ndim == 3 and x.shape[2] == 1:
            x = x[:, :, 0]

        cv2.imwrite(save_dir, x)
