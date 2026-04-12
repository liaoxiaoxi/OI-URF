import os
import time
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.cm as cm
import cv2

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import kornia

import args
from loss import loss as Loss
from model import model
from utils import utils
from utils.utils import save_img

from utils.utils import RGB2YCrCb, YCbCr2RGB

# =========================================================
# Basic setup
# =========================================================
model_name = "OI-URF"
device_id = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = device_id
device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")

seed = getattr(args.args, "seed", 0)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.benchmark = True

CKPT_PATH = getattr(args.args, "test_model_path", "")
# TEST_MODE = getattr(args.args, "test_mode", "synthetic_eval")   # synthetic_eval / real_infer
TEST_MODE = getattr(args.args, "test_mode", "synthetic_eval")
TEST_BATCH_SIZE = getattr(args.args, "test_batch_size", 1)
EVAL_EPOCH_FOR_LOSS = getattr(args.args, "eval_epoch_for_loss", getattr(args.args, "Epoch", 100) - 1)

VIS_TEST_DIR = getattr(args.args, "vis_test_dir", getattr(args.args, "vis_train_dir", ""))
IR_TEST_DIR = getattr(args.args, "ir_test_dir", getattr(args.args, "ir_train_dir", ""))

assert CKPT_PATH != "", "Please set args.args.test_model_path"
assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
assert VIS_TEST_DIR != "" and IR_TEST_DIR != "", "Please set vis_test_dir / ir_test_dir"

now = int(time.time())
nowTime = time.strftime("%Y%m%d_%H-%M-%S", time.localtime(now))
ckpt_tag = Path(CKPT_PATH).stem

ROOT_SAVE_DIR = getattr(args.args, "test_save_dir", "./test_results")
save_root_dir = os.path.join(ROOT_SAVE_DIR, f"{nowTime}_{model_name}_{ckpt_tag}_{TEST_MODE}")
save_vis_dir = os.path.join(save_root_dir, "paper_vis")
save_log_dir = os.path.join(save_root_dir, "logs")

utils.check_dir(save_root_dir)
utils.check_dir(save_vis_dir)
utils.check_dir(save_log_dir)

# =========================================================
# CSV logger
# =========================================================
CSV_LOG = os.path.join(save_log_dir, "test_log.csv")
with open(CSV_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "name", "mode",
        "loss", "grid_ms", "grid_full", "photo", "smooth", "fold",
        "epe", "flowmag", "disp_pred", "disp_gt",
        "l1_grad", "conf_pred_mean",
        "gate_mean", "gate_entropy",
        "router_ent_v", "router_ent_i",
        "loss_cm", "loss_same", "loss_fusion"
    ])


def append_csv(row):
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def grid_inbounds_mask_soft(grid, pad_px=2.5):
    """
    grid: [B,H,W,2]
    return: [B,1,H,W] in [0,1]
    在边界附近给软过渡，避免黑边锯齿
    """
    B, H, W, _ = grid.shape

    # pixel -> normalized margin
    mx = 2.0 * pad_px / max(W - 1, 1)
    my = 2.0 * pad_px / max(H - 1, 1)

    gx = grid[..., 0].abs()
    gy = grid[..., 1].abs()

    ax = ((1.0 + mx - gx) / mx).clamp(0.0, 1.0)
    ay = ((1.0 + my - gy) / my).clamp(0.0, 1.0)

    return (ax * ay).unsqueeze(1)


# =========================================================
# Helpers
# =========================================================
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def grid_inbounds_mask_hard_margin(grid, border_px=8):
    """
    grid: [B,H,W,2]
    return: [B,1,H,W], 1=有效, 0=无效
    把靠近图像边界 border_px 像素以内的区域也视为无效
    """
    B, H, W, _ = grid.shape

    mx = 2.0 * border_px / max(W - 1, 1)
    my = 2.0 * border_px / max(H - 1, 1)

    gx = grid[..., 0]
    gy = grid[..., 1]

    m = (
            (gx >= -1.0 + mx) & (gx <= 1.0 - mx) &
            (gy >= -1.0 + my) & (gy <= 1.0 - my)
    )
    return m.float().unsqueeze(1)


def grad_xy(feat):
    dx = feat[:, :, :, 1:] - feat[:, :, :, :-1]
    dy = feat[:, :, 1:, :] - feat[:, :, :-1, :]
    return dx, dy


def build_identity_grid(B, H, W, device, dtype):
    g = kornia.utils.create_meshgrid(H, W, normalized_coordinates=True, device=device)
    return g.to(dtype=dtype).repeat(B, 1, 1, 1)


def make_checkerboard_pair(a, b, tile=16):
    """
    a, b: [B,1,H,W]
    return: [B,1,H,W]
    """
    B, _, H, W = a.shape
    yy = torch.arange(H, device=a.device).view(1, 1, H, 1) // tile
    xx = torch.arange(W, device=a.device).view(1, 1, 1, W) // tile
    m = ((yy + xx) % 2).float()
    return a * m + b * (1.0 - m)


def invert_dense_grid(grid_pull, iters=12, clamp_each_iter=True, final_clamp=True):
    B, H, W, _ = grid_pull.shape
    base = build_identity_grid(B, H, W, grid_pull.device, grid_pull.dtype)
    y = base
    x = y.clone()
    grid_img = grid_pull.permute(0, 3, 1, 2).contiguous()

    for _ in range(iters):
        fx = F.grid_sample(
            grid_img, x,
            align_corners=True,
            mode="bilinear",
            padding_mode="border"
        )
        fx = fx.permute(0, 2, 3, 1).contiguous()
        x = x + (y - fx)
        if clamp_each_iter:
            x = x.clamp(-1, 1)

    if final_clamp:
        x = x.clamp(-1, 1)

    return x


def grid_downsample(grid, scale):
    g = grid.permute(0, 3, 1, 2)
    g = F.avg_pool2d(g, kernel_size=scale, stride=scale, ceil_mode=True)
    return g.permute(0, 2, 3, 1).contiguous()


def folding_penalty_grid(grid):
    gx = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    gy = grid[:, 1:, :, :] - grid[:, :-1, :, :]
    gx = gx[:, :-1, :, :]
    gy = gy[:, :, :-1, :]
    a = gx[..., 0]
    b = gy[..., 0]
    c = gx[..., 1]
    d = gy[..., 1]
    det = a * d - b * c
    return F.relu(-det).mean()


def flow_smooth_loss_from_grid(grid):
    B, H, W, _ = grid.shape
    idg = build_identity_grid(B, H, W, grid.device, grid.dtype)
    flow = (grid - idg).permute(0, 3, 1, 2).contiguous()
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return charbonnier(dx).mean() + charbonnier(dy).mean()


def rgb_to_ycbcr_tensor(x):
    return RGB2YCrCb(x)


def ycbcr_to_rgb_tensor(y, c1, c2):
    return YCbCr2RGB(y, c1, c2)


def gray_to_3ch(x):
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def cosine_loss(a, b, eps=1e-6):
    a = F.normalize(a, dim=1, eps=eps)
    b = F.normalize(b, dim=1, eps=eps)
    return 1.0 - (a * b).sum(1).mean()


# =========================================================
# Same schedule weights as train
# =========================================================
REG_FIRST_EPOCHS = getattr(args.args, "reg_first_epochs", 0)


def w_grid_ms(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 2.0
    if epoch < 60:
        return 1.0
    return 0.8


def w_grid_full(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.8
    if epoch < 60:
        return 0.5
    return 0.3


def w_photo(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.30
    if epoch < 60:
        return 0.20
    return 0.15


def w_smooth(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.08
    return 0.05


def w_fold(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.08
    if epoch < 80:
        return 0.12
    return 0.08


def w_conf(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.0
    if epoch < REG_FIRST_EPOCHS + 10:
        return 0.01
    return 0.03


def w_cm(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.0
    if epoch < REG_FIRST_EPOCHS + 10:
        return 0.01
    if epoch < REG_FIRST_EPOCHS + 40:
        return 0.03
    return 0.05


def w_gate(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.0
    if epoch < REG_FIRST_EPOCHS + 10:
        return 0.002
    return 0.01


def w_fuse(epoch):
    return 0.0 if epoch < REG_FIRST_EPOCHS else 1.0


def w_same(epoch):
    if epoch < REG_FIRST_EPOCHS:
        return 0.0
    return 0.4


# =========================================================
# Visualization helpers
# =========================================================
def to_uint8(x):
    x = x.detach().float().cpu().numpy()
    x = np.clip(x, 0, 1)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_gray_tensor(x, path):
    arr = to_uint8(x.squeeze())
    Image.fromarray(arr).save(path)


def save_color_tensor(x, path):
    if x.dim() == 4:
        x = x[0]
    arr = x.detach().float().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _tensor_to_heatmap_2d(x):
    """
    Accepts:
      [H,W]
      [1,H,W]
      [C,H,W]
      [1,C,H,W]
    Returns:
      2D numpy array [H,W]
    """
    x = x.detach().float().cpu()

    if x.dim() == 4:
        x = x[0]  # [C,H,W]

    if x.dim() == 3:
        # [C,H,W]
        if x.size(0) == 1:
            x = x[0]
        else:
            x = x.abs().mean(dim=0)

    elif x.dim() == 2:
        pass
    else:
        x = x.squeeze()
        if x.dim() == 2:
            pass
        elif x.dim() == 3:
            if x.size(0) == 1:
                x = x[0]
            else:
                x = x.abs().mean(dim=0)
        else:
            raise ValueError(f"Unsupported heatmap tensor shape: {tuple(x.shape)}")

    return x.numpy()


def save_heatmap_tensor(x, path, cmap="turbo"):
    x = _tensor_to_heatmap_2d(x)

    vmin = np.percentile(x, 1)
    vmax = np.percentile(x, 99)
    x = np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

    colormap = cm.get_cmap(cmap)
    rgb = colormap(x)[..., :3]
    rgb = (rgb * 255).astype(np.uint8)

    Image.fromarray(rgb).save(path)


def save_heatmap_tensor_pair(x, y, path_x, path_y, cmap="turbo"):
    def to_rgb(arr):
        arr = _tensor_to_heatmap_2d(arr)
        vmin = np.percentile(arr, 1)
        vmax = np.percentile(arr, 99)
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        colormap = cm.get_cmap(cmap)
        rgb = colormap(arr)[..., :3]
        return (rgb * 255).astype(np.uint8)

    Image.fromarray(to_rgb(x)).save(path_x)
    Image.fromarray(to_rgb(y)).save(path_y)


def save_overlay_map(img, heat, path, cmap="turbo", alpha=0.55):
    img = img.detach().cpu().squeeze().numpy()
    heat = heat.detach().cpu().squeeze().numpy()
    H, W = img.shape
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    colormap = cm.get_cmap(cmap)
    heat_rgb = colormap(heat)[..., :3]
    img_rgb = np.stack([img, img, img], axis=-1)

    overlay = (1 - alpha) * img_rgb + alpha * heat_rgb
    overlay = (overlay * 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def flow_to_rgb(flow_hw2):
    f = flow_hw2.detach().float().cpu().numpy()
    fx = f[..., 0]
    fy = f[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)

    hsv = np.zeros((*fx.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 1] = np.clip(mag / (mag.max() + 1e-8), 0, 1)
    hsv[..., 2] = 1.0

    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h).astype(np.int32)
    ff = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * ff)
    t = v * (1.0 - s * (1.0 - ff))
    i = i % 6

    rgb = np.zeros_like(hsv)
    conds = [
        (i == 0, np.stack([v, t, p], axis=-1)),
        (i == 1, np.stack([q, v, p], axis=-1)),
        (i == 2, np.stack([p, v, t], axis=-1)),
        (i == 3, np.stack([p, q, v], axis=-1)),
        (i == 4, np.stack([t, p, v], axis=-1)),
        (i == 5, np.stack([v, p, q], axis=-1)),
    ]
    for mask, val in conds:
        rgb[mask] = val[mask]

    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def flow_to_rgb_with_max(flow_hw2, mag_max=None, eps=1e-8):
    """
    flow_hw2: [H,W,2], normalized grid flow
    hue: direction
    saturation: magnitude / mag_max
    value: 1
    """
    f = flow_hw2.detach().float().cpu().numpy()
    fx = f[..., 0]
    fy = f[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)

    if mag_max is None:
        mag_max = float(mag.max())

    hsv = np.zeros((*fx.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)  # direction
    hsv[..., 1] = np.clip(mag / (mag_max + eps), 0, 1)  # magnitude
    hsv[..., 2] = 1.0

    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h).astype(np.int32)
    ff = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * ff)
    t = v * (1.0 - s * (1.0 - ff))
    i = i % 6

    rgb = np.zeros_like(hsv)
    conds = [
        (i == 0, np.stack([v, t, p], axis=-1)),
        (i == 1, np.stack([q, v, p], axis=-1)),
        (i == 2, np.stack([p, v, t], axis=-1)),
        (i == 3, np.stack([p, q, v], axis=-1)),
        (i == 4, np.stack([t, p, v], axis=-1)),
        (i == 5, np.stack([v, p, q], axis=-1)),
    ]
    for mask, val in conds:
        rgb[mask] = val[mask]

    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def save_router_hist(w_patch, path):
    w = w_patch[0].detach().float().cpu().numpy()
    hist = w.mean(axis=(1, 2))
    hist = hist / (hist.sum() + 1e-8)
    np.savetxt(path, hist, fmt="%.6f")


def smooth_mask(mask, ksize=5):
    """
    mask: [B,1,H,W] in [0,1]
    轻微平滑边界，让黑边更自然
    """
    pad = ksize // 2
    mask = F.avg_pool2d(mask, kernel_size=ksize, stride=1, padding=pad)
    return mask.clamp(0.0, 1.0)


# =========================================================
# Dataset
# =========================================================
class TestDataset(data.Dataset):
    def __init__(self, vis_dir, ir_dir, rgb_transform, gray_transform):
        self.rgb_transform = rgb_transform
        self.gray_transform = gray_transform

        vis = {p.name: p for p in Path(vis_dir).glob("*")}
        ir = {p.name: p for p in Path(ir_dir).glob("*")}
        names = sorted(vis.keys() & ir.keys())

        self.names = names
        self.vis_paths = [vis[n] for n in names]
        self.ir_paths = [ir[n] for n in names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        vis_rgb = Image.open(str(self.vis_paths[idx])).convert("RGB")
        ir_img = Image.open(str(self.ir_paths[idx])).convert("L")
        return self.rgb_transform(vis_rgb), self.gray_transform(ir_img), name


rgb_tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.args.img_size, args.args.img_size]),
    torchvision.transforms.ToTensor()
])

gray_tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.args.img_size, args.args.img_size]),
    torchvision.transforms.ToTensor()
])

dataset = TestDataset(VIS_TEST_DIR, IR_TEST_DIR, rgb_tf, gray_tf)
data_iter = data.DataLoader(dataset=dataset, shuffle=False, batch_size=TEST_BATCH_SIZE, num_workers=0)

print(f"Test samples: {len(dataset)}")
print(f"Mode: {TEST_MODE}")
print(f"Checkpoint: {CKPT_PATH}")
print(f"Save dir: {save_root_dir}")

# =========================================================
# Loss helpers
# =========================================================
Lgrad = Loss.L_Grad().to(device)
CC = Loss.CorrelationCoefficient().to(device)

# =========================================================
# Build modules
# =========================================================
FE = model.FE().to(device)
vis_MSFE = model.MSFE().to(device)
ir_MSFE = model.MSFE().to(device)
FD = model.FD().to(device)


OBRT = model.OperatorBankSharedExperts(
    C=8, patch_size=4, num_experts=4,
    share_ratio=0.0, tau_router=0.7,
    expert_kernel=3,
    gate_mode="entropy", gate_temp=0.8,
    gate_floor=0.05,
).to(device)

ImageDeformation = model.ImageTransform(
    max_deg=13,
    max_trans_px=13,
    max_scale=0,  # 这里保留也没事，但当前随机缩放主要由 scale_min/max 控制
    et_kernel=61,
    et_sigma=20.0,
    et_alpha_px=3,

    fixed_scale=None,  # None = 随机缩放
    fixed_elastic=True,
    fixed_elastic_seed=1234,

    scale_min=0.97,
    scale_max=1.03,
).to(device)
Fusion = model.Fusion().to(device)
denoise_reg = model.DenoiseReg(base=32, T=15, max_disp=0.2, step_size=0.08).to(device)

# =========================================================
# Load checkpoint
# =========================================================
print(f"\nLoading checkpoint from: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)

FE.load_state_dict(ckpt["FE"])
ir_MSFE.load_state_dict(ckpt["ir_MSFE"])
vis_MSFE.load_state_dict(ckpt["vis_MSFE"])
OBRT.load_state_dict(ckpt["obrt"])
Fusion.load_state_dict(ckpt["fusion"])
FD.load_state_dict(ckpt["FD"])

if "denoise_reg" in ckpt:
    denoise_reg.load_state_dict(ckpt["denoise_reg"])

FE.eval()
vis_MSFE.eval()
ir_MSFE.eval()
FD.eval()
OBRT.eval()
Fusion.eval()
denoise_reg.eval()
ImageDeformation.eval()


def save_feature_gray(x, path, qmin=1, qmax=99):
    x = x.detach().float().cpu().squeeze().numpy()
    vmin = np.percentile(x, qmin)
    vmax = np.percentile(x, qmax)
    x = np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(x, mode="L").save(path)


# =========================================================
# Main test
# =========================================================
@torch.no_grad()
def test():
    all_loss = []
    all_grid_ms = []
    all_grid_full = []
    all_photo = []
    all_smooth = []
    all_fold = []
    all_epe = []
    all_flowmag = []
    all_disp_pred = []
    all_disp_gt = []
    all_l1_grad = []
    all_conf = []
    all_gate = []
    all_gate_entropy = []
    all_ent_v = []
    all_ent_i = []
    all_loss_cm = []
    all_loss_same = []
    all_loss_fusion = []

    for step, batch in enumerate(tqdm(data_iter)):
        vis_rgb = batch[0].to(device)  # [B,3,H,W]
        ir = batch[1].to(device)  # [B,1,H,W]
        names = batch[2]

        vis, vis_c1, vis_c2 = rgb_to_ycbcr_tensor(vis_rgb)
        B = vis.size(0)

        # -------------------------------------------------
        # mode branch
        # -------------------------------------------------
        if TEST_MODE == "synthetic_eval":
            grid_pull = ImageDeformation.generate_grid(vis)
            vis_rgb_d = F.grid_sample(vis_rgb, grid_pull, align_corners=True, mode="bilinear", padding_mode="border")
            ir_d = F.grid_sample(ir, grid_pull, align_corners=True, mode="bilinear", padding_mode="border")

            # 黑边版 deformed IR
            ir_d_heibianban = F.grid_sample(
                ir, grid_pull,
                align_corners=True, mode="bilinear", padding_mode="zeros"
            )

            valid_src = F.grid_sample(
                torch.ones_like(ir), grid_pull,
                align_corners=True,
                mode="nearest",
                padding_mode="zeros"
            )

            vis_d, visd_c1, visd_c2 = rgb_to_ycbcr_tensor(vis_rgb_d)

            # GT逆网格只保留稳定版
            grid_inv_gt = invert_dense_grid(
                grid_pull, iters=12,
                clamp_each_iter=True, final_clamp=True
            )

            grid_inv_gt_raw = invert_dense_grid(
                grid_pull, iters=30,
                clamp_each_iter=False, final_clamp=False
            )

        elif TEST_MODE == "real_infer":
            vis_rgb_d = vis_rgb
            ir_d = ir
            vis_d, visd_c1, visd_c2 = vis, vis_c1, vis_c2
            grid_inv_gt = None
            grid_inv_gt_raw = None

            # 占位，后面再生成 blackfusion
            ir_d_heibianban = None
            valid_src = None

        else:
            raise ValueError(f"Unknown TEST_MODE: {TEST_MODE}")

        # -------------------------------------------------
        # backbone features
        # -------------------------------------------------
        vis_1 = FE(vis)
        ir_1 = FE(ir)
        visd_1 = FE(vis_d)
        ird_1 = FE(ir_d)

        vis_fe = vis_MSFE(vis_1)
        ir_fe = ir_MSFE(ir_1)
        visd_fe = vis_MSFE(visd_1)
        ird_fe = ir_MSFE(ird_1)

        fusion_image_1, _ = FD(vis_fe + ir_fe)
        fusiond_image_1, _ = FD(visd_fe + ird_fe)
        # -------------------------------------------------
        # direct fusion without OBRT / registration
        # -------------------------------------------------
        fusion_image_noreg = Fusion(vis_fe, ir_fe)
        fusion_rgb_noreg = ycbcr_to_rgb_tensor(fusion_image_noreg.clamp(0, 1), vis_c1, vis_c2)
        # -------------------------------------------------
        # direct fusion with deformed IR feature (no warp)
        # vis_fe + ird_fe，故意不做配准，直接送入 fusion_module
        # -------------------------------------------------
        # -------------------------------------------------
        # direct fusion with black-border deformed IR feature (no warp)
        # 先用 ir_d_heibianban 提特征，再和 vis_fe 直接融合
        # -------------------------------------------------
        if ir_d_heibianban is not None:
            ird_heibianban_1 = FE(ir_d_heibianban)
            ird_heibianban_fe = ir_MSFE(ird_heibianban_1)
        else:
            # real_infer 下没有 synthetic 生成的黑边版时，退化为 ird_fe
            ird_heibianban_fe = ird_fe

        fusion_image_ir_d_noreg = Fusion(vis_fe, ird_heibianban_fe)
        fusion_rgb_ir_d_noreg = ycbcr_to_rgb_tensor(
            fusion_image_ir_d_noreg.clamp(0, 1),
            vis_c1, vis_c2
        )
        fusion_rgb_ir_d_noreg_black = None

        if ir_d_heibianban is not None:
            ir_border_mask = (ir_d_heibianban > (1.0 / 255.0)).float()  # [B,1,H,W]

            ir_border_mask = F.interpolate(
                ir_border_mask, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            ir_border_mask = F.interpolate(
                ir_border_mask, size=fusion_rgb_ir_d_noreg.shape[-2:],
                mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)

            ir_border_mask = kornia.filters.gaussian_blur2d(
                ir_border_mask, (5, 5), (0.8, 0.8)
            ).clamp(0.0, 1.0)

            fusion_rgb_ir_d_noreg_black = fusion_rgb_ir_d_noreg * ir_border_mask
        # -------------------------------------------------
        # direct fusion without registration + input-IR black border
        # real_infer 下：哪里输入 ir 是黑的，fusion_rgb_noregblack 也强制为黑
        # -------------------------------------------------
        # fusion_rgb_noregblack = None
        # if TEST_MODE == "real_infer":
        #     # 黑边 mask：1=有效区域，0=输入 ir 的黑色区域
        #     # 这里给一个很小阈值，避免 resize 后纯黑边界出现极小非零值
        #     ir_valid_mask = (ir > (1.0 / 255.0)).float()   # [B,1,H,W]

        #     # 广播到 3 通道，直接把 fusion_rgb_noreg 的对应位置压成黑色
        #     fusion_rgb_noregblack = fusion_rgb_noreg * ir_valid_mask
        # -------------------------------------------------
        # OBRT
        # -------------------------------------------------
        VISDP_vis_f, aux_v = OBRT(vis_fe, modality="vis", return_aux=True)
        IRDP_ird_f, aux_id = OBRT(ird_fe, modality="ir", return_aux=True)
        VISDP_visd_f, aux_vd = OBRT(visd_fe, modality="vis", return_aux=True)
        IRDP_ir_f, aux_i = OBRT(ir_fe, modality="ir", return_aux=True)

        F_fix = VISDP_vis_f
        F_mov = IRDP_ird_f
        # -------------------------------------------------
        # DenoiseReg
        # -------------------------------------------------
        out_reg = denoise_reg(F_fix, F_mov, return_steps=True)

        # 可视化用原始预测网格（不 clamp）
        grid_pred_raw = out_reg["grid"]

        # 训练/评估/对齐仍然用 clamp 后的版本，避免后续不稳定
        grid_pred = grid_pred_raw.clamp(-0.999, 0.999)

        # -------------------------------------------------
        # backwarp / aligned feature
        # -------------------------------------------------
        ir_back_pred = F.grid_sample(ir_d, grid_pred, align_corners=True, mode="bilinear", padding_mode="border")
        l1_back = (ir_back_pred - ir).abs().mean()

        F_mov_aligned = F.grid_sample(F_mov, grid_pred, align_corners=True, mode="bilinear", padding_mode="border")
        ird_fe_aligned = F.grid_sample(ird_fe, grid_pred, align_corners=True, mode="bilinear", padding_mode="border")
        fusion_image_sample = Fusion(vis_fe, ird_fe_aligned)

        # C2RF-style color restoration
        fusion_rgb_sample = ycbcr_to_rgb_tensor(fusion_image_sample.clamp(0, 1), vis_c1, vis_c2)
        fusion_rgb_1 = ycbcr_to_rgb_tensor(fusion_image_1.clamp(0, 1), vis_c1, vis_c2)
        fusiond_rgb_1 = ycbcr_to_rgb_tensor(fusiond_image_1.clamp(0, 1), visd_c1, visd_c2)
        blackfusion_rgb = None
        if TEST_MODE == "real_infer":
            ird_fe_aligned_black = F.grid_sample(
                ird_fe, grid_pred,
                align_corners=True, mode="bilinear", padding_mode="zeros"
            )
            fusion_image_black = Fusion(vis_fe, ird_fe_aligned_black)
            blackfusion_rgb = ycbcr_to_rgb_tensor(fusion_image_black.clamp(0, 1), vis_c1, vis_c2)
        # -------------------------------------------------
        # losses
        # -------------------------------------------------
        loss_fold = folding_penalty_grid(grid_pred)
        loss_smooth = flow_smooth_loss_from_grid(grid_pred)

        gmean = aux_v["g_pix"].mean()
        loss_gate = F.relu(0.05 - gmean)

        loss_conf = torch.tensor(0.0, device=device)
        conf_pred_mean = 0.0
        conf_pred_last = None
        if ("conf_preds" in out_reg) and (len(out_reg["conf_preds"]) > 0):
            conf_pred_last = out_reg["conf_preds"][-1]
            conf_pred_mean = conf_pred_last.mean().item()
            loss_conf = 0.05 * F.relu(0.25 - conf_pred_last.mean())

        loss_cm = 0.5 * cosine_loss(F_fix, F_mov_aligned.detach()) + \
                  0.5 * cosine_loss(F_fix.detach(), F_mov_aligned)

        loss_fusion = (
                Lgrad(vis, ir, fusion_image_sample) + Loss.Loss_intensity(vis, ir, fusion_image_sample) +
                Lgrad(vis, ir, fusion_image_1) + Loss.Loss_intensity(vis, ir, fusion_image_1) +
                Lgrad(vis_d, ir_d, fusiond_image_1) + Loss.Loss_intensity(vis_d, ir_d, fusiond_image_1)
        )

        loss_same = (
                (1.0 - CC(VISDP_vis_f, IRDP_ir_f) + F.mse_loss(VISDP_vis_f, IRDP_ir_f)) / 2 +
                (1.0 - CC(VISDP_visd_f, IRDP_ird_f) + F.mse_loss(VISDP_visd_f, IRDP_ird_f)) / 2
        )

        # -------------------------------------------------
        # GT-supervised metrics if synthetic_eval
        # -------------------------------------------------
        if grid_inv_gt is not None:
            gt_match = grid_inv_gt
            if (grid_pred.shape[1] != grid_inv_gt.shape[1]) or (grid_pred.shape[2] != grid_inv_gt.shape[2]):
                g = grid_inv_gt.permute(0, 3, 1, 2)
                g = F.interpolate(g, size=grid_pred.shape[1:3], mode="bilinear", align_corners=True)
                gt_match = g.permute(0, 2, 3, 1).contiguous()

            loss_grid_full = F.l1_loss(grid_pred, gt_match)
            loss_grid_2 = F.l1_loss(grid_downsample(grid_pred, 2), grid_downsample(gt_match, 2))
            loss_grid_4 = F.l1_loss(grid_downsample(grid_pred, 4), grid_downsample(gt_match, 4))
            loss_grid_ms = 1.0 * loss_grid_full + 1.5 * loss_grid_2 + 2.0 * loss_grid_4

            total_loss = (
                    w_grid_ms(EVAL_EPOCH_FOR_LOSS) * loss_grid_ms +
                    w_grid_full(EVAL_EPOCH_FOR_LOSS) * loss_grid_full +
                    w_photo(EVAL_EPOCH_FOR_LOSS) * l1_back +
                    w_smooth(EVAL_EPOCH_FOR_LOSS) * loss_smooth +
                    w_fold(EVAL_EPOCH_FOR_LOSS) * loss_fold +
                    w_conf(EVAL_EPOCH_FOR_LOSS) * loss_conf +
                    w_cm(EVAL_EPOCH_FOR_LOSS) * loss_cm +
                    w_gate(EVAL_EPOCH_FOR_LOSS) * loss_gate +
                    w_same(EVAL_EPOCH_FOR_LOSS) * loss_same +
                    w_fuse(EVAL_EPOCH_FOR_LOSS) * loss_fusion
            )

            epe = torch.sqrt(((grid_pred - gt_match) ** 2).sum(dim=-1) + 1e-8).mean().item()

            B0, H0, W0, _ = grid_pred.shape
            idg = build_identity_grid(B0, H0, W0, grid_pred.device, grid_pred.dtype)
            flowmag = torch.sqrt(((grid_pred - idg) ** 2).sum(dim=-1) + 1e-8).mean().item()
            disp_pred = flowmag
            disp_gt = torch.sqrt(((gt_match - idg) ** 2).sum(dim=-1) + 1e-8).mean().item()
        else:
            loss_grid_full = torch.tensor(0.0, device=device)
            loss_grid_ms = torch.tensor(0.0, device=device)
            total_loss = (
                    w_photo(EVAL_EPOCH_FOR_LOSS) * l1_back +
                    w_smooth(EVAL_EPOCH_FOR_LOSS) * loss_smooth +
                    w_fold(EVAL_EPOCH_FOR_LOSS) * loss_fold +
                    w_conf(EVAL_EPOCH_FOR_LOSS) * loss_conf +
                    w_cm(EVAL_EPOCH_FOR_LOSS) * loss_cm +
                    w_gate(EVAL_EPOCH_FOR_LOSS) * loss_gate +
                    w_same(EVAL_EPOCH_FOR_LOSS) * loss_same +
                    w_fuse(EVAL_EPOCH_FOR_LOSS) * loss_fusion
            )

            B0, H0, W0, _ = grid_pred.shape
            idg = build_identity_grid(B0, H0, W0, grid_pred.device, grid_pred.dtype)
            flowmag = torch.sqrt(((grid_pred - idg) ** 2).sum(dim=-1) + 1e-8).mean().item()
            disp_pred = flowmag

            epe = float("nan")
            disp_gt = float("nan")

        # -------------------------------------------------
        # metrics
        # -------------------------------------------------
        dx1, dy1 = grad_xy(ir_back_pred)
        dx2, dy2 = grad_xy(ir)
        l1_grad = ((dx1 - dx2).abs().mean() + (dy1 - dy2).abs().mean()).item()

        pv = aux_v["w_patch"].mean(dim=(0, 2, 3))
        pi = aux_id["w_patch"].mean(dim=(0, 2, 3))
        ent_v = -(pv * pv.clamp_min(1e-8).log()).sum().item()
        ent_i = -(pi * pi.clamp_min(1e-8).log()).sum().item()

        gate_map = aux_v["g_pix"]
        gate_entropy = -(gate_map.clamp_min(1e-8) * gate_map.clamp_min(1e-8).log()).mean().item()
        for b in range(B):
            sample_name = Path(names[b]).stem
            sample_dir = os.path.join(save_vis_dir, sample_name)
            utils.check_dir(sample_dir)

            vis_b = vis[b:b + 1]
            vis_rgb_d_b = vis_rgb_d[b:b + 1]
            vis_rgb_b = vis_rgb[b:b + 1]
            ir_b = ir[b:b + 1]
            ird_b = ir_d[b:b + 1]
            vis_fe_b = vis_fe[b:b + 1]
            ird_fe_b = ird_fe[b:b + 1]
            vis_c1_b = vis_c1[b:b + 1]
            vis_c2_b = vis_c2[b:b + 1]
            ird_black_b = ir_d_heibianban[b:b + 1] if ir_d_heibianban is not None else ird_b

            # 单独保存 ir_d 和 ir 的误差图
            err_ir_d = (ird_b - ir_b).abs()
            save_gray_tensor(err_ir_d[0], os.path.join(sample_dir, "ir_d_vs_ir_error.png"))

            # 后面继续你的 reg_steps 保存逻辑
            # -------------------------------------------------
            # step-wise visualization for CrossGate / iterative reg
            # -------------------------------------------------
            # -------------------------------------------------
            # step-wise visualization for CrossGate / iterative reg
            # 完全对齐 ir_warp_back 的黑边逻辑
            # -------------------------------------------------
            if "grid_steps" in out_reg:
                step_dir = os.path.join(sample_dir, "reg_steps")
                utils.check_dir(step_dir)

                grid_steps = out_reg["grid_steps"]
                flow_steps = out_reg["flow_steps"]
                conf_steps = out_reg["conf_preds"]

                ird_black_b = ir_d_heibianban[b:b + 1] if ir_d_heibianban is not None else ird_b
                valid_src_b_step = valid_src[b:b + 1] if valid_src is not None else torch.ones_like(ird_black_b)

                save_gray_tensor(ir_b[0], os.path.join(step_dir, "step_ref_ir.png"))
                save_gray_tensor(ird_black_b[0], os.path.join(step_dir, "step_input_ir_deformed.png"))
                save_gray_tensor(vis_b[0], os.path.join(step_dir, "step_ref_vis.png"))

                for s in range(len(grid_steps)):
                    step_raw_b = torch.nan_to_num(
                        grid_steps[s][b:b + 1],
                        nan=0.0, posinf=2.0, neginf=-2.0
                    )
                    flow_sb = flow_steps[s][b:b + 1]
                    conf_sb = conf_steps[s][b:b + 1]

                    # 给 step-wise fusion 单独用一个更稳的 grid
                    step_grid_b = step_raw_b.clamp(-0.999, 0.999)

                    # =====================================================
                    # A. warp 可视化：保留你现在的黑边逻辑
                    # =====================================================
                    ir_step_img = F.grid_sample(
                        ird_black_b,
                        step_raw_b,
                        align_corners=True,
                        mode="bilinear",
                        padding_mode="zeros"
                    )

                    step_valid_b = F.grid_sample(
                        valid_src_b_step,
                        step_raw_b,
                        align_corners=True,
                        mode="nearest",
                        padding_mode="zeros"
                    )

                    step_oob_b = grid_inbounds_mask_hard_margin(step_raw_b, border_px=1)
                    step_hard_mask_b = (step_valid_b > 0.5).float() * step_oob_b

                    step_mask_soft_b = F.interpolate(
                        step_hard_mask_b.float(),
                        scale_factor=0.5,
                        mode="bilinear",
                        align_corners=False
                    )
                    step_mask_soft_b = F.interpolate(
                        step_mask_soft_b,
                        size=step_hard_mask_b.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    ).clamp(0.0, 1.0)

                    step_mask_soft_b = kornia.filters.gaussian_blur2d(
                        step_mask_soft_b, (5, 5), (0.8, 0.8)
                    ).clamp(0.0, 1.0)

                    step_core_b = 1.0 - F.max_pool2d(
                        1.0 - step_hard_mask_b.float(),
                        kernel_size=5,
                        stride=1,
                        padding=2
                    )

                    step_mask_smooth_b = torch.where(
                        step_core_b > 0.5,
                        torch.ones_like(step_mask_soft_b),
                        step_mask_soft_b
                    ).clamp(0.0, 1.0)

                    # 最终黑边版 warp 图
                    ir_step = ir_step_img * step_mask_smooth_b

                    # =====================================================
                    # B. error 可视化：恢复原来 testcolor 的逻辑
                    # =====================================================
                    ir_step_err = F.grid_sample(
                        ird_b,
                        step_raw_b,
                        align_corners=True,
                        mode="bilinear",
                        padding_mode="border"
                    )
                    err_step = (ir_step_err - ir_b).abs()

                    chk_step = make_checkerboard_pair(vis_b, ir_step, tile=6)

                    ird_fe_step = F.grid_sample(
                        ird_fe_b,
                        step_grid_b,
                        align_corners=True,
                        mode="bilinear",
                        padding_mode="border"
                    )

                    fusion_step_y = Fusion(vis_fe_b, ird_fe_step)
                    fusion_step_rgb = ycbcr_to_rgb_tensor(
                        fusion_step_y.clamp(0, 1),
                        vis_c1_b,
                        vis_c2_b
                    )

                    # 与 step_warp 完全一致的黑边
                    fusion_step_y_black = fusion_step_y * step_mask_smooth_b
                    fusion_step_rgb_black = fusion_step_rgb * step_mask_smooth_b

                    # 保存 step-wise fusion
                    save_gray_tensor(
                        fusion_step_y_black[0],
                        os.path.join(step_dir, f"step_{s + 1}_fusion_y.png")
                    )
                    save_color_tensor(
                        fusion_step_rgb_black[0],
                        os.path.join(step_dir, f"step_{s + 1}_fusion_rgb.png")
                    )

                    # 保存其他 step 可视化
                    save_gray_tensor(ir_step[0], os.path.join(step_dir, f"step_{s + 1}_warp.png"))
                    save_gray_tensor(err_step[0], os.path.join(step_dir, f"step_{s + 1}_error.png"))
                    save_gray_tensor(chk_step[0], os.path.join(step_dir, f"step_{s + 1}_checker.png"))

                    save_heatmap_tensor(conf_sb[0], os.path.join(step_dir, f"step_{s + 1}_conf.png"))
                    save_overlay_map(
                        ir_step[0],
                        conf_sb[0],
                        os.path.join(step_dir, f"step_{s + 1}_conf_overlay.png")
                    )
                    flow_rgb_s = flow_to_rgb(flow_sb[0])
                    Image.fromarray(flow_rgb_s).save(os.path.join(step_dir, f"step_{s + 1}_flow_rgb.png"))

            # =========================================================
            # step-wise confidence maps
            # =========================================================
            if ("conf_preds" in out_reg) and (len(out_reg["conf_preds"]) > 0):
                conf_dir = os.path.join(sample_dir, "reg_steps_conf")
                utils.check_dir(conf_dir)

                conf_mean_list = []
                for s, conf_t in enumerate(out_reg["conf_preds"], start=1):
                    conf_b_s = conf_t[b:b + 1]  # [1,1,H,W]
                    conf_mean_list.append(conf_b_s.mean().item())
                # 把每一步的均值也记下来，后面挑图方便
                with open(os.path.join(conf_dir, "conf_step_means.txt"), "w") as f:
                    for s, m in enumerate(conf_mean_list, start=1):
                        f.write(f"step_{s:02d}: {m:.6f}\n")

            VISDP_vis_f_b = fusion_rgb_1[b:b + 1].abs().mean(dim=1, keepdim=True)
            # 保存 VISDP_vis_f 的通道均值图
            save_feature_gray(
                VISDP_vis_f_b[0],
                os.path.join(sample_dir, "VISDP_vis_f_mean.png")
            )

            save_heatmap_tensor(
                VISDP_vis_f_b[0],
                os.path.join(sample_dir, "VISDP_vis_f_heatmair_d_heibianbanp.png")
            )

            # 单样本变量]
            grid_b = grid_pred[b:b + 1]
            fusion_rgb_b = fusion_rgb_sample[b:b + 1]
            fusion_rgb_noreg_b = fusion_rgb_noreg[b:b + 1]
            fusion_image_ir_d_noreg_b = fusion_image_ir_d_noreg[b:b + 1]
            fusion_rgb_ir_d_noreg_b = fusion_rgb_ir_d_noreg[b:b + 1]
            if TEST_MODE == "real_infer":
                mask = np.load("mask_pred_smooth.npy")  # [H, W]
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

                # 如果尺寸不一致，插值到 fusion 尺寸
                if mask.shape[-2:] != fusion_rgb_noreg.shape[-2:]:
                    mask = F.interpolate(mask, size=fusion_rgb_noreg.shape[-2:], mode="bilinear", align_corners=False)

                fusion_rgb_noregblack = fusion_rgb_noreg_b * mask
            ir_d_heibianban_b = ir_d_heibianban[b:b + 1] if ir_d_heibianban is not None else None
            blackfusion_b = blackfusion_rgb[b:b + 1] if blackfusion_rgb is not None else None
            valid_src_b = valid_src[b:b + 1] if valid_src is not None else None

            ir_back_src_b = ir_d_heibianban_b if ir_d_heibianban_b is not None else ird_b

            # =========================================================
            # 1) prediction warp_back
            #    图像内容：raw 预测网格 clamp 后稳定采样
            # =========================================================
            pred_raw_b = torch.nan_to_num(
                grid_pred_raw[b:b + 1],
                nan=0.0, posinf=2.0, neginf=-2.0
            )

            # 1) 图像内容：prediction 自己的 raw warp
            ir_back_pred_img_b = F.grid_sample(
                ir_back_src_b,
                pred_raw_b,
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros"
            )

            # 2) 前向有效区：必须来自“硬 valid_src”
            if valid_src_b is not None:
                pred_valid_b = F.grid_sample(
                    valid_src_b,
                    pred_raw_b,
                    align_corners=True,
                    mode="nearest",
                    padding_mode="zeros"
                )
            else:
                pred_valid_b = torch.ones_like(ir_back_pred_img_b)

            # 3) prediction 自己是否越界：硬判定
            pred_oob_b = grid_inbounds_mask_hard_margin(pred_raw_b, border_px=1)

            # 4) 最终硬 mask：既要在图内，也要落在 forward 有效区
            pred_hard_mask_b = (pred_valid_b > 0.5).float() * pred_oob_b

            # 可选：把无效区稍微扩一圈，防止边缘漏亮
            # pred_invalid_b = 1.0 - pred_hard_mask_b
            # pred_invalid_b = F.max_pool2d(pred_invalid_b, kernel_size=3, stride=1, padding=1)
            # pred_hard_mask_b = 1.0 - pred_invalid_b

            # 5) 最终 prediction warp_back
            # 只对已经得到的黑边 mask 做轻微平滑，不改前面逻辑
            # 1) 先把 hard mask 下采样再上采样，专门消除台阶锯齿
            pred_mask_soft_b = F.interpolate(
                pred_hard_mask_b.float(),
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False
            )

            pred_mask_soft_b = F.interpolate(
                pred_mask_soft_b,
                size=pred_hard_mask_b.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).clamp(0.0, 1.0)

            # 2) 再做一次很轻的高斯，让边更顺
            pred_mask_soft_b = kornia.filters.gaussian_blur2d(
                pred_mask_soft_b, (5, 5), (0.8, 0.8)
            ).clamp(0.0, 1.0)

            # 3) 只保留“腐蚀后的内部主体”，不要保留原始锯齿边
            pred_core_b = 1.0 - F.max_pool2d(
                1.0 - pred_hard_mask_b.float(),
                kernel_size=5,
                stride=1,
                padding=2
            )

            # 4) 内部保持1，只有边缘区域用 soft mask
            pred_mask_smooth_b = torch.where(
                pred_core_b > 0.5,
                torch.ones_like(pred_mask_soft_b),
                pred_mask_soft_b
            ).clamp(0.0, 1.0)

            ir_back_pred_b = ir_back_pred_img_b * pred_mask_smooth_b
            # 直接用当前样本的 prediction 黑边 mask 生成 noreg 黑边融合图
            # [1,3,H,W] * [1,1,H,W] 会自动广播
            if TEST_MODE == "synthetic_eval":
                fusion_rgb_noregblack = fusion_rgb_sample * pred_mask_smooth_b
                np.save(
                    os.path.join("mask_pred_smooth.npy"),
                    pred_mask_smooth_b[0, 0].detach().cpu().numpy().astype(np.float32)
                )

                save_gray_tensor(
                    pred_mask_smooth_b[0],
                    os.path.join("mask_pred_smooth.png")
                )

            # 只用 prediction 自己 warp 回来的有效区域，不再叠加 raw oob mask
            pred_mask_b = smooth_mask(pred_valid_b, ksize=5)

            ir_back_gt_b = None
            gt_mask_b = None
            gt_valid_b = None
            gt_oob_b = None

            if grid_inv_gt is not None:
                gt_sample_b = grid_inv_gt[b:b + 1]

                ir_back_gt_img_b = F.grid_sample(
                    ir_back_src_b,
                    gt_sample_b,
                    align_corners=True,
                    mode="bilinear",
                    padding_mode="zeros"
                )

                if valid_src_b is not None:
                    gt_valid_b = F.grid_sample(
                        valid_src_b,
                        gt_sample_b,
                        align_corners=True,
                        mode="bilinear",
                        padding_mode="zeros"
                    ).clamp(0.0, 1.0).pow(1.12)
                else:
                    gt_valid_b = torch.ones_like(ir_back_gt_img_b)

                if grid_inv_gt_raw is not None:
                    gt_raw_b = torch.nan_to_num(
                        grid_inv_gt_raw[b:b + 1],
                        nan=0.0, posinf=2.0, neginf=-2.0
                    )
                    gt_oob_b = grid_inbounds_mask_soft(gt_raw_b, pad_px=2.5)
                    gt_mask_b = smooth_mask(gt_valid_b * gt_oob_b, ksize=5)
                else:
                    gt_mask_b = smooth_mask(gt_valid_b, ksize=5)

                ir_back_gt_b = ir_back_gt_img_b * gt_mask_b
            else:
                gt_mask_b = None
                ir_back_gt_b = None

            # 原始 IR 加 GT 黑边
            ir_with_gt_border_b = ir_b * gt_mask_b if gt_mask_b is not None else ir_b.clone()

            # fusion_rgb_noreg 加 GT 黑边
            fusion_rgb_noreg_with_gt_border_b = (
                fusion_rgb_noreg_b * gt_mask_b if gt_mask_b is not None
                else fusion_rgb_noreg_b.clone()
            )
            # 原始 IR 上叠加与 ir_warp_backyuanshi.png 相同的黑边

            err_pred_b = (ir_back_pred_b - ir_b).abs()

            # =========================================================
            # 4) 把中间 mask 也保存出来，方便你检查
            # =========================================================
            save_gray_tensor(pred_valid_b[0], os.path.join(sample_dir, "mask_pred_valid.png"))
            save_gray_tensor(pred_hard_mask_b[0], os.path.join(sample_dir, "mask_pred_final.png"))

            if gt_valid_b is not None:
                save_gray_tensor(gt_valid_b[0], os.path.join(sample_dir, "mask_gt_valid.png"))
            if gt_oob_b is not None:
                save_gray_tensor(gt_oob_b[0], os.path.join(sample_dir, "mask_gt_oob.png"))
            if gt_mask_b is not None:
                save_gray_tensor(gt_mask_b[0], os.path.join(sample_dir, "mask_gt_final.png"))

            gate_b_vis = aux_v["g_pix"][b:b + 1]
            gate_b_ir = aux_id["g_pix"][b:b + 1]

            flow_b = grid_b - build_identity_grid(
                1, grid_pred.shape[1], grid_pred.shape[2], grid_pred.device, grid_pred.dtype
            )
            # ===== predicted inverse flow =====
            idg_1 = build_identity_grid(
                1, grid_pred.shape[1], grid_pred.shape[2], grid_pred.device, grid_pred.dtype
            )

            flow_pred_b = grid_b - idg_1  # [1,H,W,2]

            # 保存预测 flow 原始数值
            np.save(
                os.path.join(sample_dir, "flow_pred.npy"),
                flow_pred_b[0].detach().cpu().numpy().astype(np.float32)
            )

            # ===== GT flow(s) for synthetic_eval =====
            if grid_inv_gt is not None:
                gt_grid_b = grid_inv_gt[b:b + 1]
                flow_gt_inv_b = gt_grid_b - idg_1  # GT inverse flow，和预测直接对比这个
                flow_gt_fwd_b = grid_pull[b:b + 1] - idg_1  # 合成时真正施加的 forward deformation，可选保存

                np.save(
                    os.path.join(sample_dir, "flow_gt_inv.npy"),
                    flow_gt_inv_b[0].detach().cpu().numpy().astype(np.float32)
                )
                np.save(
                    os.path.join(sample_dir, "flow_gt_forward.npy"),
                    flow_gt_fwd_b[0].detach().cpu().numpy().astype(np.float32)
                )

                # 用 shared max 做可视化，pred/gt 才能公平对比
                mag_pred = torch.sqrt((flow_pred_b[0, ..., 0] ** 2 + flow_pred_b[0, ..., 1] ** 2)).max().item()
                mag_gt_inv = torch.sqrt((flow_gt_inv_b[0, ..., 0] ** 2 + flow_gt_inv_b[0, ..., 1] ** 2)).max().item()
                mag_gt_fwd = torch.sqrt((flow_gt_fwd_b[0, ..., 0] ** 2 + flow_gt_fwd_b[0, ..., 1] ** 2)).max().item()

                shared_max_inv = max(mag_pred, mag_gt_inv, 1e-8)
                shared_max_all = max(mag_pred, mag_gt_inv, mag_gt_fwd, 1e-8)

                pred_rgb_shared = flow_to_rgb_with_max(flow_pred_b[0], mag_max=shared_max_inv)
                gt_inv_rgb_shared = flow_to_rgb_with_max(flow_gt_inv_b[0], mag_max=shared_max_inv)
                gt_fwd_rgb_shared = flow_to_rgb_with_max(flow_gt_fwd_b[0], mag_max=shared_max_all)

                Image.fromarray(pred_rgb_shared).save(os.path.join(sample_dir, "flow_pred_rgb_shared.png"))
                Image.fromarray(gt_inv_rgb_shared).save(os.path.join(sample_dir, "flow_gt_inv_rgb_shared.png"))
                Image.fromarray(gt_fwd_rgb_shared).save(os.path.join(sample_dir, "flow_gt_forward_rgb_shared.png"))

                # 也可以顺手存一个差值幅度图
                flow_diff_b = flow_pred_b - flow_gt_inv_b
                flow_diff_mag = torch.sqrt(flow_diff_b[..., 0] ** 2 + flow_diff_b[..., 1] ** 2).unsqueeze(1)
                save_heatmap_tensor(flow_diff_mag[0], os.path.join(sample_dir, "flow_diff_mag.png"))

            panel_color = torch.cat([
                vis_rgb_b,
                gray_to_3ch(ir_b),
                gray_to_3ch(ird_b),
                gray_to_3ch(ir_back_pred_b),
                fusion_rgb_noreg_b.clamp(0, 1),
                fusion_rgb_b.clamp(0, 1),
                gray_to_3ch(err_pred_b.clamp(0, 1))
            ], dim=3)
            save_color_tensor(panel_color, os.path.join(sample_dir, "panel_color.jpg"))

            save_gray_tensor(vis_b[0], os.path.join(sample_dir, "vis_y.png"))
            save_color_tensor(vis_rgb_d_b, os.path.join(sample_dir, "vis_rgb_d_b.png"))
            save_color_tensor(vis_rgb_b, os.path.join(sample_dir, "vis_rgb.png"))
            save_gray_tensor(ir_b[0], os.path.join(sample_dir, "ir.png"))
            save_gray_tensor(ird_b[0], os.path.join(sample_dir, "ir_deformed_or_input.png"))

            if ir_d_heibianban_b is not None:
                save_gray_tensor(
                    ir_d_heibianban_b[0],
                    os.path.join(sample_dir, "ir_deformed_or_inputheibianban.png")
                )

            save_gray_tensor(ir_back_pred_b[0], os.path.join(sample_dir, "ir_warp_back.png"))


            save_gray_tensor(err_pred_b[0], os.path.join(sample_dir, "warp_error.png"))
            save_gray_tensor(fusion_image_sample[b], os.path.join(sample_dir, "fusion_gray.png"))
            save_color_tensor(fusion_rgb_b, os.path.join(sample_dir, "fusion_rgb.png"))

            save_color_tensor(
                fusion_rgb_ir_d_noreg_b,
                os.path.join(sample_dir, "fusion_ir_d_noreg_rgb.png")
            )
            fusion_rgb_ir_d_noreg_black_b = (
                fusion_rgb_ir_d_noreg_black[b:b + 1]
                if fusion_rgb_ir_d_noreg_black is not None else None
            )



            save_color_tensor(
                fusion_rgb_noregblack.clamp(0, 1),
                os.path.join(sample_dir, "fusion_rgb_noregblack.png")
            )
            save_color_tensor(
                fusion_rgb_noreg_with_gt_border_b,
                os.path.join(sample_dir, "fusion_rgb_noreg_with_gt_blackborder.png")
            )
            save_color_tensor(fusion_rgb_1[b:b + 1], os.path.join(sample_dir, "fusion_decoder_direct_rgb.png"))
            save_color_tensor(fusiond_rgb_1[b:b + 1], os.path.join(sample_dir, "fusion_decoder_deformed_rgb.png"))

            save_heatmap_tensor_pair(
                gate_b_vis[0], gate_b_ir[0],
                os.path.join(sample_dir, "gate_map_vis.png"),
                os.path.join(sample_dir, "gate_map_ir.png")
            )
            save_overlay_map(ir_b[0], gate_b_vis[0], os.path.join(sample_dir, "gate_overlay_vis.png"))
            save_overlay_map(ir_b[0], gate_b_ir[0], os.path.join(sample_dir, "gate_overlay_ir.png"))

            if conf_pred_last is not None:
                conf_b = conf_pred_last[b:b + 1]
                save_heatmap_tensor(conf_b[0], os.path.join(sample_dir, "conf_map.png"))
                save_overlay_map(ir_b[0], conf_b[0], os.path.join(sample_dir, "conf_overlay.png"))

            flow_rgb = flow_to_rgb(flow_b[0])
            Image.fromarray(flow_rgb).save(os.path.join(sample_dir, "flow_rgb.png"))

            save_router_hist(aux_v["w_patch"][b:b + 1], os.path.join(sample_dir, "router_hist_vis.txt"))
            save_router_hist(aux_id["w_patch"][b:b + 1], os.path.join(sample_dir, "router_hist_ir.txt"))

            with open(os.path.join(sample_dir, "sample_stats.txt"), "w") as f:
                f.write(f"name={sample_name}\n")
                f.write(f"mode={TEST_MODE}\n")
                f.write(f"loss={float(total_loss.item()):.6f}\n")
                f.write(f"grid_ms={float(loss_grid_ms.item()):.6f}\n")
                f.write(f"grid_full={float(loss_grid_full.item()):.6f}\n")
                f.write(f"photo={float(l1_back.item()):.6f}\n")
                f.write(f"smooth={float(loss_smooth.item()):.6f}\n")
                f.write(f"fold={float(loss_fold.item()):.6f}\n")
                f.write(f"epe={epe}\n")
                f.write(f"flowmag={flowmag:.6f}\n")
                f.write(f"disp_pred={disp_pred:.6f}\n")
                f.write(f"disp_gt={disp_gt}\n")
                f.write(f"l1_grad={l1_grad:.6f}\n")
                f.write(f"conf_pred_mean={conf_pred_mean:.6f}\n")
                f.write(f"gate_mean={gmean.item():.6f}\n")
                f.write(f"gate_entropy={gate_entropy:.6f}\n")
                f.write(f"router_ent_v={ent_v:.6f}\n")
                f.write(f"router_ent_i={ent_i:.6f}\n")
                f.write(f"loss_cm={float(loss_cm.item()):.6f}\n")
                f.write(f"loss_same={float(loss_same.item()):.6f}\n")
                f.write(f"loss_fusion={float(loss_fusion.item()):.6f}\n")

            append_csv([
                sample_name, TEST_MODE,
                float(total_loss.item()), float(loss_grid_ms.item()), float(loss_grid_full.item()),
                float(l1_back.item()), float(loss_smooth.item()), float(loss_fold.item()),
                epe, float(flowmag), float(disp_pred), disp_gt,
                float(l1_grad), float(conf_pred_mean),
                float(gmean.item()), float(gate_entropy),
                float(ent_v), float(ent_i),
                float(loss_cm.item()), float(loss_same.item()), float(loss_fusion.item())
            ])
        # -------------------------------------------------
        # accumulate mean
        # -------------------------------------------------
        all_loss.append(float(total_loss.item()))
        all_grid_ms.append(float(loss_grid_ms.item()))
        all_grid_full.append(float(loss_grid_full.item()))
        all_photo.append(float(l1_back.item()))
        all_smooth.append(float(loss_smooth.item()))
        all_fold.append(float(loss_fold.item()))
        all_l1_grad.append(float(l1_grad))
        all_conf.append(float(conf_pred_mean))
        all_gate.append(float(gmean.item()))
        all_gate_entropy.append(float(gate_entropy))
        all_ent_v.append(float(ent_v))
        all_ent_i.append(float(ent_i))
        all_loss_cm.append(float(loss_cm.item()))
        all_loss_same.append(float(loss_same.item()))
        all_loss_fusion.append(float(loss_fusion.item()))
        all_flowmag.append(float(flowmag))
        all_disp_pred.append(float(disp_pred))

        if not np.isnan(epe):
            all_epe.append(float(epe))
        if not np.isnan(disp_gt):
            all_disp_gt.append(float(disp_gt))

        print(f"epe={epe if not np.isnan(epe) else 'nan'} flowmag={flowmag:.4f} "
              f"l1_back={l1_back.item():.4f} l1_grad={l1_grad:.4f} fold={loss_fold.item():.4f}")

        print(f"[TEST S{step}] loss={total_loss.item():.4f} grid={loss_grid_ms.item():.4f} "
              f"photo={l1_back.item():.4f} smooth={loss_smooth.item():.4f} "
              f"fold={loss_fold.item():.4f} disp_pred={disp_pred:.4f} "
              f"disp_gt={disp_gt if not np.isnan(disp_gt) else 'nan'} mode={TEST_MODE}")

        print(f"conf_pred_mean={conf_pred_mean:.3f} gate_mean={gmean.item():.3f} "
              f"gate_entropy={gate_entropy:.3f}")

    # -------------------------------------------------
    # final summary
    # -------------------------------------------------
    summary_txt = os.path.join(save_log_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"mode={TEST_MODE}\n")
        f.write(f"ckpt={CKPT_PATH}\n")
        f.write(f"num_samples={len(dataset)}\n")
        f.write(f"loss={np.mean(all_loss):.6f}\n")
        f.write(f"grid_ms={np.mean(all_grid_ms):.6f}\n")
        f.write(f"grid_full={np.mean(all_grid_full):.6f}\n")
        f.write(f"photo={np.mean(all_photo):.6f}\n")
        f.write(f"smooth={np.mean(all_smooth):.6f}\n")
        f.write(f"fold={np.mean(all_fold):.6f}\n")
        f.write(f"epe={(np.mean(all_epe) if len(all_epe) > 0 else float('nan'))}\n")
        f.write(f"flowmag={np.mean(all_flowmag):.6f}\n")
        f.write(f"disp_pred={np.mean(all_disp_pred):.6f}\n")
        f.write(f"disp_gt={(np.mean(all_disp_gt) if len(all_disp_gt) > 0 else float('nan'))}\n")
        f.write(f"l1_grad={np.mean(all_l1_grad):.6f}\n")
        f.write(f"conf_pred_mean={np.mean(all_conf):.6f}\n")
        f.write(f"gate_mean={np.mean(all_gate):.6f}\n")
        f.write(f"gate_entropy={np.mean(all_gate_entropy):.6f}\n")
        f.write(f"router_ent_v={np.mean(all_ent_v):.6f}\n")
        f.write(f"router_ent_i={np.mean(all_ent_i):.6f}\n")
        f.write(f"loss_cm={np.mean(all_loss_cm):.6f}\n")
        f.write(f"loss_same={np.mean(all_loss_same):.6f}\n")
        f.write(f"loss_fusion={np.mean(all_loss_fusion):.6f}\n")

    print("\n================ TEST SUMMARY ================")
    print(f" mode          {TEST_MODE}")
    print(f" ckpt          {CKPT_PATH}")
    print(f" num_samples   {len(dataset)}")
    print(f" loss          {np.mean(all_loss):.6f}")
    print(f" grid_ms       {np.mean(all_grid_ms):.6f}")
    print(f" grid_full     {np.mean(all_grid_full):.6f}")
    print(f" photo         {np.mean(all_photo):.6f}")
    print(f" smooth        {np.mean(all_smooth):.6f}")
    print(f" fold          {np.mean(all_fold):.6f}")
    print(f" epe           {np.mean(all_epe) if len(all_epe) > 0 else float('nan')}")
    print(f" flowmag       {np.mean(all_flowmag):.6f}")
    print(f" disp_pred     {np.mean(all_disp_pred):.6f}")
    print(f" disp_gt       {np.mean(all_disp_gt) if len(all_disp_gt) > 0 else float('nan')}")
    print(f" l1_grad       {np.mean(all_l1_grad):.6f}")
    print(f" conf_mean     {np.mean(all_conf):.6f}")
    print(f" gate_mean     {np.mean(all_gate):.6f}")
    print(f" gate_entropy  {np.mean(all_gate_entropy):.6f}")
    print(f" router_ent_v  {np.mean(all_ent_v):.6f}")
    print(f" router_ent_i  {np.mean(all_ent_i):.6f}")
    print(f" loss_cm       {np.mean(all_loss_cm):.6f}")
    print(f" loss_same     {np.mean(all_loss_same):.6f}")
    print(f" loss_fusion   {np.mean(all_loss_fusion):.6f}")
    print("==============================================\n")


if __name__ == "__main__":
    test()