import os
import time
from pathlib import Path
import math
import csv
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from PIL import Image
from tqdm import tqdm
import kornia
import args
from loss import loss as Loss
from model import model
from utils import utils
from utils.utils import save_img
from utils.utils import RGB2YCrCb, YCbCr2RGB
import matplotlib.cm as cm
import cv2


# =========================================================
# Basic setup
# =========================================================
model_name = "OI-URF"
device_id = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = device_id
device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")

now = int(time.time())
nowTime = time.strftime("%Y%m%d_%H-%M-%S", time.localtime(now))
save_model_dir = args.args.train_save_model_dir + "/" + nowTime + "_" + model_name + "_model"
save_img_dir = args.args.train_save_img_dir + "/" + nowTime + "_" + model_name + "_img"
save_vis_dir = os.path.join(save_img_dir, "paper_vis")
save_log_dir = os.path.join(save_img_dir, "paper_logs")

utils.check_dir(save_model_dir)
utils.check_dir(save_img_dir)
utils.check_dir(save_vis_dir)
utils.check_dir(save_log_dir)

torch.backends.cudnn.benchmark = True
torch.manual_seed(getattr(args.args, "seed", 0))
random.seed(getattr(args.args, "seed", 0))
np.random.seed(getattr(args.args, "seed", 0))

RESUME_PATH = getattr(args.args, "resume_path", "")
START_EPOCH = 0


# =========================================================
# CSV logger
# =========================================================
CSV_LOG = os.path.join(save_log_dir, "train_log.csv")
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "step", "stage",
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


# =========================================================
# Helpers
# =========================================================
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def grad_xy(feat):
    dx = feat[:, :, :, 1:] - feat[:, :, :, :-1]
    dy = feat[:, :, 1:, :] - feat[:, :, :-1, :]
    return dx, dy


def build_identity_grid(B, H, W, device, dtype):
    g = kornia.utils.create_meshgrid(H, W, normalized_coordinates=True, device=device)
    return g.to(dtype=dtype).repeat(B, 1, 1, 1)


def invert_dense_grid(grid_pull, iters=12):
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
        x = (x + (y - fx)).clamp(-1, 1)
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

def save_heatmap_tensor(x, path, cmap="turbo"):
    x = x.detach().float().cpu().squeeze().numpy()
    vmin = np.percentile(x, 1)
    vmax = np.percentile(x, 99)
    x = np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

    colormap = cm.get_cmap(cmap)
    rgb = colormap(x)[..., :3]
    rgb = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def save_heatmap_tensor_pair(x, y, path_x, path_y, cmap="turbo"):
    def to_rgb(arr):
        arr = arr.detach().float().cpu().squeeze().numpy()
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


def save_router_hist(w_patch, path):
    w = w_patch[0].detach().float().cpu().numpy()
    hist = w.mean(axis=(1, 2))
    hist = hist / (hist.sum() + 1e-8)
    np.savetxt(path, hist, fmt="%.6f")


# =========================================================
# Dataset
# =========================================================
class TrainDataset(data.Dataset):
    def __init__(self, vis_dir, ir_dir, rgb_transform, gray_transform):
        self.rgb_transform = rgb_transform
        self.gray_transform = gray_transform

        vis = {p.name: p for p in Path(vis_dir).glob("*")}
        ir = {p.name: p for p in Path(ir_dir).glob("*")}
        names = sorted(vis.keys() & ir.keys())

        self.vis_paths = [vis[n] for n in names]
        self.ir_paths = [ir[n] for n in names]

    def __len__(self):
        return len(self.vis_paths)

    def __getitem__(self, idx):
        vis_rgb = Image.open(str(self.vis_paths[idx])).convert("RGB")
        ir_img = Image.open(str(self.ir_paths[idx])).convert("L")
        return self.rgb_transform(vis_rgb), self.gray_transform(ir_img)


rgb_tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.args.img_size, args.args.img_size]),
    torchvision.transforms.ToTensor()
])

gray_tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.args.img_size, args.args.img_size]),
    torchvision.transforms.ToTensor()
])

dataset = TrainDataset(args.args.vis_train_dir, args.args.ir_train_dir, rgb_tf, gray_tf)
data_iter = data.DataLoader(dataset=dataset, shuffle=True, batch_size=args.args.batch_size, num_workers=0)

iter_num = len(data_iter)
save_image_iter = max(1, int(iter_num / args.args.save_image_num))


Lgrad = Loss.L_Grad().to(device)
CC = Loss.CorrelationCoefficient().to(device)

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

ImageDeformation =model.ImageTransform(
    max_deg=10,
    max_trans_px=10,
    max_scale=0.1,   # 这里保留也没事，但当前随机缩放主要由 scale_min/max 控制
    et_kernel=61,
    et_sigma=20.0,
    et_alpha_px=4.0,

    fixed_scale=None,         # None = 随机缩放
    fixed_elastic=False,
    fixed_elastic_seed=1234,

    scale_min=0.95,
    scale_max=1.05,
).to(device)

Fusion = model.Fusion().to(device)
denoise_reg = model.DenoiseReg(base=32, T=6, max_disp=0.2, step_size=0.03).to(device)

optimizer_reg = torch.optim.Adam(denoise_reg.parameters(), lr=5e-4)

optimizer_FE = torch.optim.Adam(
    [{"params": FE.parameters()},
     {"params": vis_MSFE.parameters()},
     {"params": ir_MSFE.parameters()},
     {"params": FD.parameters()}],
    lr=2e-4
)

optimizer_OBRT = torch.optim.Adam(OBRT.parameters(), lr=8e-4)
optimizer_Fusion = torch.optim.Adam(Fusion.parameters(), lr=2e-4)

if RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f"\nLoading checkpoint from: {RESUME_PATH}")
    ckpt = torch.load(RESUME_PATH, map_location=device)

    FE.load_state_dict(ckpt["FE"])
    ir_MSFE.load_state_dict(ckpt["ir_MSFE"])
    vis_MSFE.load_state_dict(ckpt["vis_MSFE"])
    OBRT.load_state_dict(ckpt["obrt"])
    Fusion.load_state_dict(ckpt["fusion"])
    FD.load_state_dict(ckpt["FD"])

    if "denoise_reg" in ckpt:
        denoise_reg.load_state_dict(ckpt["denoise_reg"])

    try:
        START_EPOCH = int(os.path.basename(RESUME_PATH).split("epoch")[1].split("_")[0])
    except Exception:
        START_EPOCH = 0

    print(f"Resume training from epoch {START_EPOCH}\n")


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

def set_requires_grad(m, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)

def set_stage(epoch: int):
    if epoch < REG_FIRST_EPOCHS:
        set_requires_grad(FE, False)
        set_requires_grad(vis_MSFE, False)
        set_requires_grad(ir_MSFE, False)
        set_requires_grad(FD, False)
        set_requires_grad(Fusion, False)
        set_requires_grad(OBRT, False)
        set_requires_grad(denoise_reg, True)
    else:
        set_requires_grad(FE, True)
        set_requires_grad(vis_MSFE, True)
        set_requires_grad(ir_MSFE, True)
        set_requires_grad(FD, True)
        set_requires_grad(OBRT, True)
        set_requires_grad(Fusion, True)
        set_requires_grad(denoise_reg, True)

def cosine_loss(a, b, eps=1e-6):
    a = F.normalize(a, dim=1, eps=eps)
    b = F.normalize(b, dim=1, eps=eps)
    return 1.0 - (a * b).sum(1).mean()

def train(epoch: int):
    set_stage(epoch)

    epoch_loss_grid = []
    epoch_l1_back = []
    epoch_disp = []
    epoch_gate_mean = []
    epoch_conf_mean = []

    rand_step = random.randint(0, max(0, len(data_iter) - 1))
    saved_epoch_panel = False

    for step, x in enumerate(data_iter):
        vis_rgb = x[0].to(device)   # [B,3,H,W]
        ir = x[1].to(device)        # [B,1,H,W]

        vis, vis_c1, vis_c2 = rgb_to_ycbcr_tensor(vis_rgb)

        # -------------------------------------------------
        # 1) synth deformation
        # -------------------------------------------------
        with torch.no_grad():
            grid_pull = ImageDeformation.generate_grid(vis)
            vis_rgb_d = F.grid_sample(vis_rgb, grid_pull, align_corners=True, mode="bilinear", padding_mode="border")
            ir_d = F.grid_sample(ir, grid_pull, align_corners=True, mode="bilinear", padding_mode="border")
            vis_d, visd_c1, visd_c2 = rgb_to_ycbcr_tensor(vis_rgb_d)
            grid_inv_gt = invert_dense_grid(grid_pull, iters=12).clamp(-1, 1)
        # -------------------------------------------------
        # 2) backbone features
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
        # 3) OBRT
        # -------------------------------------------------

        VISDP_vis_f, aux_v = OBRT(vis_fe, modality="vis", return_aux=True)
        IRDP_ird_f, aux_id = OBRT(ird_fe, modality="ir", return_aux=True)
        VISDP_visd_f, aux_vd = OBRT(visd_fe, modality="vis", return_aux=True)
        IRDP_ir_f, aux_i = OBRT(ir_fe, modality="ir", return_aux=True)

        # CrossGate is now inside denoise_reg
        F_fix = VISDP_vis_f

        F_mov = IRDP_ird_f


        if epoch < REG_FIRST_EPOCHS:
            F_fix_in = F_fix.detach()
            F_mov_in = F_mov.detach()
        else:
            F_fix_in = F_fix
            F_mov_in = F_mov

        # -------------------------------------------------
        # 4) DenoiseReg
        # -------------------------------------------------
        out_reg = denoise_reg(F_fix_in, F_mov_in, return_steps=False)
        grid_pred = out_reg["grid"].clamp(-0.999, 0.999)

        # -------------------------------------------------
        # 5) backwarp
        # -------------------------------------------------
        ir_back_pred = F.grid_sample(ir_d, grid_pred, align_corners=True, mode="bilinear", padding_mode="border")
        l1_back = (ir_back_pred - ir).abs().mean()

        # -------------------------------------------------
        # 6) grid supervision
        # -------------------------------------------------
        gt_match = grid_inv_gt
        if (grid_pred.shape[1] != grid_inv_gt.shape[1]) or (grid_pred.shape[2] != grid_inv_gt.shape[2]):
            g = grid_inv_gt.permute(0, 3, 1, 2)
            g = F.interpolate(g, size=grid_pred.shape[1:3], mode="bilinear", align_corners=True)
            gt_match = g.permute(0, 2, 3, 1).contiguous()

        loss_grid_full = F.l1_loss(grid_pred, gt_match)
        loss_grid_2 = F.l1_loss(grid_downsample(grid_pred, 2), grid_downsample(gt_match, 2))
        loss_grid_4 = F.l1_loss(grid_downsample(grid_pred, 4), grid_downsample(gt_match, 4))
        loss_grid_ms = 1.0 * loss_grid_full + 1.5 * loss_grid_2 + 2.0 * loss_grid_4

        loss_fold = folding_penalty_grid(grid_pred)
        loss_smooth = flow_smooth_loss_from_grid(grid_pred)

        # -------------------------------------------------
        # 7) E2E losses
        # -------------------------------------------------
        # gmean = aux_v["g_pix"].mean()
        # loss_gate = F.relu(0.05 - gmean)

        loss_conf = torch.tensor(0.0, device=device)
        conf_pred_mean = 0.0
        conf_pred_last = None
        if ("conf_preds" in out_reg) and (len(out_reg["conf_preds"]) > 0):
            conf_pred_last = out_reg["conf_preds"][-1]
            conf_pred_mean = conf_pred_last.mean().item()
            loss_conf = 0.05 * F.relu(0.25 - conf_pred_last.mean())

        F_mov_aligned = F.grid_sample(F_mov, grid_pred, align_corners=True, mode="bilinear", padding_mode="border")
        loss_cm = 0.5 * cosine_loss(F_fix, F_mov_aligned.detach()) + 0.5 * cosine_loss(F_fix.detach(), F_mov_aligned)

        ird_fe_aligned = F.grid_sample(ird_fe, grid_pred, align_corners=True, mode="bilinear", padding_mode="border")
        fusion_image_sample = Fusion(vis_fe, ird_fe_aligned)

        # C2RF-style color restoration (not used in loss, only for output)
        fusion_rgb_sample = ycbcr_to_rgb_tensor(fusion_image_sample.clamp(0, 1), vis_c1, vis_c2)
        fusion_rgb_1 = ycbcr_to_rgb_tensor(fusion_image_1.clamp(0, 1), vis_c1, vis_c2)
        fusiond_rgb_1 = ycbcr_to_rgb_tensor(fusiond_image_1.clamp(0, 1), visd_c1, visd_c2)

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
        # 8) total loss
        # -------------------------------------------------
        loss = (
            w_grid_ms(epoch) * loss_grid_ms +
            w_grid_full(epoch) * loss_grid_full +
            w_photo(epoch) * l1_back +
            w_smooth(epoch) * loss_smooth +
            w_fold(epoch) * loss_fold +
            w_conf(epoch) * loss_conf +
            w_cm(epoch) * loss_cm +
            # w_gate(epoch) * loss_gate +
            w_same(epoch) * loss_same +
            w_fuse(epoch) * loss_fusion
        )

        # -------------------------------------------------
        # 9) backward / step
        # -------------------------------------------------
        optimizer_reg.zero_grad()
        optimizer_OBRT.zero_grad()
        optimizer_FE.zero_grad()
        optimizer_Fusion.zero_grad()

        loss.backward()
        optimizer_reg.step()

        if epoch >= REG_FIRST_EPOCHS:
            optimizer_OBRT.step()
            optimizer_FE.step()
            optimizer_Fusion.step()


        # -------------------------------------------------
        # 10) metrics
        # -------------------------------------------------
        B0, H0, W0, _ = grid_pred.shape
        idg = build_identity_grid(B0, H0, W0, grid_pred.device, grid_pred.dtype)
        disp_pred = torch.sqrt(((grid_pred - idg) ** 2).sum(dim=-1) + 1e-8).mean().item()
        disp_gt = torch.sqrt(((gt_match - idg) ** 2).sum(dim=-1) + 1e-8).mean().item()

        epoch_loss_grid.append(loss_grid_ms.detach().item())
        epoch_l1_back.append(l1_back.detach().item())
        epoch_disp.append(disp_pred)
        # epoch_gate_mean.append(gmean.detach().item())
        epoch_conf_mean.append(conf_pred_mean)
        if step % 50 == 0:
            with torch.no_grad():
                idg = build_identity_grid(grid_pred.size(0), grid_pred.size(1), grid_pred.size(2),
                                          grid_pred.device, grid_pred.dtype)
                epe = torch.sqrt(((grid_pred - gt_match) ** 2).sum(dim=-1) + 1e-8).mean().item()
                flowmag = torch.sqrt(((grid_pred - idg) ** 2).sum(dim=-1) + 1e-8).mean().item()

                dx1, dy1 = grad_xy(ir_back_pred)
                dx2, dy2 = grad_xy(ir)
                l1_grad = ((dx1 - dx2).abs().mean() + (dy1 - dy2).abs().mean()).item()

                pv = aux_v["w_patch"].mean(dim=(0, 2, 3))
                pi = aux_id["w_patch"].mean(dim=(0, 2, 3))
                ent_v = -(pv * pv.clamp_min(1e-8).log()).sum().item()
                ent_i = -(pi * pi.clamp_min(1e-8).log()).sum().item()

                # gate_map = aux_v["g_pix"]
                # gate_entropy = -(gate_map.clamp_min(1e-8) * gate_map.clamp_min(1e-8).log()).mean().item()

            print(f"epe={epe:.4f} flowmag={flowmag:.4f} l1_back={l1_back.item():.4f} "
                  f"l1_grad={l1_grad:.4f} fold={loss_fold.item():.4f}")
            print(f"[E{epoch} S{step}] loss={loss.item():.4f} grid={loss_grid_ms.item():.4f} "
                  f"photo={l1_back.item():.4f} smooth={loss_smooth.item():.4f} "
                  f"fold={loss_fold.item():.4f} disp_pred={disp_pred:.4f} disp_gt={disp_gt:.4f} "
                  f"stage={'REG' if epoch < REG_FIRST_EPOCHS else 'E2E'}")
            # print(f"conf_pred_mean={conf_pred_mean:.3f} gate_mean={gmean.item():.3f} "
            #       f"gate_entropy={gate_entropy:.3f}")

            append_csv([
                epoch, step, "REG" if epoch < REG_FIRST_EPOCHS else "E2E",
                float(loss.item()), float(loss_grid_ms.item()), float(loss_grid_full.item()),
                float(l1_back.item()), float(loss_smooth.item()), float(loss_fold.item()),
                float(epe), float(flowmag), float(disp_pred), float(disp_gt),
                float(l1_grad), float(conf_pred_mean),
                # float(gmean.item()), float(gate_entropy),
                float(ent_v), float(ent_i),
                float(loss_cm.item()), float(loss_same.item()), float(loss_fusion.item())
            ])

        # -------------------------------------------------
        # 11) paper visualization
        # -------------------------------------------------
        if (step == rand_step) and (not saved_epoch_panel):
            saved_epoch_panel = True
            b = random.randint(0, vis.size(0) - 1)
            with torch.no_grad():
                vis_b = vis[b:b + 1]
                vis_rgb_b = vis_rgb[b:b + 1]
                ir_b = ir[b:b + 1]
                ird_b = ir_d[b:b + 1]
                fusion_rgb_b = fusion_rgb_sample[b:b + 1]

                ir_back_pred_b = F.grid_sample(ird_b, grid_pred[b:b + 1], align_corners=True,
                                               mode="bilinear", padding_mode="border")
                err_pred = (ir_back_pred_b - ir_b).abs()
                gate_b = aux_v["g_pix"][b:b + 1]
                gate_b1 = aux_id["g_pix"][b:b + 1]
                flow_b = (grid_pred[b:b + 1] - build_identity_grid(
                    1, grid_pred.shape[1], grid_pred.shape[2], grid_pred.device, grid_pred.dtype
                ))

                vis_dir = os.path.join(save_vis_dir, f"epoch_{epoch:03d}")
                utils.check_dir(vis_dir)

                panel_gray = torch.cat([
                    vis_b, ir_b, ird_b, ir_back_pred_b,
                    fusion_image_sample[b:b + 1].clamp(0, 1), err_pred.clamp(0, 1)
                ], dim=3)
                save_img(panel_gray, os.path.join(vis_dir, "panel_gray.jpg"))

                panel_color = torch.cat([
                    vis_rgb_b,
                    gray_to_3ch(ir_b),
                    gray_to_3ch(ird_b),
                    gray_to_3ch(ir_back_pred_b),
                    fusion_rgb_b.clamp(0, 1),
                    gray_to_3ch(err_pred.clamp(0, 1))
                ], dim=3)
                save_color_tensor(panel_color, os.path.join(vis_dir, "panel_color.jpg"))

                save_gray_tensor(vis_b[0], os.path.join(vis_dir, "vis_y.png"))
                save_color_tensor(vis_rgb_b, os.path.join(vis_dir, "vis_rgb.png"))
                save_gray_tensor(ir_b[0], os.path.join(vis_dir, "ir.png"))
                save_gray_tensor(ird_b[0], os.path.join(vis_dir, "ir_deformed.png"))
                save_gray_tensor(ir_back_pred_b[0], os.path.join(vis_dir, "ir_warp_back.png"))
                save_gray_tensor(err_pred[0], os.path.join(vis_dir, "warp_error.png"))
                save_gray_tensor(fusion_image_sample[b], os.path.join(vis_dir, "fusion_gray.png"))
                save_color_tensor(fusion_rgb_b, os.path.join(vis_dir, "fusion_rgb.png"))
                save_color_tensor(fusion_rgb_1[b:b + 1], os.path.join(vis_dir, "FD_direct_rgb.png"))
                save_color_tensor(fusiond_rgb_1[b:b + 1], os.path.join(vis_dir, "FD_deformed_rgb.png"))

                save_heatmap_tensor_pair(
                    gate_b[0], gate_b1[0],
                    os.path.join(vis_dir, "gate_map_vis.png"),
                    os.path.join(vis_dir, "gate_map_ir.png")
                )
                save_overlay_map(ir_b[0], gate_b[0], os.path.join(vis_dir, "gate_overlay_vis.png"))

                if conf_pred_last is not None:
                    conf_b = conf_pred_last[b:b + 1]
                    save_heatmap_tensor(conf_b[0], os.path.join(vis_dir, "conf_map.png"))
                    save_overlay_map(ir_b[0], conf_b[0], os.path.join(vis_dir, "conf_overlay.png"))

                flow_rgb = flow_to_rgb(flow_b[0])
                Image.fromarray(flow_rgb).save(os.path.join(vis_dir, "flow_rgb.png"))

                save_router_hist(aux_v["w_patch"][b:b + 1], os.path.join(vis_dir, "router_hist_vis.txt"))
                save_router_hist(aux_id["w_patch"][b:b + 1], os.path.join(vis_dir, "router_hist_ir.txt"))

                with open(os.path.join(vis_dir, "epoch_stats.txt"), "w") as f:
                    f.write(f"epoch={epoch}\n")
                    f.write(f"stage={'REG' if epoch < REG_FIRST_EPOCHS else 'E2E'}\n")
                    f.write(f"loss={loss.item():.6f}\n")
                    f.write(f"grid_ms={loss_grid_ms.item():.6f}\n")
                    f.write(f"photo={l1_back.item():.6f}\n")
                    f.write(f"smooth={loss_smooth.item():.6f}\n")
                    f.write(f"fold={loss_fold.item():.6f}\n")
                    f.write(f"disp_pred={disp_pred:.6f}\n")
                    f.write(f"disp_gt={disp_gt:.6f}\n")
                    f.write(f"conf_pred_mean={conf_pred_mean:.6f}\n")
                    # f.write(f"gate_mean={gmean.item():.6f}\n")

        # -------------------------------------------------
        # 12) periodic compact save
        # -------------------------------------------------
        if step % save_image_iter == 0 and epoch % 2 == 0:
            out_name = os.path.join(save_img_dir, f"{epoch}epoch{step}step_denoise_color.jpg")
            with torch.no_grad():
                err_all = (ir_back_pred - ir).abs()
                out = torch.cat([
                    vis_rgb,
                    gray_to_3ch(ir),
                    gray_to_3ch(ir_d),
                    gray_to_3ch(ir_back_pred),
                    fusion_rgb_sample.clamp(0, 1),
                    gray_to_3ch(err_all.clamp(0, 1))
                ], dim=3)
            save_color_tensor(out, out_name)

        # -------------------------------------------------
        # 13) checkpoint
        # -------------------------------------------------
        if ((epoch + 1) == args.args.Epoch and (step + 1) % iter_num == 0) or (
                epoch % args.args.save_model_num == 0 and (step + 1) % iter_num == 0):
            ckpts = {
                "FE": FE.state_dict(),
                "ir_MSFE": ir_MSFE.state_dict(),
                "vis_MSFE": vis_MSFE.state_dict(),
                "obrt": OBRT.state_dict(),
                "fusion": Fusion.state_dict(),
                "FD": FD.state_dict(),
                "denoise_reg": denoise_reg.state_dict(),
            }
            save_dir = f"{save_model_dir}/epoch{epoch}_iter{step + 1}.pth"
            torch.save(ckpts, save_dir)

    print()
    print(f" -epoch {epoch}  stage={'REG' if epoch < REG_FIRST_EPOCHS else 'E2E'}")
    print(f" -loss_grid_ms   {np.mean(epoch_loss_grid):.6f}")
    print(f" -l1_back        {np.mean(epoch_l1_back):.6f}")
    print(f" -disp_mean      {np.mean(epoch_disp):.6f}")
    print(f" -gate_mean      {np.mean(epoch_gate_mean):.6f}")
    print(f" -conf_mean      {np.mean(epoch_conf_mean):.6f}")
    print()
# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    FE.train()
    vis_MSFE.train()
    ir_MSFE.train()
    FD.train()
    OBRT.train()
    Fusion.train()
    denoise_reg.train()
    ImageDeformation.train()

    for epoch in tqdm(range(START_EPOCH, args.args.Epoch)):
        train(epoch)