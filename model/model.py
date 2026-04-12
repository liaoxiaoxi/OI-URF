# model.py (PATCHED: AC_FALSE unified + stable DenoiseReg GN)
import math
import torch
import torch.nn as nn

import kornia


class MSFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(8, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(32, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 8, 3, 1), nn.BatchNorm2d(8), nn.ReLU(),
        )

    def forward(self, x):
        return self.E(x)


class FD(nn.Module):
    def __init__(self):
        super().__init__()
        self.D_0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, 3, 1),
            nn.BatchNorm2d(8),
        )
        self.D = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(8, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(32, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(16, 4, 3, 1), nn.BatchNorm2d(4), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(4, 1, 3, 1),
        )

    def forward(self, x):
        out_f = self.D_0(x)
        out = self.D(out_f)
        return out, out_f

class FE(nn.Module):
    def __init__(self):
        super().__init__()
        self.B = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(1, 4, 3, 1), nn.BatchNorm2d(4), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(4, 8, 3, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(8, 8, 3, 1), nn.BatchNorm2d(8), nn.ReLU(),
        )

    def forward(self, x):
        return self.B(x)

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(8, 8, 3, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(8, 4, 3, 1), nn.BatchNorm2d(4), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(4, 1, 3, 1),
        )

    def forward(self, vis, ir):
        return self.D(vis + ir)


class PositionalEncodingSBE(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len,1,E]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, t):  # (S,B,E)
        S = t.size(0)
        t = t + self.pe[:S].to(dtype=t.dtype, device=t.device)
        return self.dropout(t)


def make_base_grid(B, H, W, device, dtype):
    g = kornia.utils.create_meshgrid(H, W, normalized_coordinates=True, device=device)  # [1,H,W,2]
    return g.to(dtype=dtype).repeat(B, 1, 1, 1)  # [B,H,W,2]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import get_gaussian_kernel2d


def make_base_grid(B, H, W, device, dtype):
    if H > 1:
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        ys = torch.zeros(H, device=device, dtype=dtype)

    if W > 1:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    else:
        xs = torch.zeros(W, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([xx, yy], dim=-1)   # [H,W,2]
    base = base.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]
    return base


def synth_grid_affine_plus_elastic(
        img,  # [B,C,H,W]
        max_deg=5.0,
        max_trans_px=6.0,
        max_scale=0.03,   # 保留这个参数，但下面主要用 scale_min/scale_max
        et_kernel=61,
        et_sigma=20.0,
        et_alpha_px=3.0,
        align_corners=True,

        fixed_scale=None,          # 传具体缩放值，如 1.0 / 1.03；None表示随机
        fixed_elastic=False,       # True = elastic 固定
        fixed_elastic_seed=1234,   # 固定 elastic 的随机种子

        scale_min=0.95,            # 随机缩放最小值
        scale_max=1.05,            # 随机缩放最大值
):
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    # ---------- affine ----------
    # rotation: 随机 [-max_deg, max_deg]
    deg = (torch.rand(B, device=device, dtype=dtype) - 0.5) * 2.0 * max_deg
    ang = deg * (math.pi / 180.0)

    # scale: 固定 or 随机 [scale_min, scale_max]
    if fixed_scale is None:
        sc = scale_min + (scale_max - scale_min) * torch.rand(B, device=device, dtype=dtype)
    else:
        sc = torch.full((B,), float(fixed_scale), device=device, dtype=dtype)

    # translation in pixel
    tx_px = (torch.rand(B, device=device, dtype=dtype) - 0.5) * 2.0 * max_trans_px
    ty_px = (torch.rand(B, device=device, dtype=dtype) - 0.5) * 2.0 * max_trans_px

    # pixel -> normalized coordinates
    if align_corners:
        tx = tx_px / ((W - 1) * 0.5) if W > 1 else torch.zeros_like(tx_px)
        ty = ty_px / ((H - 1) * 0.5) if H > 1 else torch.zeros_like(ty_px)
    else:
        tx = tx_px / (W * 0.5)
        ty = ty_px / (H * 0.5)

    cos = torch.cos(ang) * sc
    sin = torch.sin(ang) * sc

    theta = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0] = cos.to(torch.float32)
    theta[:, 0, 1] = (-sin).to(torch.float32)
    theta[:, 1, 0] = sin.to(torch.float32)
    theta[:, 1, 1] = cos.to(torch.float32)
    theta[:, 0, 2] = tx.to(torch.float32)
    theta[:, 1, 2] = ty.to(torch.float32)

    grid_aff = F.affine_grid(theta, size=img.size(), align_corners=align_corners).to(dtype)

    # ---------- elastic ----------
    if et_alpha_px <= 0:
        disp_el = torch.zeros(B, H, W, 2, device=device, dtype=dtype)
    else:
        if fixed_elastic:
            # 生成一个固定噪声场，然后复制到 batch
            if device.type == "cuda":
                g = torch.Generator(device=device)
            else:
                g = torch.Generator()

            g.manual_seed(int(fixed_elastic_seed))
            noise = torch.rand((1, 2, H, W), generator=g, device=device, dtype=dtype) * 2.0 - 1.0
            noise = noise.repeat(B, 1, 1, 1)
        else:
            noise = torch.rand(B, 2, H, W, device=device, dtype=dtype) * 2.0 - 1.0

        k = get_gaussian_kernel2d(
            (et_kernel, et_kernel),
            (et_sigma, et_sigma)
        ).to(device=device, dtype=dtype)
        k = k[None, None, :, :]   # [1,1,k,k]
        pad = et_kernel // 2

        noise_p = F.pad(noise, (pad, pad, pad, pad), mode="reflect")
        k2 = k.repeat(2, 1, 1, 1)   # [2,1,k,k]
        disp2 = F.conv2d(noise_p, k2, groups=2)   # [B,2,H,W]

        # 去均值
        disp2 = disp2 - disp2.mean(dim=(2, 3), keepdim=True)

        # 归一化到 [-1,1] 量级
        mx = disp2.flatten(2).abs().max(dim=2)[0].view(B, 2, 1, 1).clamp_min(1e-6)
        disp2 = disp2 / mx

        # pixel -> normalized
        if align_corners:
            ax = et_alpha_px / ((W - 1) * 0.5) if W > 1 else 0.0
            ay = et_alpha_px / ((H - 1) * 0.5) if H > 1 else 0.0
        else:
            ax = et_alpha_px / (W * 0.5)
            ay = et_alpha_px / (H * 0.5)

        disp2[:, 0] *= ax
        disp2[:, 1] *= ay

        disp_el = disp2.permute(0, 2, 3, 1).contiguous()   # [B,H,W,2]

    # ---------- compose ----------
    base = make_base_grid(B, H, W, device, dtype)
    grid = base + (grid_aff - base) + disp_el
    return grid


class ImageTransform(nn.Module):
    def __init__(
            self,
            max_deg=5.0,
            max_trans_px=6.0,
            max_scale=0.03,
            et_kernel=61,
            et_sigma=20.0,
            et_alpha_px=3.0,

            fixed_scale=None,        # 具体缩放值，如1.0 / 1.03；None表示随机
            fixed_elastic=False,
            fixed_elastic_seed=1234,

            scale_min=0.95,
            scale_max=1.05,
    ):
        super().__init__()
        self.max_deg = max_deg
        self.max_trans_px = max_trans_px
        self.max_scale = max_scale
        self.et_kernel = et_kernel
        self.et_sigma = et_sigma
        self.et_alpha_px = et_alpha_px

        self.fixed_scale = fixed_scale
        self.fixed_elastic = fixed_elastic
        self.fixed_elastic_seed = fixed_elastic_seed

        self.scale_min = scale_min
        self.scale_max = scale_max

    def generate_grid(self, x):
        return synth_grid_affine_plus_elastic(
            x,
            max_deg=self.max_deg,
            max_trans_px=self.max_trans_px,
            max_scale=self.max_scale,
            et_kernel=self.et_kernel,
            et_sigma=self.et_sigma,
            et_alpha_px=self.et_alpha_px,
            align_corners=True,

            fixed_scale=self.fixed_scale,
            fixed_elastic=self.fixed_elastic,
            fixed_elastic_seed=self.fixed_elastic_seed,

            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )

    def forward(self, img1, img2):
        assert img1.size() == img2.size()
        grid = self.generate_grid(img1)

        img1_w = F.grid_sample(
            img1, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        img2_w = F.grid_sample(
            img2, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        return img1_w, img2_w, grid
# def synth_grid_affine_plus_elastic(
#         img,  # [B,C,H,W]
#         max_deg=5.0,
#         max_trans_px=6.0,  # ✅ 用像素控制平移
#         max_scale=0.03,
#         et_kernel=61,
#         et_sigma=20.0,
#         et_alpha_px=3.0,  # ✅ elastic 像素幅度（3px 很稳，5px开始明显，8px容易液化）
#         align_corners=True,  # 你现有代码是 True，就统一 True（别混）
# ):
#     B, C, H, W = img.shape
#     device, dtype = img.device, img.dtype

#     # ---------- affine theta (直接对齐 affine_grid 语义，最稳) ----------
#     deg = (torch.rand(B, device=device) - 0.5) * 2.0 * max_deg
#     ang = deg * (math.pi / 180.0)
#     sc = 1.0 + (torch.rand(B, device=device) - 0.5) * 2.0 * max_scale

#     # 像素平移 -> 归一化平移
#     tx = (torch.rand(B, device=device) - 0.5) * 2.0 * (max_trans_px / ((W - 1) * 0.5))
#     ty = (torch.rand(B, device=device) - 0.5) * 2.0 * (max_trans_px / ((H - 1) * 0.5))

#     cos = torch.cos(ang) * sc
#     sin = torch.sin(ang) * sc
#     theta = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
#     theta[:, 0, 0] = cos
#     theta[:, 0, 1] = -sin
#     theta[:, 1, 0] = sin
#     theta[:, 1, 1] = cos
#     theta[:, 0, 2] = tx
#     theta[:, 1, 2] = ty

#     grid_aff = F.affine_grid(theta, size=img.size(), align_corners=align_corners).to(dtype)

#     # ---------- elastic (低频、小幅、像素可控) ----------
#     noise = torch.rand(B, 2, H, W, device=device, dtype=dtype) * 2.0 - 1.0

#     k = get_gaussian_kernel2d((et_kernel, et_kernel), (et_sigma, et_sigma)).to(device=device, dtype=dtype)
#     k = k[None, None, :, :]
#     pad = et_kernel // 2

#     noise_p = F.pad(noise, (pad, pad, pad, pad), mode="reflect")
#     k2 = k.repeat(2, 1, 1, 1)
#     disp2 = F.conv2d(noise_p, k2, groups=2)  # [B,2,H,W]

#     # 去均值 + 用 max 而不是 std（std 太小会导致幅度爆）
#     disp2 = disp2 - disp2.mean(dim=(2, 3), keepdim=True)
#     mx = disp2.flatten(2).abs().max(dim=2)[0].view(B, 2, 1, 1).clamp_min(1e-6)
#     disp2 = disp2 / mx

#     # 像素幅度 -> 归一化幅度
#     ax = (et_alpha_px / ((W - 1) * 0.5))
#     ay = (et_alpha_px / ((H - 1) * 0.5))
#     disp2[:, 0] *= ax
#     disp2[:, 1] *= ay

#     disp_el = disp2.permute(0, 2, 3, 1).contiguous()  # [B,H,W,2]

#     # ---------- compose ----------
#     base = make_base_grid(B, H, W, device, dtype)
#     grid = base + (grid_aff - base) + disp_el
#     return grid


# class ImageTransform(nn.Module):
#     def __init__(
#             self,
#             max_deg=5.0,
#             max_trans_px=6.0,
#             max_scale=0.03,
#             et_kernel=61,
#             et_sigma=20.0,
#             et_alpha_px=3.0,
#     ):
#         super().__init__()
#         self.max_deg = max_deg
#         self.max_trans_px = max_trans_px
#         self.max_scale = max_scale
#         self.et_kernel = et_kernel
#         self.et_sigma = et_sigma
#         self.et_alpha_px = et_alpha_px

#     def generate_grid(self, x):
#         return synth_grid_affine_plus_elastic(
#             x,
#             max_deg=self.max_deg,
#             max_trans_px=self.max_trans_px,
#             max_scale=self.max_scale,
#             et_kernel=self.et_kernel,
#             et_sigma=self.et_sigma,
#             et_alpha_px=self.et_alpha_px,
#             align_corners=True,
#         )

#     def forward(self, img1, img2):
#         assert img1.size() == img2.size()
#         grid = self.generate_grid(img1)

#         img1_w = F.grid_sample(
#             img1, grid,
#             mode="bilinear",
#             padding_mode="border",
#             align_corners=True
#         )
#         img2_w = F.grid_sample(
#             img2, grid,
#             mode="bilinear",
#             padding_mode="border",
#             align_corners=True
#         )
#         return img1_w, img2_w, grid


# =========================================================
# OBRT (shared experts, your logic kept)
# =========================================================
class _ExpertOp(nn.Module):
    def __init__(self, C: int, k: int = 3, act: str = "silu"):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(C, C, kernel_size=k, padding=pad, groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(C)
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class OperatorBankSharedExperts(nn.Module):
    """
    Dual-modality router + partially shared experts
    Also supports all-private setting:
        - share_ratio = 0.0
        - num_experts = 4
    then each modality uses only 4 private experts.
    """

    def __init__(self, C=8, patch_size=4, num_experts=8,
                 share_ratio=0.5,
                 tau_router=1.0, expert_kernel=3, hidden_ratio=0.5,
                 gate_mode="entropy", gate_temp=1.0,
                 gate_floor: float = 0.0):
        super().__init__()
        self.C = C
        self.P = patch_size
        self.E = num_experts
        self.tau = tau_router
        self.gate_mode = gate_mode
        self.gate_temp = gate_temp
        self.gate_floor = float(gate_floor)

        # -------- allow zero shared experts --------
        E_shared = int(round(num_experts * share_ratio))
        E_shared = max(0, min(E_shared, num_experts))
        self.E_shared = E_shared
        self.E_priv = num_experts - E_shared

        hidden = max(8, int((2 * C) * hidden_ratio))
        self.router_v = nn.Sequential(
            nn.Conv2d(2 * C, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
        )
        self.router_i = nn.Sequential(
            nn.Conv2d(2 * C, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
        )
        # shared experts can be zero-length
        self.ops_shared = nn.ModuleList([
            _ExpertOp(C, k=expert_kernel) for _ in range(self.E_shared)
        ])
        self.ops_v_priv = nn.ModuleList([
            _ExpertOp(C, k=expert_kernel) for _ in range(self.E_priv)
        ])
        self.ops_i_priv = nn.ModuleList([
            _ExpertOp(C, k=expert_kernel) for _ in range(self.E_priv)
        ])
    @staticmethod
    def _patch_stats(x, P):
        mu = F.avg_pool2d(x, kernel_size=P, stride=P)
        msq = F.avg_pool2d(x * x, kernel_size=P, stride=P)
        var = (msq - mu * mu).clamp_min(1e-6)
        std = torch.sqrt(var)
        return mu, std
    @staticmethod
    def _upsample_patch_map(m, P, H, W):
        out = m.repeat_interleave(P, dim=2).repeat_interleave(P, dim=3)
        return out[:, :, :H, :W]

    def _gate_from_w(self, w_patch, eps=1e-8):
        if self.gate_mode == "max":
            g = w_patch.max(dim=1, keepdim=True)[0]
        else:
            ent = -(w_patch * (w_patch.clamp_min(eps).log())).sum(dim=1, keepdim=True)
            g = 1.0 - ent / math.log(self.E + eps)
        g = (g / self.gate_temp).clamp(0.0, 1.0)
        if self.gate_floor > 0:
            g = g.clamp_min(self.gate_floor)
        return g
    def _select_router(self, modality: str):
        if modality in ["vis", "v", "visible"]:
            return self.router_v, self.ops_v_priv
        if modality in ["ir", "i", "infrared"]:
            return self.router_i, self.ops_i_priv
        raise ValueError(f"Unknown modality: {modality}")

    def forward(self, Fm, modality="vis", return_aux=True):
        B, C, H, W = Fm.shape
        assert C == self.C

        router, ops_priv = self._select_router(modality)

        mu, std = self._patch_stats(Fm, self.P)
        d = torch.cat([mu, std], dim=1)  # [B,2C,Hp,Wp]
        w_patch = F.softmax(router(d) / self.tau, dim=1)  # [B,E,Hp,Wp]

        g_patch = self._gate_from_w(w_patch)  # [B,1,Hp,Wp]

        w_pix = self._upsample_patch_map(w_patch, self.P, H, W)  # [B,E,H,W]

        g_pix = self._upsample_patch_map(g_patch, self.P, H, W)  # [B,1,H,W]

        delta = torch.zeros_like(Fm)
        for e in range(self.E):
            we = w_pix[:, e:e + 1]

            if e < self.E_shared:
                op = self.ops_shared[e]
            else:
                op = ops_priv[e - self.E_shared]

            delta = delta + we * op(Fm)

        F_tilde = Fm + delta

        if return_aux:
            return F_tilde, {
                "w_patch": w_patch,
                "g_patch": g_patch,
                "g_pix": g_pix
            }
        return F_tilde


class CrossGateMix1D(nn.Module):
    def __init__(self, embed_dim: int, context_kernel: int = 5, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.dwconv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=context_kernel,
            padding=context_kernel // 2,
            groups=embed_dim
        )
        self.gate_x = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mix_y = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim)
        )

        self.gate_y = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mix_x = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def _ctx(self, t):
        # t: [S,B,E]
        tb = t.permute(1, 2, 0)  # [B,E,S]
        tb = self.dwconv(tb)
        return tb.permute(2, 0, 1)  # [S,B,E]

    def forward(self, xl, xs, conf_seq=None):
        xl_ctx = self._ctx(xl)
        xs_ctx = self._ctx(xs)

        if conf_seq is not None:
            conf_e = conf_seq.expand(-1, -1, xl.size(-1)).to(xl.dtype)
        else:
            conf_e = 1.0

        g_l = torch.sigmoid(self.gate_x(xl_ctx))
        m_s = self.mix_y(xs_ctx)
        xl_out = xl + self.dropout(conf_e * g_l * m_s)

        g_s = torch.sigmoid(self.gate_y(xs_ctx))
        m_l = self.mix_x(xl_ctx)
        xs_out = xs + self.dropout(conf_e * g_s * m_l)

        return xl_out, xs_out


class Crossgate(nn.Module):
    """
    Symmetric dual-input multi-scale CrossGate for registration loop.

    - both ref and mov are encoded into large-scale/context and small-scale/detail branches
    - gated interaction is performed at each scale
    - interacted multi-scale features are fused into interacted ref/mov features
    """

    def __init__(
            self,
            in_ch: int = 8,
            patch_size: int = 4,
            channel: int = 4,
            dropout: float = 0.1,
            use_self_attn: bool = False
    ):
        super().__init__()

        self.P = patch_size
        self.C = channel
        self.in_ch = in_ch
        self.embed_dim = self.C * self.P * self.P
        # self.embed_dim=1
        self.use_self_attn = use_self_attn

        # -----------------------------
        # shared-scale encoders
        # -----------------------------
        self.ctx_encoder = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_ch, channel, 3, 1), nn.InstanceNorm2d(channel), nn.PReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(channel, channel, 3, 1), nn.InstanceNorm2d(channel), nn.PReLU(),
        )
        self.det_encoder = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_ch, channel, 3, 1), nn.InstanceNorm2d(channel), nn.PReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(channel, channel, 3, 1), nn.InstanceNorm2d(channel), nn.PReLU(),
        )

        # context / large receptive field
        self.ctx_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 7, padding=3), nn.ReLU(inplace=True),
        )
        # detail / small receptive field
        self.det_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0), nn.ReLU(inplace=True),
        )

        # token PE / optional self-attn
        self.pe = PositionalEncodingSBE(self.embed_dim, dropout=dropout, max_len=4096)
        if use_self_attn:
            self.SA_ctx_ref = nn.MultiheadAttention(self.embed_dim, num_heads=1, dropout=dropout, batch_first=False)
            self.SA_ctx_mov = nn.MultiheadAttention(self.embed_dim, num_heads=1, dropout=dropout, batch_first=False)
            self.SA_det_ref = nn.MultiheadAttention(self.embed_dim, num_heads=1, dropout=dropout, batch_first=False)
            self.SA_det_mov = nn.MultiheadAttention(self.embed_dim, num_heads=1, dropout=dropout, batch_first=False)

        # gated interaction at each scale
        self.XGate_ctx = CrossGateMix1D(
            embed_dim=self.embed_dim,
            context_kernel=5,
            mlp_ratio=2.0,
            dropout=dropout
        )
        self.XGate_det = CrossGateMix1D(
            embed_dim=self.embed_dim,
            context_kernel=5,
            mlp_ratio=2.0,
            dropout=dropout
        )

        # fuse multi-scale interacted ref/mov features
        self.ref_fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel * 2, in_ch, 3, 1),
            nn.InstanceNorm2d(in_ch),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, in_ch, 3, 1),
            nn.InstanceNorm2d(in_ch),
            nn.PReLU(),
        )
        self.mov_fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel * 2, in_ch, 3, 1),
            nn.InstanceNorm2d(in_ch),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, in_ch, 3, 1),
            nn.InstanceNorm2d(in_ch),
            nn.PReLU(),
        )

    def _self_attn(self, t, mha):
        qkv = self.pe(t)
        out = mha(qkv, qkv, qkv, need_weights=False)[0]
        return out + t

    def _pad_to_patch(self, x):
        B, C, H, W = x.shape
        pad_h = (self.P - H % self.P) % self.P
        pad_w = (self.P - W % self.P) % self.P
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W, pad_h, pad_w

    def _tokens_from_feat(self, feat):
        feat, H0, W0, pad_h, pad_w = self._pad_to_patch(feat)
        Hpad, Wpad = feat.shape[2], feat.shape[3]
        unfold = nn.Unfold(kernel_size=(self.P, self.P), stride=self.P)
        fold = nn.Fold(output_size=(Hpad, Wpad), kernel_size=(self.P, self.P), stride=self.P)
        tok = unfold(feat).transpose(1, 2).transpose(0, 1).contiguous()  # [S,B,E]
        return tok, fold, H0, W0, pad_h, pad_w, Hpad, Wpad

    def _feat_from_tokens(self, tok, fold, H0, W0):
        feat = fold(tok.transpose(0, 1).transpose(1, 2).contiguous())
        feat = feat[:, :, :H0, :W0]
        return feat

    def forward(self, x_ref, x_mov=None, conf_map=None, return_aux=False):
        if x_mov is None:
            x_mov = x_ref

        # -----------------------------
        # multi-scale features for both ref and mov
        # -----------------------------
        ref_ctx = self.ctx_conv(self.ctx_encoder(x_ref))
        ref_det = self.det_conv(self.det_encoder(x_ref))

        mov_ctx = self.ctx_conv(self.ctx_encoder(x_mov))
        mov_det = self.det_conv(self.det_encoder(x_mov))

        # -----------------------------
        # tokenize each scale
        # -----------------------------
        tr_ctx, fold_ctx_r, H0, W0, pad_h, pad_w, Hpad, Wpad = self._tokens_from_feat(ref_ctx)
        tm_ctx, fold_ctx_m, _, _, _, _, _, _ = self._tokens_from_feat(mov_ctx)

        tr_det, fold_det_r, _, _, _, _, _, _ = self._tokens_from_feat(ref_det)
        tm_det, fold_det_m, _, _, _, _, _, _ = self._tokens_from_feat(mov_det)

        conf_seq = None
        if conf_map is not None:
            if pad_h > 0 or pad_w > 0:
                conf_map = F.pad(conf_map, (0, pad_w, 0, pad_h), mode="reflect")
            conf_p = F.avg_pool2d(conf_map, kernel_size=self.P, stride=self.P)
            conf_seq = conf_p.flatten(2).permute(2, 0, 1).contiguous()  # [S,B,1]

        # -----------------------------
        # optional self-attention
        # -----------------------------
        if self.use_self_attn:
            tr_ctx = self._self_attn(tr_ctx, self.SA_ctx_ref)
            tm_ctx = self._self_attn(tm_ctx, self.SA_ctx_mov)
            tr_det = self._self_attn(tr_det, self.SA_det_ref)
            tm_det = self._self_attn(tm_det, self.SA_det_mov)

        # -----------------------------
        # gated interaction at each scale
        # -----------------------------
        tr_ctx_i, tm_ctx_i = self.XGate_ctx(tr_ctx, tm_ctx, conf_seq=conf_seq)
        tr_det_i, tm_det_i = self.XGate_det(tr_det, tm_det, conf_seq=conf_seq)

        # -----------------------------
        # fold back to 2D
        # -----------------------------
        ref_ctx_i = self._feat_from_tokens(tr_ctx_i, fold_ctx_r, H0, W0)
        mov_ctx_i = self._feat_from_tokens(tm_ctx_i, fold_ctx_m, H0, W0)

        ref_det_i = self._feat_from_tokens(tr_det_i, fold_det_r, H0, W0)
        mov_det_i = self._feat_from_tokens(tm_det_i, fold_det_m, H0, W0)

        # -----------------------------
        # multi-scale fusion
        # -----------------------------
        Fr_int = self.ref_fuse(torch.cat([ref_ctx_i, ref_det_i], dim=1))
        Fm_int = self.mov_fuse(torch.cat([mov_ctx_i, mov_det_i], dim=1))

        if return_aux:
            return Fr_int, Fm_int, {
                "ref_ctx_feat": ref_ctx_i,
                "mov_ctx_feat": mov_ctx_i,
                "ref_det_feat": ref_det_i,
                "mov_det_feat": mov_det_i,
            }
        return Fr_int, Fm_int


def flow_to_grid(flow, base_grid):
    return (base_grid + flow).clamp(-1, 1)


def warp(x, grid_y2x):
    return F.grid_sample(
        x, grid_y2x,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )


def compose_grids(grid_prev, grid_delta):
    """
    Compose pull-back grids:
    new_grid = grid_prev o grid_delta
    both: [B,H,W,2]
    """
    g = grid_prev.permute(0, 3, 1, 2).contiguous()
    out = F.grid_sample(
        g, grid_delta,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )
    return out.permute(0, 2, 3, 1).contiguous()


class AlignStateEmbedding(nn.Module):
    """
    Alignment-state embedding without gate.
    Encodes current mismatch / deformation / confidence status.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _mean(x):
        return x.flatten(1).mean(1)

    def forward(self, diff, flow2, conf2=None):
        dx = flow2[:, 0]
        dy = flow2[:, 1]

        if conf2 is None:
            conf_mean = torch.zeros(
                diff.size(0),
                device=diff.device,
                dtype=diff.dtype
            )
        else:
            conf_mean = self._mean(conf2)

        stats = torch.stack([
            self._mean(diff.abs()),
            torch.sqrt(diff.pow(2).flatten(1).mean(1) + 1e-8),
            self._mean(dx.abs()),
            self._mean(dy.abs()),
            dx.flatten(1).std(dim=1, unbiased=False),
            dy.flatten(1).std(dim=1, unbiased=False),
        ], dim=1)  # [B,7]

        return self.mlp(stats)


class ConvGNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, groups=8):
        super().__init__()
        g = min(groups, cout)
        while cout % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.GroupNorm(g, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EncoderHalf(nn.Module):
    """
    output: base*2 channels, spatial /2
    """

    def __init__(self, in_ch=8, base=32):
        super().__init__()
        self.c0 = nn.Sequential(
            ConvGNReLU(in_ch, base),
            ConvGNReLU(base, base)
        )
        self.c1 = nn.Sequential(
            ConvGNReLU(base, base * 2, s=2),
            ConvGNReLU(base * 2, base * 2)
        )

    def forward(self, x):
        x = self.c0(x)
        x = self.c1(x)
        return x


class DeformStateUpdate(nn.Module):
    """
    State-conditioned deformation predictor using interacted ref/mov features.
    """

    def __init__(self, feat_ch, hid_ch=None, sdim=64):
        super().__init__()
        hid_ch = feat_ch if hid_ch is None else hid_ch

        # Fr_int, Fm_int, flow2
        cin = feat_ch * 2 + 2
        self.pre = nn.Sequential(
            ConvGNReLU(cin, hid_ch),
            ConvGNReLU(hid_ch, hid_ch),
        )

        self.film = nn.Linear(sdim, hid_ch * 2)

        self.to_vel = nn.Conv2d(hid_ch, 2, 3, 1, 1)
        nn.init.normal_(self.to_vel.weight, mean=0.0, std=2e-3)
        nn.init.zeros_(self.to_vel.bias)

        self.to_conf = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch // 2, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, Fr_int, Fm_int, flow2, state_emb):
        x = torch.cat([Fr_int, Fm_int, flow2], dim=1)
        x = self.pre(x)

        gb = self.film(state_emb)
        B, C, _, _ = x.shape
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        x = x * (1.0 + gamma) + beta

        vel2 = self.to_vel(x)
        conf2 = self.to_conf(x)
        return vel2, conf2


class DenoiseReg(nn.Module):
    """
    Cross-Gated Deformation Evolution Registration
    - each step warps moving feature
    - each step interacts Fref and current Fmov through CrossGate
    - updater predicts deformation from interacted ref/mov features
    - update by deformation composition
    ref/mov: [B,8,H,W]
    """

    def __init__(
            self,
            base=32,
            T=8,
            max_disp=0.20,
            step_size=0.08,
            cross_patch=4,
            cross_dropout=0.1,
            cross_self_attn=False,
    ):
        super().__init__()
        self.T = int(T)
        self.max_disp = float(max_disp)
        self.step_size = float(step_size)

        self.enc = EncoderHalf(in_ch=8, base=base)
        feat_ch = base * 2

        self.crossgate   = Crossgate(
            in_ch=feat_ch,
            patch_size=cross_patch,
            channel=max(8, feat_ch // 4),
            # channel=1,
            dropout=cross_dropout,
            use_self_attn=cross_self_attn
        )

        self.state_embed = AlignStateEmbedding(dim=64)
        self.updater = DeformStateUpdate(feat_ch=feat_ch, hid_ch=feat_ch, sdim=64)

    @staticmethod
    def _down_hw2(flow_hw2, h, w):
        f = flow_hw2.permute(0, 3, 1, 2).contiguous()
        f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=True)
        return f

    @staticmethod
    def _up_2hw(flow_2hw, H, W):
        f = F.interpolate(flow_2hw, size=(H, W), mode="bilinear", align_corners=True)
        return f.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _resize_map(x, H, W):
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

    def forward(self, ref, mov, return_steps=False):
        B, _, H, W = ref.shape
        device, dtype = ref.device, ref.dtype

        base_grid = make_base_grid(B, H, W, device, dtype)
        phi = base_grid.clone()

        Fref = self.enc(ref)
        h2, w2 = Fref.shape[2], Fref.shape[3]

        prev_conf2 = None
        steps_feat = []
        grid_steps = []
        flow_steps = []
        conf_preds = []
        feat_preds = []

        for s in range(self.T):
            mov_w = warp(mov, phi)
            Fmov = self.enc(mov_w)

            Fr_int, Fm_int, aux_cg = self.crossgate(
                Fref, Fmov,
                conf_map=None,
                return_aux=True
            )

            flow_now = phi - base_grid
            flow2 = self._down_hw2(flow_now, h2, w2)

            state_emb = self.state_embed(
                Fr_int - Fm_int,
                flow2,
                prev_conf2
            )

            vel2_raw, conf2 = self.updater(
                Fr_int, Fm_int, flow2, state_emb
            )

            conf2 = kornia.filters.gaussian_blur2d(conf2, (7, 7), (2.0, 2.0))
            vel2 = torch.tanh(vel2_raw) * self.step_size
            vel2 = vel2 * conf2
            vel2 = vel2.clamp(-0.20, 0.20)
            vel2 = kornia.filters.gaussian_blur2d(vel2, (7, 7), (2.0, 2.0))

            vel = self._up_2hw(vel2, H, W)
            delta_grid = flow_to_grid(vel, base_grid)

            phi = compose_grids(phi, delta_grid).clamp(-1, 1)
            flow_total = phi - base_grid
            flow_total = self.max_disp * torch.tanh(flow_total / self.max_disp)
            phi = flow_to_grid(flow_total, base_grid)

            prev_conf2 = conf2

            conf_preds.append(self._resize_map(conf2, H, W))
            feat_preds.append(self._resize_map(Fm_int.mean(dim=1, keepdim=True), H, W))

            if return_steps:
                steps_feat.append(warp(mov, phi))  # 还是保留，调试时可能有用
                grid_steps.append(phi.clone())  # 关键：每一步 grid
                flow_steps.append((phi - base_grid).clone())  # 关键：每一步 flow

        out = {
            "flow": phi - base_grid,
            "grid": phi,
            "warp": warp(mov, phi),
            "conf_preds": conf_preds,
            "feat_preds": feat_preds,
        }

        if return_steps:
            out["warp_steps"] = steps_feat
            out["grid_steps"] = grid_steps
            out["flow_steps"] = flow_steps

        return out