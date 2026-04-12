#loss.py
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from model import model

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def Loss_intensity(vis, ir, image_fused):
    assert (vis.size() == ir.size() == image_fused.size())
    ir_li = F.l1_loss(image_fused, ir)
    vis_li = F.l1_loss(image_fused, vis)
    li = ir_li + vis_li
    return li

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, img1, img2, image_fused=None):
        if image_fused == None:
            image_1_Y = img1[:, :1, :, :]
            image_2_Y = img2[:, :1, :, :]
            gradient_1 = self.sobelconv(image_1_Y)
            gradient_2 = self.sobelconv(image_2_Y)
            Loss_gradient = F.l1_loss(gradient_1, gradient_2)
            return Loss_gradient
        else:
            image_1_Y = img1[:, :1, :, :]
            image_2_Y = img2[:, :1, :, :]
            image_fused_Y = image_fused[:, :1, :, :]
            gradient_1 = self.sobelconv(image_1_Y)
            gradient_2 = self.sobelconv(image_2_Y)
            gradient_fused = self.sobelconv(image_fused_Y)
            gradient_joint = torch.max(gradient_1, gradient_2)
            Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
            return Loss_gradient


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]

        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class CorrelationCoefficient(nn.Module):
    def __init__(self):
        super(CorrelationCoefficient, self).__init__()

    def c_CC(self, A, B):
        A_mean = torch.mean(A, dim=[2, 3], keepdim=True)
        B_mean = torch.mean(B, dim=[2, 3], keepdim=True)
        A_sub_mean = A - A_mean
        B_sub_mean = B - B_mean
        sim = torch.sum(torch.mul(A_sub_mean, B_sub_mean))
        A_sdev = torch.sqrt(torch.sum(torch.pow(A_sub_mean, 2)))
        B_sdev = torch.sqrt(torch.sum(torch.pow(B_sub_mean, 2)))
        out = sim / (A_sdev * B_sdev)
        return out

    def forward(self, A, B, Fusion=None):
        if Fusion is None:
            CC = self.c_CC(A, B)
        else:
            r_1 = self.c_CC(A, Fusion)
            r_2 = self.c_CC(B, Fusion)
            CC = (r_1 + r_2) / 2
        return CC

class L_softmatch_topk(nn.Module):
    """
    不用 GT 矩阵的匹配监督：
      1) soft-argmax 坐标回归（SmoothL1）
      2) topk 命中概率 -log(p_true)

    输入：
      topk_idx: [Wn,B,Nt,K]
      topk_w:   [Wn,B,Nt,K]
      index_r:  [B,2,H*W]（你的 ImageTransform 输出）
    """
    def __init__(self, height=256, weight=256, eps=1e-8):
        super().__init__()
        self.height = height
        self.weight = weight
        self.eps = eps

    def forward(self, topk_idx, topk_w, index_r, small_window_size: int, large_window_size: int):
        device = topk_idx.device
        Wn, B, Nt, K = topk_idx.shape
        s = small_window_size
        L = large_window_size
        Ns = L * L

        # --- 预计算每个窗口的绝对索引表（与旧 L_correspondence 同源，但更严谨映射） ---
        base_index = torch.arange(0, self.height * self.weight, device=device, dtype=torch.float32)\
                          .reshape(self.height, self.weight)[None, None]  # [1,1,H,W]

        unfold_sw = nn.Unfold(kernel_size=(s, s), stride=s)
        sw_abs_all = unfold_sw(base_index)[0]                 # [Nt, Wn]
        sw_abs = sw_abs_all.transpose(0, 1).contiguous()      # [Wn, Nt]

        lw_abs_bw = model.df_window_partition(base_index, L, s, is_bewindow=False)  # [1, Ns, Wn]
        lw_abs = lw_abs_bw[0].transpose(0, 1).contiguous()    # [Wn, Ns]

        flow_list = []
        ptr_list  = []

        for i in range(B):
            all_x_abs = index_r[i, 0, :].to(dtype=torch.float32)  # [H*W]
            all_y_abs = index_r[i, 1, :].to(dtype=torch.float32)  # [H*W]

            for j in range(Wn):
                lw_win_abs = lw_abs[j]  # [Ns]
                sw_win_abs = sw_abs[j]  # [Nt]

                # 找到落在该大窗的 y（目标）位置： (lw_pos, global_pos)
                lw_pos, g_pos = (lw_win_abs.unsqueeze(1) == all_y_abs).nonzero(as_tuple=True)
                if g_pos.numel() == 0:
                    continue

                # 对应的 x（源）绝对索引
                x_abs = all_x_abs[g_pos]  # [M]
                # 再落回该小窗： (sw_pos, m_pos) 其中 m_pos 对应 x_abs 的索引
                sw_pos, m_pos = (sw_win_abs.unsqueeze(1) == x_abs).nonzero(as_tuple=True)
                if sw_pos.numel() == 0:
                    continue

                # 对应的 gt 列（相对大窗 Ns 内）
                gt_col = lw_pos[m_pos]  # [M2]
                gt_row = sw_pos         # [M2]

                # 过滤掉 padding 的 (0,0) 对
                sw_abs_sel = sw_win_abs[gt_row]
                lw_abs_sel = lw_win_abs[gt_col]
                valid = (sw_abs_sel != 0) | (lw_abs_sel != 0)
                if valid.sum() == 0:
                    continue

                gt_row = gt_row[valid]
                gt_col = gt_col[valid]

                # ---- 1) 命中概率：p_true = sum_k w * 1[idx==gt] ----
                idx_m = topk_idx[j, i, gt_row, :]              # [M, K]
                w_m   = topk_w[j, i, gt_row, :]                # [M, K]
                hit = (idx_m == gt_col.unsqueeze(1)).float()   # [M, K]
                p_true = (w_m * hit).sum(dim=1).clamp_min(self.eps)  # [M]
                ptr_list.append((-torch.log(p_true)).mean())

                # ---- 2) soft-argmax 坐标回归（在大窗坐标系内） ----
                # pred: E[x], E[y]
                idx_f = idx_m.to(torch.float32)
                xk = torch.remainder(idx_f, L)                 # [M, K]
                yk = torch.floor(idx_f / L)                    # [M, K]
                x_hat = (w_m * xk).sum(dim=1)                  # [M]
                y_hat = (w_m * yk).sum(dim=1)                  # [M]

                x_gt = torch.remainder(gt_col.to(torch.float32), L)
                y_gt = torch.floor(gt_col.to(torch.float32) / L)

                pred_xy = torch.stack([x_hat, y_hat], dim=1)    # [M,2]
                gt_xy   = torch.stack([x_gt,  y_gt ], dim=1)    # [M,2]
                flow_list.append(F.smooth_l1_loss(pred_xy, gt_xy))

        if len(flow_list) == 0:
            loss_flow = torch.tensor(0.0, device=device)
            loss_ptr  = torch.tensor(0.0, device=device)
        else:
            loss_flow = torch.stack(flow_list).mean()
            loss_ptr  = torch.stack(ptr_list).mean()

        return loss_flow, loss_ptr


class L_correspondence(nn.Module):
    def __init__(self, height=256, weight=256, eps=1e-8):
        super(L_correspondence, self).__init__()
        self.height = height
        self.weight = weight
        self.eps = eps

    def forward(self, correspondence_matrixs, index_r):
        """
        correspondence_matrixs: [Wn, B, Nt, Ns]
            ——支持“行稀疏”的 dense 矩阵：每行仅 Top-K 列为非零，未选中的列=0（行和可为1或<1均可）
        index_r: [B, 2, H*W]（与你现有一致）
        """
        size = correspondence_matrixs.size()
        device = correspondence_matrixs.device
        s  = int(math.sqrt(size[2]))  # small_window_size
        L  = int(math.sqrt(size[3]))  # large_window_size
        B  = size[1]
        Wn = size[0]

        # ---- 预计算每个窗口中，小窗/大窗对应的绝对索引表（与你原逻辑一致）----
        base_index = torch.arange(0, self.height * self.weight, device=device).reshape(self.height, self.weight).float()
        base_index = base_index[None, None]  # [1,1,H,W]

        unfold_win = nn.Unfold(kernel_size=(s, s), stride=s)
        sw_abs_all = unfold_win(base_index)[0]                 # [Nt, Wn]
        sw_abs     = sw_abs_all.transpose(0, 1).contiguous()   # [Wn, Nt]

        lw_abs_bw = model.df_window_partition(base_index, L, s, is_bewindow=False)  # [1, Ns, Wn]
        lw_abs    = lw_abs_bw[0].transpose(0, 1).contiguous()  # [Wn, Ns]

        # ---- 逐 batch / 窗口 计算：只在“真值列”上取概率做损失 ----
        ce_list  = []
        l1_list  = []

        for i in range(B):
            # 这一张图的 index_r
            # index_r[i,0,:] = 源全图绝对索引； index_r[i,1,:] = 目标全图绝对索引
            all_x_abs = index_r[i, 0, :]    # [H*W]
            all_y_abs = index_r[i, 1, :]    # [H*W]

            for j in range(Wn):
                # 本窗口的小窗/大窗的绝对索引表
                lw_win_abs = lw_abs[j]      # [Ns]
                sw_win_abs = sw_abs[j]      # [Nt]

                # 把“落在当前大窗”的全图 y 位置筛出来
                # indices: (in_lw_idx, in_all_idx) 使得 lw_win_abs[in_lw_idx] == all_y_abs[in_all_idx]
                in_lw_idx, in_all_idx = (lw_win_abs.unsqueeze(1) == all_y_abs).nonzero(as_tuple=True)

                if in_all_idx.numel() == 0:
                    continue  # 该窗口没有可用的 GT 对应，跳过

                # 这些全图对应的 x 位置再落回当前小窗，得到每个 Nt 行的“真值列”在 Ns 内的索引
                # insw: 找 sw_win_abs == all_x_abs[in_all_idx]
                insw_sw_idx, _ = (sw_win_abs.unsqueeze(1) == all_x_abs[in_all_idx]).nonzero(as_tuple=True)
                # 过滤掉 (0,0) 对（与你原逻辑一致）
                valid_mask = torch.logical_or(
                    sw_win_abs[insw_sw_idx] != 0,
                    lw_win_abs[in_lw_idx]   != 0
                )
                if valid_mask.sum() == 0:
                    continue

                sw_rows = insw_sw_idx[valid_mask]   # 这些是真实的 Nt 行索引
                lw_cols = in_lw_idx[valid_mask]     # 这些是真实的 Ns 列索引

                # 取出预测矩阵（可能是行稀疏：未选中列为0）
                pred = correspondence_matrixs[j, i]  # [Nt, Ns]
                pred = pred / (pred.sum(dim=-1, keepdim=True) + 1e-8)

                # 只在 (row=sw_rows, col=lw_cols) 的真值位置取概率
                p_true = pred[sw_rows, lw_cols].clamp_min(self.eps)  # [M]

                # CE：-log(p_true)
                ce_list.append(-torch.log(p_true).mean())

                # L1（只在真值列）：|p_true - 1|
                l1_list.append((1.0 - p_true).abs().mean())

        # 汇总
        if len(ce_list) == 0:
            loss_ce = torch.tensor(0.0, device=device)
            loss_l1 = torch.tensor(0.0, device=device)
        else:
            loss_ce = torch.stack(ce_list).mean()
            loss_l1 = torch.stack(l1_list).mean()

        return loss_ce, loss_l1