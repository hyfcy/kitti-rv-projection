# -*- coding: utf-8 -*-
"""
自动识别哪些格子是"地面"，并可视化。

================================ 来源标注（写笔记/邮件时务必分清）================================
本脚本用到三类信息，归属完全不同：

  [论文]  OverlapMamba (arXiv:2405.07966) 公式 (5)：
              u = 0.5·[1 − arctan(y,x)/π]·w
              v = [1 − (arcsin(z/r) + f_up) / f]·h
          作用：把 3D 点映射到 (行, 列)。**论文里只有这个，没有任何"地面"相关内容。**
          实现见 rv_projection.py。

  [规格]  HDL-64E 垂直视场角：上 2.0° / 下 24.9°（业界公开参数）
          **论文没有给出这个数值**，是本脚本按 KITTI 常用配置填的默认值。
          最大距离 50 m —— 这个**是论文里的**（论文写明 KITTI 用 50 m）。

  [自推]  地面几何关系： 距离 ≈ h / sin|仰角|      ← 基础平面几何，非论文内容
          推导：传感器离地 h，激光以俯仰角 θ 向下打，落地距离 = h / sin|θ|（直角三角形）。
          同时 h 本身也是**从数据反推的**：h = 距离 × sin|仰角|（取底部若干行）。
          作用：作为**独立于论文**的校验标准 —— 用论文的公式去检验论文自己会变成循环论证。

  ⚠️ 判据的固有歧义（务必知道，否则会过度解读结果）
  -------------------------------------------------------------------------------------
  本判据只能回答"这个格子是否落在一个与地面同距离的平面上"，**不能单独证明它是地面**。

  原因：竖直面在该行的径向距离 = X / cos(θ)（X 为它到车的水平距离），
        地面在该行的径向距离 = h / sin|θ|。
        当 X ≈ h / tan|θ| 时，两者数值相同 → 无法区分。

  实测：row 0（仰角 +1.79°）的地面模型是 55.75 m，
        而一堵位于水平距离 55.7 m 的竖墙在该行的径向距离同样是 55.75 m。

  影响：在**低仰角（上部若干行）歧义很大**；在**底部若干行**（仰角陡、h/tan|θ| 变小）歧义收敛。
        因此输出的"地面格子数占比"应理解为"与地面同距离的格子占比"，不要当作精确的地面分割。

  要真正区分，需要额外信息：点的法向量（地面法向朝上）、邻近行的距离变化趋势、
  或该点是否落在连续的地面网格上。
  -------------------------------------------------------------------------------------

所以：地面几何不是论文的贡献，是为了验证实现正确性而引入的基础几何工具。
=========================================================================================

用法:
    python ground_mask.py kitti_real_00_000844.bin
    python ground_mask.py kitti_real_00_000844.bin --tol 0.30 --outdir out_ground
"""

import re
import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _n in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        font_manager.findfont(_n, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_n]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

import rv_projection as m


# ---------------------------------------------------------------- 加粗精简
# 报告里只保留「含单位的指标」和「方括号标签」两类加粗，其余去掉。
# 理由：自动生成的报告若通篇加粗，等于没有重点，读起来是噪音。
_UNITS = r'(%|％|m|cm|mm|km|GB|MB|KB|TB|s|ms|fps|FPS|°|行|列|帧|秒|米|度|倍|次)'
_RE_METRIC = re.compile(r'^[~约≈]?\s*[\d,]+(\.\d+)?\s*[\u4e00-\u9fff]*\s*' + _UNITS + r'$')
_RE_KILO = re.compile(r'^\d{1,3}(,\d{3})+$')
_RE_LABEL = re.compile(r'^\[[^\[\]]{1,6}\]$')
_RE_BOLD = re.compile(r'\*\*([^*\n]+)\*\*')


def thin_bold(text):
    """只保留指标型/标签型加粗"""
    def keep(t):
        t = t.strip()
        if not t:
            return False
        if _RE_LABEL.match(t) or _RE_KILO.match(t):
            return True
        if re.search(r'\d', t) and _RE_METRIC.match(t):
            return True
        return False

    def repl(m):
        return m.group(0) if keep(m.group(1)) else m.group(1)

    return _RE_BOLD.sub(repl, text)


def per_row_pitch(rows, fov_up=2.0, fov_down=24.9):
    """每行中心的仰角（度）"""
    f = fov_up + fov_down
    return np.array([fov_up - f * ((r + 0.5) / rows) for r in range(rows)])


def calibrate_height(ri, valid, pitch_deg, n_bottom=12):
    """用最底部若干行反推传感器高度 h = 距离 * sin|仰角|"""
    rows = ri.shape[0]
    hs = []
    for row in range(rows - 1, max(0, rows - 1 - n_bottom), -1):
        v = ri[row][valid[row]]
        if v.size < 20:
            continue
        p = math.radians(abs(pitch_deg[row]))
        if p < 1e-3:
            continue
        hs.append(float(np.median(v)) * math.sin(p))
    return (float(np.median(hs)), len(hs)) if hs else (None, 0)


def ground_mask(ri, valid, pitch_deg, h, tol):
    """返回布尔掩码 + 每格偏差"""
    rows, cols = ri.shape
    pred = np.full((rows, cols), np.nan)
    for r in range(rows):
        p = math.radians(abs(pitch_deg[r]))
        if p < 1e-3:
            continue
        pred[r, :] = h / math.sin(p)
    diff = ri - pred
    mask = valid & (np.abs(diff) <= tol)
    return mask, diff, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_path")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=900)
    ap.add_argument("--max-range", type=float, default=50.0)
    ap.add_argument("--tol", type=float, default=0.35, help="与地面模型的容差(米)")
    ap.add_argument("--outdir", default="out_ground")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pc = m.load_kitti_bin(args.bin_path)
    ri, valid, st = m.project_to_rv(pc, args.rows, args.cols, max_range=args.max_range)

    pitch_deg = per_row_pitch(args.rows)
    h, n = calibrate_height(ri, valid, pitch_deg)
    if h is None:
        raise SystemExit("无法反推传感器高度（底部行数据太少），试试 --rows/--max-range")

    print("传感器高度（由底部 %d 行反推）: %.2f m" % (n, h))

    mask, diff, pred = ground_mask(ri, valid, pitch_deg, h, args.tol)
    print("有数据的格子: %d" % int(valid.sum()))
    print("判定为地面的格子: %d (%.1f%% of 有效格子)"
          % (int(mask.sum()), 100.0 * mask.sum() / max(1, valid.sum())))
    print("容差: ±%.2f m" % args.tol)

    # ---- 图 1：地面掩码叠加在原距离图上
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), dpi=140)

    im0 = axes[0].imshow(ri, cmap="turbo", aspect="auto",
                         vmin=0, vmax=args.max_range, interpolation="nearest")
    axes[0].set_title("距离图（原始）—— 底部深蓝=近处地面，上方橙红=远处物体", fontsize=10)
    fig.colorbar(im0, ax=axes[0], pad=0.01, label="距离 (m)")

    axes[1].imshow(ri, cmap="turbo", aspect="auto",
                   vmin=0, vmax=args.max_range, interpolation="nearest", alpha=0.45)
    overlay = np.ma.masked_where(~mask, np.ones_like(ri))
    axes[1].imshow(overlay, cmap="autumn_r", aspect="auto", interpolation="nearest", alpha=0.9)
    axes[1].set_title("叠加识别结果：红色 = 符合「地面平面几何」的格子（容差 ±%.2f m）"
                      "    白色/浅色 = 未被判为地面" % args.tol, fontsize=10)
    axes[1].set_xlabel("列（yaw / 方位角）")
    for ax in axes:
        ax.set_ylabel("行（pitch）")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "ground_overlay.png"))
    plt.close(fig)

    # ---- 图 2：取几列，画「实测 vs 地面模型」
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), dpi=140, sharey=True)
    for ax, c in zip(axes, (100, 200, 450, 750)):
        rows = np.arange(args.rows)
        meas = ri[:, c].copy()
        meas[~valid[:, c]] = np.nan
        ax.plot(meas, rows, "o-", ms=3, lw=1, label="实测")
        ax.plot(pred[:, c], rows, "--", lw=1, color="green", label="地面模型 h/sin|θ|")
        ax.invert_yaxis()
        ax.set_title("第 %d 列" % c, fontsize=9)
        ax.set_xlabel("距离 (m)")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("行号")
            ax.legend(fontsize=7)
    fig.suptitle("实测距离 vs 地面模型：两条线重合的区段就是地面", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "ground_columns.png"))
    plt.close(fig)

    # ---- 报告
    rows_ok = [r for r in range(args.rows) if mask[r].sum() > 30]
    lines = [
        "# 地面识别结果",
        "",
        "- 数据: `%s`" % os.path.basename(args.bin_path),
        "- 传感器高度（由底部 %d 行反推）: **%.2f m**" % (n, h),
        "- 判据: 距离与 `h / sin|仰角|` 的偏差 ≤ %.2f m" % args.tol,
        "",
        "> ⚠️ **判据的固有歧义**：本结果应理解为「落在与地面同距离平面上的格子」，",
        "> **不等于**精确的地面分割。原因是竖直面在该行的径向距离 `X/cos(θ)` 与地面的",
        "> `h/sin|θ|` 在 `X ≈ h/tan|θ|` 时数值相同，**在低仰角行无法区分**。",
        "> 底部若干行（仰角陡）歧义较小。要严格区分需要点的法向量等额外信息。",
        "",
        "- 有数据的格子: %d" % int(valid.sum()),
        "- 判定为地面的格子: **%d**（占有效格子 %.1f%%）"
        % (int(mask.sum()), 100.0 * mask.sum() / max(1, valid.sum())),
        "",
        "## 每行里属于地面的格子数",
        "",
        "| 行号 | 仰角 | 地面在水平方向的距离 h/tan\\|θ\\| | 该行地面格子数 | 该行总格子数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in range(args.rows):
        p = abs(math.radians(pitch_deg[r]))
        hd = ("%.1f m" % (h / math.tan(p))) if p > 1e-3 else "∞"
        lines.append("| %d | %+.2f° | %s | %d | %d |"
                     % (r, pitch_deg[r], hd, int(mask[r].sum()), int(valid[r].sum())))
    lines += [
        "",
        "## 结论",
        "",
        "地面主要集中在**下方若干行**（仰角越朝下，地面越近）。",
        "上方的行若不符合地面模型，说明该方向上被竖直结构（墙、车身）占据。",
        "",
        "产出文件:",
        "- `ground_overlay.png` — 原图 + 地面识别叠加",
        "- `ground_columns.png` — 四列的「实测 vs 模型」对照折线",
        "",
    ]
    with open(os.path.join(args.outdir, "ground_report.md"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(thin_bold("\n".join(lines)))

    print("产出目录:", args.outdir)
    print("  ground_overlay.png / ground_columns.png / ground_report.md")


if __name__ == "__main__":
    main()
