# -*- coding: utf-8 -*-
"""
RV 投影脚本：把一帧 KITTI LiDAR 点云投成 64x900 的距离图 (Range View)

对应论文: OverlapMamba (arXiv:2405.07966) 公式 (5)

用法:
    python rv_projection.py <点云.bin> [--rows 64] [--cols 900] [--outdir out]
    python rv_projection.py --demo          # 没数据时用合成点云自测

输出 (默认写到脚本同目录的 out/):
    rv_range.png        距离图 (越近越红)
    rv_valid.png        有效像素掩码 (哪些格子有点)
    pointcloud_top.png  俯视图 (用于对照)
    report.md           统计数据 + 观察记录模板
"""

import re
import argparse
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接存文件
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 让图里的中文能正常显示（Windows 自带字体优先）
for _name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        font_manager.findfont(_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_name]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

# ---------------------------------------------------------------- 参数
KITTI_FOV_UP_DEG = 2.0     # 上视场角
KITTI_FOV_DOWN_DEG = 24.9  # 下视场角 (合计约 26.9 度，HDL-64E 规格)
MAX_RANGE_M = 50.0         # 论文里 KITTI 用的最大距离，用于过滤远点


# ---------------------------------------------------------------- 读取
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


def load_kitti_bin(path):
    """KITTI velodyne .bin: 每行 4 个 float32 = x, y, z, intensity"""
    pc = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pc[:, :4]


def make_demo_points(n=20000, seed=0):
    """合成一帧"像个街道"的点云，用于没有数据时自测脚本是否跑通"""
    rng = np.random.default_rng(seed)
    # 地面：半径 5~40m 的环带
    r = rng.uniform(5, 40, n // 2)
    theta = rng.uniform(-math.pi, math.pi, n // 2)
    ground = np.stack([r * np.cos(theta), r * np.sin(theta), rng.normal(-1.7, 0.05, n // 2)], 1)
    # 两侧"墙/车"：竖直面
    m = n - n // 2
    x = rng.uniform(-30, 30, m)
    side = rng.choice([-1.0, 1.0], m)
    wall = np.stack([x, side * rng.uniform(4, 12, m), rng.uniform(-1.7, 1.5, m)], 1)
    pts = np.concatenate([ground, wall], 0)
    inten = rng.uniform(0, 1, (pts.shape[0], 1))
    return np.concatenate([pts, inten], 1).astype(np.float32)


# ---------------------------------------------------------------- 投影
def project_to_rv(points, rows=64, cols=900,
                  fov_up_deg=KITTI_FOV_UP_DEG, fov_down_deg=KITTI_FOV_DOWN_DEG,
                  max_range=MAX_RANGE_M):
    """
    论文公式 (5):
        u = 0.5 * [1 - atan2(y, x) / pi] * w
        v = [1 - (asin(z / r) + f_up) / f] * h

    角度约定（容易搞错，务必留意）：
        论文里 f_up 指"朝下那部分的幅度"，pitch 取"向下为正"的约定。
        本实现沿用 KITTI/LiDAR 常见约定：pitch = arcsin(z/r)，
        **朝上为正、朝下为负**。两者相差一个负号，因此 v 的写法为：
            v = (1 - (pitch + fov_down) / f) * h
        效果：pitch = -fov_down（最朝下）→ v = h（图像底行，符合直觉）
              pitch = +fov_up  （最朝上）→ v = 0（图像顶行）
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points[:, :3], axis=1)

    keep = r > 1e-6                      # 去掉原点附近的无效点
    if max_range is not None:
        keep &= r <= max_range
    x, y, z, r = x[keep], y[keep], z[keep], r[keep]

    fov_up = math.radians(fov_up_deg)
    fov_down = math.radians(fov_down_deg)
    f = fov_up + fov_down                # 垂直总视场角

    yaw = np.arctan2(y, x)               # [-pi, pi]
    pitch = np.arcsin(np.clip(z / r, -1.0, 1.0))   # 朝上为正

    u = 0.5 * (1.0 - yaw / math.pi) * cols
    v = (1.0 - (pitch + fov_down) / f) * rows

    u = np.floor(u).astype(np.int64)
    v = np.floor(v).astype(np.int64)

    inside = (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
    u, v, r_in = u[inside], v[inside], r[inside]

    range_image = np.full((rows, cols), np.nan, dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=bool)

    # 同一格子有多个点时取最近的那个（标准做法，避免被远处点覆盖）
    order = np.argsort(-r_in)            # 从远到近写入，近的覆盖远的
    range_image[v[order], u[order]] = r_in[order]
    valid[v[order], u[order]] = True

    stats = {
        "total_points": int(points.shape[0]),
        "kept_points": int(u.shape[0]),
        "dropped_out_of_fov": int(points.shape[0] - u.shape[0]),
        "filled_cells": int(valid.sum()),
        "cells": int(rows * cols),
        "fill_ratio": float(valid.sum()) / float(rows * cols),
    }
    return range_image, valid, stats


# ---------------------------------------------------------------- 统计
def analyze(points, range_image, valid, rows, cols, max_range, n_points_in_fov):
    r = np.linalg.norm(points[:, :3], axis=1)
    r = r[r > 1e-6]
    lines = []
    lines.append("## 1. 点云本身")
    lines.append("")
    lines.append(f"- 总点数: **{points.shape[0]}**")
    lines.append(f"- 距离分布 (m): 最小 {r.min():.2f} / 中位数 {np.median(r):.2f} / 均值 {r.mean():.2f} / 最大 {r.max():.2f}")
    for q in (1, 5, 25, 50, 75, 95, 99):
        lines.append(f"  - {q}% 分位: {np.percentile(r, q):.2f} m")
    lines.append(f"- z 范围 (m): {points[:,2].min():.2f} ~ {points[:,2].max():.2f}（地面大约在 -1.7 m）")
    lines.append(f"- intensity 范围: {points[:,3].min():.3f} ~ {points[:,3].max():.3f}")
    lines.append("")

    lines.append("## 2. 投影结果")
    lines.append("")
    lines.append(f"- 图像尺寸: {rows} (行/垂直) × {cols} (列/水平)，单通道，共 {rows*cols} 个格子")
    lines.append(f"- 落进网格的**点**数: **{n_points_in_fov}**（= 总点数 减去 超出视场被丢弃的点）")
    lines.append(f"- 被占据的**格子**数: **{int(valid.sum())}**")
    if valid.sum() > 0:
        lines.append(f"  - 平均每个格子挤了 **{n_points_in_fov/valid.sum():.2f}** 个点 "
                     f"→ 这就是「同一格子有多个点」的来源，脚本按**距离取最近**的那个写入")
    lines.append(f"- 填充率 (稀疏度): **{valid.mean()*100:.2f}%** → 即约 {100-valid.mean()*100:.2f}% 的格子是空的")
    lines.append(f"- 距离阈值: 只保留 {max_range:.0f} m 内的点")
    lines.append("")
    if valid.any():
        vals = range_image[valid]
        lines.append(f"- 图上距离值范围: {vals.min():.2f} ~ {vals.max():.2f} m")
        lines.append("")

    lines.append("## 3. 逐行距离（行号越小 = 激光越朝上）")
    lines.append("")
    lines.append("> 用途：逐行看距离怎么变，判断每一行打到的是什么。三条判据：")
    lines.append(">")
    lines.append("> | 纵向现象 | 说明 |")
    lines.append("> | --- | --- |")
    lines.append("> | 距离**随行号平滑递减** | 地面（角度越朝下、地面越近） |")
    lines.append("> | 距离**在某一列内恒定不变** | 竖直面（墙、车身侧面） |")
    lines.append("> | 距离**少数几次台阶式跳变** | 多个水平面叠加（车顶、台阶、路缘） |")
    lines.append("")
    lines.append("| 行号 | 该行仰角 | 格子数 | 距离中位 | 距离范围 |")
    lines.append("| --- | --- | --- | --- | --- |")
    fov_up, fov_down = 2.0, 24.9
    f = fov_up + fov_down
    for row in range(rows):
        v = range_image[row][valid[row]]
        if v.size == 0:
            pitch = fov_up - f * ((row + 0.5) / rows)   # 顶行=+2度, 底行=-24.9度
            lines.append(f"| {row} | {pitch:+.2f}° | 0 | — | — |")
            continue
        pitch = fov_up - f * ((row + 0.5) / rows)   # 顶行=+2度, 底行=-24.9度
        lines.append(f"| {row} | {pitch:+.2f}° | {v.size} | {np.median(v):.2f} m | "
                     f"{v.min():.2f} ~ {v.max():.2f} m |")
    lines.append("")

    lines.append("## 4. 复现任务单（修订版 2026-09-27）")
    lines.append("")
    lines.append("> **说明**：早先版本的题目基于有 bug 的投影结果（数据只出现在 15 行、填充率被算成 16.73%），")
    lines.append("> 其中「顶部为什么空」是 bug 造成的假现象。修正后题目重写如下。")
    lines.append("")
    lines.append("> **本复现的 AI 使用说明**：脚本代码由 AI 辅助编写，验证用计算亦由 AI 辅助完成；")
    lines.append("> 参数核对、异常发现（角度符号 bug）、结果判断由本人负责。详见 `复现日志.md` 第 0 节。")
    lines.append("")
    lines.append("**这次复现要交付什么**（只有三样，别贪多）")
    lines.append("")
    lines.append("| # | 要素 | 判据 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 1 | 独立跑通代码 | 报告里的数字能手算复核 |")
    lines.append("| 2 | 验证结果正确 | 有**独立于论文**的校验（几何关系 / 已知常数） |")
    lines.append("| 3 | 发现并解决过问题 | 能说清：现象 → 排查方法 → 根因 → 修复 |")
    lines.append("")
    lines.append("**读图三条底线**")
    lines.append("")
    lines.append("1. **先看图例（colorbar）再下判断**。数值低=蓝(近)，数值高=红(远)。")
    lines.append("2. **判断「像什么」之前，先找能算的指标**。眼睛会骗人，数字不会。")
    lines.append("3. **引用任何数字前，回报告核对**，不要凭印象填。")
    lines.append("")
    lines.append("**问题 1：地面在哪里？怎么用数据确认？**")
    lines.append("")
    lines.append("- 先校准：用底部若干行反推传感器高度 `h = 距离 × sin|仰角|`，看是否收敛到一个常数。")
    lines.append("- 再判定：比较实测距离与 `h / sin|仰角|`，偏差小的即落在与地面同距离的平面上。")
    lines.append("- 说明为什么地面距离**不可能接近 0**（提示：最陡俯仰角 = 几何盲区）。")
    lines.append("- ⚠️ 必须写出的局限：该判据有**固有歧义** —— 竖直面在低仰角行会算出与地面相同的距离，")
    lines.append("  所以它只能证明「与地面同距离」，不能单独证明「是地面」。")
    lines.append("")
    lines.append("**问题 2：图里那些白色横条是什么？**")
    lines.append("")
    lines.append("- 白色 = 该格子没有数据。要分清四种成因：")
    lines.append("  ① 该方向**没有激光束**（64 条线覆盖不满全部格子）")
    lines.append("  ② 射了但**没有回波**（天空方向、吸光材质、超量程）")
    lines.append("  ③ **被距离阈值裁掉**（脚本设了 50 m）")
    lines.append("  ④ **被近处物体遮挡**")
    lines.append("- 追问：为什么这些空条是**成排横向**出现的，而不是零散的点？")
    lines.append("  （提示：和 64 条激光线的分布方式有关）")
    lines.append("- 追问：这些缺失对后续用 Mamba 做**序列建模**意味着什么？")
    lines.append("  论文里的 SPP 模块是想缓解什么问题？")
    lines.append("")
    lines.append("**问题 3：填充率 77.43% 意味着什么？**")
    lines.append("")
    lines.append("- 先分清两个概念：**落进网格的点数** ≠ **被占据的格子数**（多个点会挤进同一格）。")
    lines.append("- 报告第 2 节给出：点数 = ?、格子数 = ?、比值 = ?。自己把这三个数抄下来并解释关系。")
    lines.append("- 追问：和问题 2 的四种成因对应起来，哪种成因占比最大？（从逐行表格看）")
    lines.append("")
    lines.append("**问题 4：同一格子有多个点时，为什么取最近的那个？**")
    lines.append("")
    lines.append("- 从「遮挡」和「最先被看到的表面」两个角度答。")
    lines.append("- 反证：如果取最远、或取平均，会分别造成什么错误？各举一个具体场景。")
    lines.append("")
    lines.append("**问题 5：为什么「墙还是车」从距离图里分不出来？**")
    lines.append("")
    lines.append("- 距离图每个格子里只有**一个数**：距离。物体的形状、材质、颜色全都在投影中丢失。")
    lines.append("- 能测的只有**几何线索**：横向跨度、纵向跨度、距离是否恒定。")
    lines.append("- 一堵墙和一辆卡车侧面，这三项测量值可以**完全一样**。")
    lines.append("- 追问：那要靠什么才能区分？论文里 MF-MOS 是怎么处理这个问题的？")
    lines.append("  （提示：它另开了一个吃距离图的**语义分支**）")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 绘图
def save_figs(points, range_image, valid, outdir, rows, cols):
    os.makedirs(outdir, exist_ok=True)

    # 距离图
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=140)
    im = ax.imshow(range_image, cmap="turbo", aspect="auto", interpolation="nearest")
    ax.set_title("Range View (distance image)  —  vertical: pitch, horizontal: yaw")
    ax.set_xlabel("yaw / azimuth  ->  %d columns (360 deg)" % cols)
    ax.set_ylabel("%d rows" % rows)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("range (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rv_range.png"))
    plt.close(fig)

    # 有效掩码
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=140)
    ax.imshow(valid, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title("Valid mask  —  white = a point fell into this cell (fill ratio %.2f%%)" % (valid.mean() * 100))
    ax.set_xlabel("yaw / azimuth")
    ax.set_ylabel("%d rows" % rows)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rv_valid.png"))
    plt.close(fig)

    # 俯视图对照
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.scatter(points[:, 0], points[:, 1], s=0.3, c=points[:, 2], cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title("Top view (x-y), color = z")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pointcloud_top.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_path", nargs="?", help="KITTI velodyne .bin 文件")
    ap.add_argument("--demo", action="store_true", help="用合成点云自测")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=900)
    ap.add_argument("--max-range", type=float, default=MAX_RANGE_M)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()

    if args.demo or not args.bin_path:
        print("[i] 使用合成点云自测（--demo）")
        points = make_demo_points()
        src = "synthetic demo cloud"
        is_synthetic = True
    else:
        if not os.path.isfile(args.bin_path):
            sys.exit("找不到文件: %s" % args.bin_path)
        points = load_kitti_bin(args.bin_path)
        src = os.path.basename(args.bin_path)
        is_synthetic = "synthetic" in src.lower()

    print("[1/3] 投影中 ...")
    ri, valid, stats = project_to_rv(points, args.rows, args.cols, max_range=args.max_range)

    print("[2/3] 绘图 ...")
    save_figs(points, ri, valid, args.outdir, args.rows, args.cols)

    print("[3/3] 生成报告 ...")
    import datetime

    md = []
    md.append("# RV 投影复现记录")
    md.append("")
    md.append(f"- 生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append(f"- 数据来源: `{src}`")
    md.append(f"- 数据性质: {'**合成数据（非真实 LiDAR 扫描，写日志时必须注明）**' if is_synthetic else '真实 KITTI 点云'}")
    md.append("- 图像尺寸: %d x %d（对应 OverlapMamba 论文公式 (5)）" % (args.rows, args.cols))
    md.append("- 运行命令: `python rv_projection.py %s`" % (args.bin_path or "--demo"))
    md.append("")
    md.append(analyze(points, ri, valid, args.rows, args.cols, args.max_range,
                      stats["kept_points"]))
    md.append("")
    md.append("## 5. 产出文件")
    md.append("")
    md.append("- `rv_range.png` — 距离图（主产物）")
    md.append("- `rv_valid.png` — 有效像素掩码（稀疏度可视化）")
    md.append("- `pointcloud_top.png` — 俯视图对照")
    md.append("")
    md.append("> 说明：本报告由 `rv_projection.py` 自动生成。")
    md.append("> 脚本编写过程中使用了 AI 辅助；参数来源、结果校验与问题排查见 `复现日志.md`。")
    md.append("")
    out_md = os.path.join(args.outdir, "report.md")
    text = thin_bold("\n".join(md) + "\n")
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print("")
    print("完成。产出目录: %s" % args.outdir)
    print("  - rv_range.png / rv_valid.png / pointcloud_top.png")
    print("  - report.md")
    print("")
    print("关键数字: 视场内点数 %d，填充率 %.2f%%" % (stats["kept_points"], stats["filled_cells"] * 100.0 / stats["cells"]))


if __name__ == "__main__":
    main()
