# -*- coding: utf-8 -*-
"""
Schneider CCT 四方法对照测试程序
================================

用途：
对同一幅图、同一批粗候选靶标，比较四种边缘方法：

M1  Sobel
M2  Original-DF3SOp-like pixel-adaptive beta (alpha=1)
M3  Scalar-beta DF3SOp
M4  Spatial-beta(x,y) DF3SOp

重点回答：
1) 标量 beta 是否在 normalize + Otsu 后基本失效；
2) 空间 beta(x,y) 是否真正改变了归一化梯度和最终二值边缘；
3) 空间 beta 是否提高理论 CCT 边缘支持，同时不过度增加白色 gap 平坦区伪边缘；
4) 四种边缘结果对椭圆精化和 12-bit 解码是否产生差异。

使用方法：
1) 将本文件与 schneider_cct_spatial_beta_full.py 放在同一目录；
2) 修改下面三个路径；
3) 直接运行本文件。

输出：
- comparison_summary.csv      每个目标、每种方法的详细指标
- target_summary.csv          每个目标横向对比
- overall_summary.csv         各方法总体统计
- target_xx/                  每种方法梯度、边缘、差分图、参数
"""

import os
import math
import csv
import importlib.util
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np


# ============================================================================
# 1. 用户配置
# ============================================================================

# 当前完整算法程序
BASE_PROGRAM = r"schneider_cct_spatial_beta_full.py"

# 测试图片
IMAGE_PATH = r"D:\code\python_code\work_photos\test_0_100\20260818_160006_788655_0.bmp"

# 输出目录
OUTPUT_DIR = r"D:\code\python_code\work_photos\test3"

# 是否使用当前完整程序中的高斯预处理参数
USE_BASE_GAUSSIAN_SETTING = True

EPS = 1e-8


# ============================================================================
# 2. 动态加载当前完整算法
# ============================================================================

def load_base_module(path: str):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到当前完整程序：{path}\n"
            "请修改 BASE_PROGRAM，或把两个 .py 文件放在同一目录。"
        )

    spec = importlib.util.spec_from_file_location("cct_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# 3. 通用边缘后处理
# ============================================================================

def normalize_and_otsu(mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    mag_u8 = cv2.normalize(
        mag, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    thr, edge = cv2.threshold(
        mag_u8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )
    return mag_u8, edge, float(thr)


def edge_iou(edge_a: np.ndarray, edge_b: np.ndarray, mask=None) -> float:
    a = edge_a > 0
    b = edge_b > 0
    if mask is not None:
        a = a & mask
        b = b & mask

    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    inter = np.count_nonzero(a & b)
    return float(inter / union)


def binary_diff_rate(edge_a: np.ndarray, edge_b: np.ndarray, mask=None) -> float:
    diff = (edge_a > 0) != (edge_b > 0)
    if mask is not None:
        vals = diff[mask]
    else:
        vals = diff.ravel()

    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


# ============================================================================
# 4. 四种方法
# ============================================================================

def sobel_method(gray: np.ndarray) -> Dict[str, np.ndarray]:
    src = gray.astype(np.float32)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8, edge, thr = normalize_and_otsu(mag)

    return {
        "mag": mag,
        "mag_u8": mag_u8,
        "edge": edge,
        "otsu_threshold": thr,
    }


def original_df3sop_like_method(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """
    对照原 DF3SOp 边缘模式思想：
    alpha=1，beta 由 Sobel 基准梯度逐像素自适应。

    与当前主程序粗检测保持同一思路：
        G_adapt = G0 / (2*beta_map)
    """
    src = gray.astype(np.float32)

    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    G0 = cv2.magnitude(gx, gy)

    gmin = float(np.min(G0))
    gmax = float(np.max(G0))
    gmean = float(np.mean(G0))

    F1 = 0.5 * (gmin + gmean)
    F2 = 0.5 * (gmax + gmean)

    beta_map = np.full_like(G0, 0.5, dtype=np.float32)

    mid = (G0 > F1) & (G0 <= F2)
    beta_map[mid] = 0.1 + 0.7 * (
        (G0[mid] - gmin) /
        (gmax - gmin + EPS)
    )
    beta_map[G0 > F2] = 0.1

    mag = G0 / (2.0 * beta_map + EPS)
    mag_u8, edge, thr = normalize_and_otsu(mag)

    return {
        "mag": mag,
        "mag_u8": mag_u8,
        "edge": edge,
        "otsu_threshold": thr,
        "beta_map": beta_map,
    }


def scalar_beta_method(
    base,
    gray: np.ndarray,
    alpha: float,
    beta_scalar: float
) -> Dict[str, np.ndarray]:
    """
    标量 beta 对照。

    注意：
    beta_scalar 在整个目标/图像上是一个常数，因此理论上：
        M(alpha,beta) = beta^(-alpha) * A(alpha)
    后续 min-max normalize 会消除这一全局尺度。
    """
    mx_raw, my_raw, mx, my = base.build_df3sop_kernels(
        alpha,
        beta_scalar,
        base.ZERO_SUM_CORRECTION
    )

    src = gray.astype(np.float32)
    gx = cv2.filter2D(
        src, cv2.CV_32F, mx,
        borderType=cv2.BORDER_REFLECT101
    )
    gy = cv2.filter2D(
        src, cv2.CV_32F, my,
        borderType=cv2.BORDER_REFLECT101
    )
    mag = cv2.magnitude(gx, gy)
    mag_u8, edge, thr = normalize_and_otsu(mag)

    return {
        "mag": mag,
        "mag_u8": mag_u8,
        "edge": edge,
        "otsu_threshold": thr,
        "mx": mx,
        "my": my,
    }


def alpha_only_reference_method(
    base,
    gray: np.ndarray,
    alpha: float
) -> Dict[str, np.ndarray]:
    """
    beta=1 的 alpha-only 参考，用来直接验证标量 beta 是否被 normalize 抵消。
    """
    return scalar_beta_method(base, gray, alpha, 1.0)


def spatial_beta_method(
    base,
    gray: np.ndarray,
    alpha: float,
    E: float,
    ellipse: Tuple
) -> Dict[str, np.ndarray]:

    (
        mag, mag_u8, edge,
        mx_raw, my_raw, mx, my,
        beta_map, geometry_weight, beta_edge
    ) = base.df3sop_edge_spatial(
        gray,
        alpha,
        E,
        ellipse,
        base.ZERO_SUM_CORRECTION
    )

    # 主程序内部已执行 normalize + Otsu
    # 为获取实际 Otsu 阈值，再基于相同 mag_u8 计算一次阈值值（不改变 edge）
    thr, _ = cv2.threshold(
        mag_u8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return {
        "mag": mag,
        "mag_u8": mag_u8,
        "edge": edge,
        "otsu_threshold": float(thr),
        "beta_map": beta_map,
        "geometry_weight": geometry_weight,
        "beta_edge": float(beta_edge),
    }


# ============================================================================
# 5. CCT 结构区域与评价指标
# ============================================================================

def make_masks(base, gray_shape, ellipse: Tuple):
    rho = base.ellipse_rho_map(gray_shape, ellipse)

    expected_edge = np.zeros(gray_shape, dtype=bool)
    for r0 in (1.0, base.LAMBDA_IN, base.LAMBDA_OUT):
        expected_edge |= (
            np.abs(rho - float(r0)) <=
            max(base.EDGE_BAND_HALF_WIDTH, 0.10)
        )

    # 白色 gap 中央区域，刻意远离两侧边界
    flat_lo = max(1.20, base.NOISE_RHO_MIN)
    flat_hi = max(
        flat_lo + 0.05,
        base.LAMBDA_IN - max(
            getattr(base, "NOISE_MARGIN_TO_CODE", 0.25),
            0.25
        )
    )
    flat_gap = (rho >= flat_lo) & (rho <= flat_hi)

    target_roi = rho <= (base.LAMBDA_OUT + 0.25)

    code_annulus = (
        (rho >= base.LAMBDA_IN + 0.05) &
        (rho <= base.LAMBDA_OUT - 0.05)
    )

    return {
        "rho": rho,
        "expected_edge": expected_edge,
        "flat_gap": flat_gap,
        "target_roi": target_roi,
        "code_annulus": code_annulus,
    }


def safe_percentile(arr, q=75.0):
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def safe_mean(arr):
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def evaluate_method(
    base,
    gray: np.ndarray,
    coarse_ellipse: Tuple,
    method_name: str,
    out: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    spatial_edge_reference: np.ndarray,
    alpha_only_reference: Dict[str, np.ndarray] = None,
):
    mag = out["mag"]
    mag_u8 = out["mag_u8"]
    edge = out["edge"]

    m_edge = masks["expected_edge"]
    m_flat = masks["flat_gap"]
    m_roi = masks["target_roi"]

    # 连续梯度结构指标
    raw_edge_p75 = safe_percentile(mag[m_edge], 75.0)
    raw_flat_mean = safe_mean(mag[m_flat])
    raw_edge_flat_ratio = raw_edge_p75 / (raw_flat_mean + EPS)

    norm_edge_p75 = safe_percentile(
        mag_u8[m_edge].astype(np.float32), 75.0
    )
    norm_flat_mean = safe_mean(
        mag_u8[m_flat].astype(np.float32)
    )
    norm_edge_flat_ratio = (
        norm_edge_p75 /
        (norm_flat_mean + EPS)
    )

    # 二值边缘结构指标
    expected_edge_density = safe_mean(
        (edge[m_edge] > 0).astype(np.float32)
    )
    flat_edge_density = safe_mean(
        (edge[m_flat] > 0).astype(np.float32)
    )
    roi_edge_density = safe_mean(
        (edge[m_roi] > 0).astype(np.float32)
    )

    edge_density_difference = (
        expected_edge_density -
        flat_edge_density
    )

    # 相对于 spatial beta 的二值结果差异
    iou_vs_spatial = edge_iou(
        edge,
        spatial_edge_reference,
        m_roi
    )
    diff_vs_spatial = binary_diff_rate(
        edge,
        spatial_edge_reference,
        m_roi
    )

    # 椭圆精化 + 解码
    refined = base.refine_center_ellipse(
        edge,
        coarse_ellipse
    )

    bits, canonical, reversed_canonical, phase, conf = (
        base.decode_12bit(
            gray,
            refined,
            base.N_BITS
        )
    )

    schneider_id = base.CODEBOOK.get(canonical, -1)
    if schneider_id < 0:
        schneider_id = base.CODEBOOK.get(
            reversed_canonical,
            -1
        )

    (cx0, cy0), _, _ = coarse_ellipse
    (cx1, cy1), _, _ = refined
    refine_shift_px = math.hypot(
        float(cx1) - float(cx0),
        float(cy1) - float(cy0)
    )

    row = {
        "method": method_name,
        "otsu_threshold": float(out["otsu_threshold"]),

        "raw_expected_edge_p75": raw_edge_p75,
        "raw_flat_gap_mean": raw_flat_mean,
        "raw_edge_flat_ratio": raw_edge_flat_ratio,

        "norm_expected_edge_p75": norm_edge_p75,
        "norm_flat_gap_mean": norm_flat_mean,
        "norm_edge_flat_ratio": norm_edge_flat_ratio,

        "expected_edge_density": expected_edge_density,
        "flat_gap_edge_density": flat_edge_density,
        "edge_density_difference": edge_density_difference,
        "target_roi_edge_density": roi_edge_density,

        "IoU_vs_spatial": iou_vs_spatial,
        "diff_rate_vs_spatial": diff_vs_spatial,

        "refined_center_x": float(cx1),
        "refined_center_y": float(cy1),
        "refine_shift_px": refine_shift_px,

        "bits": bits,
        "canonical_code": int(canonical),
        "reversed_canonical_code": int(reversed_canonical),
        "schneider_id": int(schneider_id),
        "phase_deg": float(phase),
        "decode_confidence": float(conf),
    }

    # scalar beta 与 alpha-only 的专门等价性测试
    if (
        method_name == "M3_scalar_beta" and
        alpha_only_reference is not None
    ):
        ref_u8 = alpha_only_reference["mag_u8"]
        ref_edge = alpha_only_reference["edge"]

        mae = float(np.mean(
            np.abs(
                mag_u8.astype(np.float32) -
                ref_u8.astype(np.float32)
            )
        ))

        row["scalar_vs_alpha_only_norm_MAE"] = mae
        row["scalar_vs_alpha_only_edge_diff_rate"] = (
            binary_diff_rate(
                edge,
                ref_edge,
                m_roi
            )
        )
        row["scalar_vs_alpha_only_edge_IoU"] = (
            edge_iou(
                edge,
                ref_edge,
                m_roi
            )
        )
    else:
        row["scalar_vs_alpha_only_norm_MAE"] = float("nan")
        row["scalar_vs_alpha_only_edge_diff_rate"] = float("nan")
        row["scalar_vs_alpha_only_edge_IoU"] = float("nan")

    return row, refined


# ============================================================================
# 6. 保存单目标可视化
# ============================================================================

def save_method_images(
    target_dir: str,
    method_name: str,
    out: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    spatial_edge: np.ndarray,
):
    method_dir = os.path.join(
        target_dir,
        method_name
    )
    os.makedirs(method_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(method_dir, "gradient.png"),
        out["mag_u8"]
    )
    cv2.imwrite(
        os.path.join(method_dir, "edge.png"),
        out["edge"]
    )

    diff = (
        (out["edge"] > 0) !=
        (spatial_edge > 0)
    )
    diff = (
        diff &
        masks["target_roi"]
    )

    cv2.imwrite(
        os.path.join(
            method_dir,
            "edge_difference_vs_spatial.png"
        ),
        np.where(diff, 255, 0).astype(np.uint8)
    )

    if "beta_map" in out:
        beta_map = out["beta_map"]
        beta_u8 = cv2.normalize(
            beta_map, None, 0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(method_dir, "beta_map.png"),
            beta_u8
        )

    if "geometry_weight" in out:
        W = out["geometry_weight"]
        W_u8 = cv2.normalize(
            W, None, 0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(method_dir, "geometry_weight.png"),
            W_u8
        )


# ============================================================================
# 7. CSV 工具
# ============================================================================

def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return

    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(
        path, "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# 8. 主程序
# ============================================================================

def main():
    base = load_base_module(BASE_PROGRAM)

    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(
            f"找不到测试图片：{IMAGE_PATH}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    image = cv2.imread(
        IMAGE_PATH,
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise RuntimeError(
            f"OpenCV 无法读取：{IMAGE_PATH}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if (
        USE_BASE_GAUSSIAN_SETTING and
        base.USE_GAUSSIAN_BLUR
    ):
        k = int(base.GAUSSIAN_KSIZE)
        if k % 2 == 0:
            k += 1
        gray_work = cv2.GaussianBlur(
            gray,
            (k, k),
            base.GAUSSIAN_SIGMA
        )
    else:
        gray_work = gray.copy()

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "input_gray.png"),
        gray
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "input_gray_work.png"),
        gray_work
    )

    # ---------------------------------------------------------------
    # 统一粗检测：所有方法共享同一批候选
    # ---------------------------------------------------------------
    G0_norm, coarse_grad, coarse_edge = (
        base.coarse_adaptive_edge(
            gray_work
        )
    )

    candidates = base.detect_coarse_candidates(
        gray_work,
        G0_norm,
        coarse_edge
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "coarse_candidates.png"
        ),
        base.draw_coarse_candidates(
            image,
            candidates
        )
    )

    print("=" * 88)
    print("Schneider CCT 四方法对照测试")
    print("=" * 88)
    print(f"IMAGE_PATH : {IMAGE_PATH}")
    print(f"候选数     : {len(candidates)}")
    print(
        f"template   : lambda_in={base.LAMBDA_IN:.4f}, "
        f"lambda_out={base.LAMBDA_OUT:.4f}"
    )

    if not candidates:
        print("未检测到粗候选，测试终止。")
        return

    detailed_rows = []
    target_rows = []

    for idx, cand in enumerate(
        candidates,
        start=1
    ):
        coarse = cand.ellipse

        # C/E/alpha/beta_edge 完全共享
        p = base.estimate_alpha_beta(
            gray_work,
            G0_norm,
            coarse
        )

        alpha = p["alpha"]
        beta_scalar = p["beta"]

        # 四方法
        m1 = sobel_method(gray_work)
        m2 = original_df3sop_like_method(
            gray_work
        )
        m3 = scalar_beta_method(
            base,
            gray_work,
            alpha,
            beta_scalar
        )
        alpha_only = alpha_only_reference_method(
            base,
            gray_work,
            alpha
        )
        m4 = spatial_beta_method(
            base,
            gray_work,
            alpha,
            p["E"],
            coarse
        )

        methods = {
            "M1_Sobel": m1,
            "M2_original_pixel_beta": m2,
            "M3_scalar_beta": m3,
            "M4_spatial_beta": m4,
        }

        masks = make_masks(
            base,
            gray_work.shape,
            coarse
        )

        target_dir = os.path.join(
            OUTPUT_DIR,
            f"target_{idx:02d}"
        )
        os.makedirs(
            target_dir,
            exist_ok=True
        )

        current_rows = {}

        for name, out in methods.items():
            row, refined = evaluate_method(
                base,
                gray_work,
                coarse,
                name,
                out,
                masks,
                m4["edge"],
                alpha_only
            )

            row.update({
                "target": idx,
                "C": float(p["C"]),
                "E": float(p["E"]),
                "alpha": float(alpha),
                "beta_scalar_or_edge": float(beta_scalar),
                "beta_spatial_min": float(np.min(m4["beta_map"])),
                "beta_spatial_max": float(np.max(m4["beta_map"])),
            })

            detailed_rows.append(row)
            current_rows[name] = row

            save_method_images(
                target_dir,
                name,
                out,
                masks,
                m4["edge"]
            )

        # 每个目标一行横向摘要
        tr = {
            "target": idx,
            "C": float(p["C"]),
            "E": float(p["E"]),
            "alpha": float(alpha),
            "beta_edge": float(beta_scalar),
            "beta_spatial_min": float(np.min(m4["beta_map"])),
            "beta_spatial_max": float(np.max(m4["beta_map"])),

            "Sobel_edge_density": current_rows["M1_Sobel"]["expected_edge_density"],
            "Original_edge_density": current_rows["M2_original_pixel_beta"]["expected_edge_density"],
            "Scalar_edge_density": current_rows["M3_scalar_beta"]["expected_edge_density"],
            "Spatial_edge_density": current_rows["M4_spatial_beta"]["expected_edge_density"],

            "Sobel_flat_density": current_rows["M1_Sobel"]["flat_gap_edge_density"],
            "Original_flat_density": current_rows["M2_original_pixel_beta"]["flat_gap_edge_density"],
            "Scalar_flat_density": current_rows["M3_scalar_beta"]["flat_gap_edge_density"],
            "Spatial_flat_density": current_rows["M4_spatial_beta"]["flat_gap_edge_density"],

            "Scalar_vs_alphaOnly_norm_MAE":
                current_rows["M3_scalar_beta"]["scalar_vs_alpha_only_norm_MAE"],
            "Scalar_vs_alphaOnly_edge_diff_rate":
                current_rows["M3_scalar_beta"]["scalar_vs_alpha_only_edge_diff_rate"],
            "Scalar_vs_alphaOnly_edge_IoU":
                current_rows["M3_scalar_beta"]["scalar_vs_alpha_only_edge_IoU"],

            "Sobel_code": current_rows["M1_Sobel"]["canonical_code"],
            "Original_code": current_rows["M2_original_pixel_beta"]["canonical_code"],
            "Scalar_code": current_rows["M3_scalar_beta"]["canonical_code"],
            "Spatial_code": current_rows["M4_spatial_beta"]["canonical_code"],

            "Sobel_ID": current_rows["M1_Sobel"]["schneider_id"],
            "Original_ID": current_rows["M2_original_pixel_beta"]["schneider_id"],
            "Scalar_ID": current_rows["M3_scalar_beta"]["schneider_id"],
            "Spatial_ID": current_rows["M4_spatial_beta"]["schneider_id"],

            "Sobel_conf": current_rows["M1_Sobel"]["decode_confidence"],
            "Original_conf": current_rows["M2_original_pixel_beta"]["decode_confidence"],
            "Scalar_conf": current_rows["M3_scalar_beta"]["decode_confidence"],
            "Spatial_conf": current_rows["M4_spatial_beta"]["decode_confidence"],
        }

        target_rows.append(tr)

        print("-" * 88)
        print(
            f"Target {idx}: "
            f"C={p['C']:.4f}, E={p['E']:.4f}, "
            f"alpha={alpha:.4f}, beta_edge={beta_scalar:.4f}"
        )
        print(
            "  Expected-edge density | "
            f"Sobel={tr['Sobel_edge_density']:.4f}, "
            f"Original={tr['Original_edge_density']:.4f}, "
            f"Scalar={tr['Scalar_edge_density']:.4f}, "
            f"Spatial={tr['Spatial_edge_density']:.4f}"
        )
        print(
            "  Flat-gap edge density | "
            f"Sobel={tr['Sobel_flat_density']:.4f}, "
            f"Original={tr['Original_flat_density']:.4f}, "
            f"Scalar={tr['Scalar_flat_density']:.4f}, "
            f"Spatial={tr['Spatial_flat_density']:.4f}"
        )
        print(
            "  Scalar beta equivalence | "
            f"norm MAE={tr['Scalar_vs_alphaOnly_norm_MAE']:.8f}, "
            f"edge diff={100*tr['Scalar_vs_alphaOnly_edge_diff_rate']:.8f}%, "
            f"IoU={tr['Scalar_vs_alphaOnly_edge_IoU']:.10f}"
        )
        print(
            "  Codes | "
            f"Sobel={tr['Sobel_code']}, "
            f"Original={tr['Original_code']}, "
            f"Scalar={tr['Scalar_code']}, "
            f"Spatial={tr['Spatial_code']}"
        )

    # ---------------------------------------------------------------
    # 保存详细结果
    # ---------------------------------------------------------------
    write_csv(
        os.path.join(
            OUTPUT_DIR,
            "comparison_summary.csv"
        ),
        detailed_rows
    )

    write_csv(
        os.path.join(
            OUTPUT_DIR,
            "target_summary.csv"
        ),
        target_rows
    )

    # ---------------------------------------------------------------
    # 总体统计
    # ---------------------------------------------------------------
    overall_rows = []

    method_names = [
        "M1_Sobel",
        "M2_original_pixel_beta",
        "M3_scalar_beta",
        "M4_spatial_beta",
    ]

    for name in method_names:
        rr = [
            r for r in detailed_rows
            if r["method"] == name
        ]

        def mean_key(k):
            vals = np.asarray(
                [float(r[k]) for r in rr],
                dtype=np.float64
            )
            vals = vals[np.isfinite(vals)]
            return (
                float(np.mean(vals))
                if vals.size
                else float("nan")
            )

        known_count = sum(
            int(r["schneider_id"]) >= 0
            for r in rr
        )

        overall_rows.append({
            "method": name,
            "n_targets": len(rr),

            "mean_raw_edge_flat_ratio":
                mean_key("raw_edge_flat_ratio"),

            "mean_norm_edge_flat_ratio":
                mean_key("norm_edge_flat_ratio"),

            "mean_expected_edge_density":
                mean_key("expected_edge_density"),

            "mean_flat_gap_edge_density":
                mean_key("flat_gap_edge_density"),

            "mean_edge_density_difference":
                mean_key("edge_density_difference"),

            "mean_target_roi_edge_density":
                mean_key("target_roi_edge_density"),

            "mean_IoU_vs_spatial":
                mean_key("IoU_vs_spatial"),

            "mean_refine_shift_px":
                mean_key("refine_shift_px"),

            "mean_decode_confidence":
                mean_key("decode_confidence"),

            "known_codebook_matches":
                known_count,
        })

    write_csv(
        os.path.join(
            OUTPUT_DIR,
            "overall_summary.csv"
        ),
        overall_rows
    )

    # ---------------------------------------------------------------
    # 自动结论
    # ---------------------------------------------------------------
    scalar_rows = [
        r for r in detailed_rows
        if r["method"] == "M3_scalar_beta"
    ]

    scalar_max_mae = max(
        float(r["scalar_vs_alpha_only_norm_MAE"])
        for r in scalar_rows
    )
    scalar_max_diff = max(
        float(r["scalar_vs_alpha_only_edge_diff_rate"])
        for r in scalar_rows
    )
    scalar_min_iou = min(
        float(r["scalar_vs_alpha_only_edge_IoU"])
        for r in scalar_rows
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "automatic_conclusion.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write("Schneider CCT four-method comparison\n")
        f.write("=" * 72 + "\n\n")

        f.write("1. Scalar-beta equivalence test\n")
        f.write(
            f"max normalized-gradient MAE = {scalar_max_mae:.10f}\n"
        )
        f.write(
            f"max binary-edge diff rate   = {scalar_max_diff:.10f}\n"
        )
        f.write(
            f"min binary-edge IoU         = {scalar_min_iou:.10f}\n"
        )

        if (
            scalar_max_mae <= 0.05 and
            scalar_max_diff <= 1e-6 and
            scalar_min_iou >= 0.999999
        ):
            f.write(
                "Conclusion: scalar beta is effectively cancelled "
                "by normalize + Otsu in the current pipeline.\n\n"
            )
        else:
            f.write(
                "Conclusion: scalar beta produces measurable differences "
                "in this dataset; inspect target-level results.\n\n"
            )

        f.write(
            "2. For method comparison, focus on:\n"
            "   - expected_edge_density: higher is generally better;\n"
            "   - flat_gap_edge_density: lower is generally better;\n"
            "   - edge_density_difference: higher is generally better;\n"
            "   - decode_confidence and recovered code consistency;\n"
            "   - use true target IDs later to compute identification rate.\n"
        )

    print("=" * 88)
    print("测试完成。")
    print(
        f"详细结果：{os.path.join(OUTPUT_DIR, 'comparison_summary.csv')}"
    )
    print(
        f"逐目标摘要：{os.path.join(OUTPUT_DIR, 'target_summary.csv')}"
    )
    print(
        f"总体摘要：{os.path.join(OUTPUT_DIR, 'overall_summary.csv')}"
    )
    print(
        f"自动结论：{os.path.join(OUTPUT_DIR, 'automatic_conclusion.txt')}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
