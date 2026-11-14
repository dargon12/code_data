# -*- coding: utf-8 -*-
"""
Schneider 圆形编码靶标（CCT）检测示例
=====================================

核心思路：
1) 不遍历 alpha / beta；
2) 先用一个“粗检测”算子寻找中央圆/椭圆候选；
3) 以中央圆建立 Schneider CCT 的归一化椭圆坐标 rho；
4) 根据靶标局部光度可分性 C 和边缘可靠度 E，计算 alpha 与空间 beta(x,y)：

       C = |mu_c - mu_g| / (mu_c + mu_g + eps)
       E = G_T / (G_T + G_N + eps)
       alpha = 0.1 + 0.9*C

       beta_edge = beta_min + (beta_max-beta_min)*alpha*E
       W(rho) = max_i exp(-(rho-r_i)^2/(2*sigma_rho^2))
       beta(x,y) = beta_max - W(rho)*(beta_max-beta_edge)

   其中：
   - mu_c：排除中央白点后的黑色中央圆盘平均灰度；
   - mu_g：中央圆与编码环之间白色分隔环平均灰度；
   - r_i = {1, lambda_in, lambda_out}；
   - G_T：上述理论结构边缘处的梯度响应；
   - G_N：白色分隔环平坦区域梯度响应。

5) 用 alpha 构造零和 DF3SOp 基础核，再以 beta(x,y)^(-alpha) 对局部梯度响应进行空间调制；
6) 利用实际模板编码环中径 rho_d=(lambda_in+lambda_out)/2 进行 12 bit 采样；\n7) 为每个目标输出 12 个采样扇区可视化图、编码环展开图和逐码元 CSV。

重要说明：
- alpha 与空间 beta(x,y) 的映射是“面向 Schneider CCT 的改进设计”，不是原 DF3SOp 论文公式。
- 原 DF3SOp 在边缘检测时通常令 alpha=1 以保持零和性质。
  本程序允许 alpha!=1，因此对 alpha 核增加零均值（zero-sum）修正。
- beta 不再作为整个卷积核的单一全局比例因子，而是在预期 CCT 结构边缘附近形成空间参数场，
  避免其作用被后续 min-max 归一化和 Otsu 阈值基本抵消。
- 粗检测阶段的 Sobel 只用于估计原论文 alpha=1 模式下的初始梯度/自适应 beta，
  最终精检测使用显式构造的 DF3SOp 核，不使用 cv2.Sobel 作为最终算子。

依赖：
    pip install opencv-python numpy

使用方法：
    只修改下面“用户配置区”的 IMAGE_PATH，然后直接运行本文件。
"""

import os
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


# ============================================================================
# 1. 用户配置区 —— 只需要在这里修改输入图像
# ============================================================================

# !!! 改成你的图片路径 !!!
IMAGE_PATH = r"D:\code\python_code\work_photos\test_0_100\20260818_160006_788655_0.bmp"

# 输出目录；留空则自动创建在输入图片同目录下
OUTPUT_DIR = r"D:\code\python_code\work_photos\test2"

# 是否显示 OpenCV 窗口（服务器/无GUI环境建议 False）
SHOW_WINDOWS = True

# Schneider CCT 位数
N_BITS = 12

# ============================================================================
# 编码码表
# ============================================================================
# 下面 4 个 canonical_code 是根据你当前提供的 4 个示例图案建立的演示码表：
#   图案1 -> 503
#   图案2 -> 1005
#   图案3 -> 763
#   图案4 -> 1003
#
# 程序会先采样 12 bit，再做循环移位归一化得到 canonical_code，
# 最后通过这个字典映射为你希望看到的编码 ID。
#
# 如果以后使用正式 Schneider/Keith 编码表，只需要替换这个字典，
# 前面的 alpha/beta 估计、DF3SOp、圆检测和 12 bit 采样均不用修改。
CODEBOOK = {
    503: 1,
    1005: 2,
    763: 3,
    1003: 4,
}

# 是否对 alpha!=1 时的 DF3SOp 做零和修正
ZERO_SUM_CORRECTION = True

# 是否对输入灰度图做轻度高斯去噪
USE_GAUSSIAN_BLUR = True
GAUSSIAN_KSIZE = 5
GAUSSIAN_SIGMA = 0.8

# ---------------------------
# 粗中央圆候选筛选参数
# ---------------------------
MIN_CONTOUR_POINTS = 30
MIN_ELLIPSE_DIAMETER_PX = 12.0
MAX_ELLIPSE_DIAMETER_RATIO = 0.30   # 相对 min(image_h, image_w)
MIN_AXIS_RATIO = 0.30                # 允许较大倾角，圆投影成椭圆
MIN_CIRCULARITY = 0.30
# 椭圆拟合残差：真实中央圆投影应与椭圆高度一致；编码扇区通常残差更大
MAX_ELLIPSE_FIT_RESIDUAL = 0.045
MIN_PHOTOMETRIC_CONTRAST = 0.18
MIN_CODE_RING_STD = 18.0
MIN_CANDIDATE_SCORE = 0.48
MAX_TARGETS = 50

# 粗候选去重：两个候选中心距离小于 min(avg_radius)*该系数时视为同一中央圆
DEDUP_CENTER_FACTOR = 0.80

# ---------------------------
# 实际 Schneider CCT 模板几何
# ---------------------------
# 根据当前 ID85 干净正视模板的径向统计，测试得到约：
#   lambda_in  = 2.28
#   lambda_out = 3.56
# 若后续更换靶标版式，请重新标定这两个值。
LAMBDA_IN = 2.28
LAMBDA_OUT = 3.56

# ---------------------------
# alpha / 空间 beta(x,y) 参数
# ---------------------------
ALPHA_MIN = 0.10
ALPHA_MAX = 1.00
BETA_MIN = 0.10
BETA_MAX = 0.80

# 中央黑圆内存在小白点，因此排除 rho<RHO_CENTER_HOLE
RHO_CENTER_HOLE = 0.15
RHO_CENTER_MAX = 0.75

# 白色分隔环采样区
RHO_SEPARATOR_MIN = 1.15
RHO_SEPARATOR_MARGIN_TO_CODE = 0.18

# 结构边缘半宽与平坦区
EDGE_BAND_HALF_WIDTH = 0.10
NOISE_RHO_MIN = 1.25
NOISE_MARGIN_TO_CODE = 0.25

# 空间 beta 几何高斯权重宽度
SPATIAL_BETA_SIGMA_RHO = 0.12

# G_T 使用环带梯度的分位数，适应分段编码环
EDGE_RESPONSE_PERCENTILE = 75.0

# ---------------------------
# 精检测和编码参数
# ---------------------------
ROI_SCALE = LAMBDA_OUT + 0.35
REFINE_CENTER_TOL_FACTOR = 0.50
REFINE_SIZE_MIN_FACTOR = 0.65
REFINE_SIZE_MAX_FACTOR = 1.45

CODE_RHO_CENTER = 0.5 * (LAMBDA_IN + LAMBDA_OUT)
CODE_RHO_HALF_WIDTH = 0.22
CODE_ANGULAR_FILL = 0.62
PHASE_STEP_DEG = 1.0

EPS = 1e-8


# ============================================================================
# 2. 数据结构
# ============================================================================

@dataclass
class Candidate:
    ellipse: Tuple
    contour_area: float
    circularity: float
    axis_ratio: float
    ellipse_fit_residual: float
    contrast_C: float
    code_std: float
    edge_support: float
    score: float


@dataclass
class TargetResult:
    index: int
    coarse_ellipse: Tuple
    refined_ellipse: Tuple
    C: float
    E: float
    alpha: float
    beta: float
    beta_map_min: float
    beta_map_max: float
    mu_center: float
    mu_separator: float
    G_target: float
    G_noise: float
    bits: str
    canonical_code: int
    reversed_canonical_code: int
    phase_deg: float
    decode_confidence: float
    schneider_id: int


# ============================================================================
# 3. 基础工具函数
# ============================================================================

def ensure_output_dir(image_path: str, output_dir: str) -> str:
    if output_dir.strip():
        out = output_dir
    else:
        parent = os.path.dirname(os.path.abspath(image_path))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out = os.path.join(parent, stem + "_cct_formula_ab_result")
    os.makedirs(out, exist_ok=True)
    return out


def save_normalized_image(path: str, arr: np.ndarray) -> None:
    """把浮点梯度等归一化为 8 bit 保存。"""
    img = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(path, img.astype(np.uint8))


def ellipse_axes(ellipse: Tuple) -> Tuple[float, float]:
    (_, _), (A, B), _ = ellipse
    return max(A / 2.0, EPS), max(B / 2.0, EPS)


def ellipse_mean_radius(ellipse: Tuple) -> float:
    a, b = ellipse_axes(ellipse)
    return 0.5 * (a + b)


def ellipse_rho_map(shape: Tuple[int, int], ellipse: Tuple) -> np.ndarray:
    """
    建立相对于中央椭圆的归一化椭圆半径 rho。

    rho = 1：中央黑色定位圆外缘
    rho = LAMBDA_IN：编码带内缘
    rho = LAMBDA_OUT：编码带外缘
    """
    h, w = shape[:2]
    (cx, cy), (A, B), theta_deg = ellipse

    yy, xx = np.indices((h, w), dtype=np.float32)
    dx = xx - float(cx)
    dy = yy - float(cy)

    theta = math.radians(float(theta_deg))
    ct = math.cos(theta)
    st = math.sin(theta)

    # 旋转到椭圆自身坐标系
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy

    a = max(float(A) / 2.0, EPS)
    b = max(float(B) / 2.0, EPS)

    rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return rho


def ellipse_normalized_angle_map(shape: Tuple[int, int], ellipse: Tuple) -> np.ndarray:
    """返回椭圆归一化坐标中的角度 [0, 360)。"""
    h, w = shape[:2]
    (cx, cy), (A, B), theta_deg = ellipse

    yy, xx = np.indices((h, w), dtype=np.float32)
    dx = xx - float(cx)
    dy = yy - float(cy)

    theta = math.radians(float(theta_deg))
    ct = math.cos(theta)
    st = math.sin(theta)

    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy

    a = max(float(A) / 2.0, EPS)
    b = max(float(B) / 2.0, EPS)

    xn = xr / a
    yn = yr / b
    ang = np.degrees(np.arctan2(yn, xn))
    ang = np.mod(ang, 360.0)
    return ang.astype(np.float32)


def circularity_of_contour(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    per = float(cv2.arcLength(contour, True))
    if per <= EPS:
        return 0.0
    return float(4.0 * math.pi * area / (per * per))



def ellipse_fit_residual(contour: np.ndarray, ellipse: Tuple) -> float:
    """
    计算轮廓点相对于拟合椭圆 rho=1 的平均绝对偏差。
    对真正的圆/椭圆轮廓，该值通常很小；对扇形编码块等非椭圆轮廓明显更大。
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    (cx, cy), (A, B), theta_deg = ellipse
    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    theta = math.radians(float(theta_deg))
    ct = math.cos(theta)
    st = math.sin(theta)
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    a = max(float(A) / 2.0, EPS)
    b = max(float(B) / 2.0, EPS)
    rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return float(np.mean(np.abs(rho - 1.0)))


def valid_outer_extent(ellipse: Tuple, shape: Tuple[int, int], scale: float = 3.15) -> bool:
    """
    粗略判断 Schneider 外编码环是否大部分仍在图像范围内。
    为防止图像边缘的局部弧段被误判为中央圆。
    """
    h, w = shape[:2]
    (cx, cy), (A, B), _ = ellipse
    r = scale * max(A, B) / 2.0
    # 允许约 15% 外伸，适应靠边靶标
    margin = 0.15 * r
    return (
        cx + r >= -margin and cx - r <= w - 1 + margin and
        cy + r >= -margin and cy - r <= h - 1 + margin
    )


# ============================================================================
# 4. 粗检测：原论文 alpha=1 边缘模式的近似实现
# ============================================================================

def coarse_adaptive_edge(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    粗检测只负责找到中央圆候选，不是最终边缘结果。

    原 DF3SOp 论文的边缘检测阶段固定 alpha=1，并让 beta 根据梯度自适应。
    当 alpha=1 时，3x3 DF3SOp 与 Sobel 型梯度只差由 beta 控制的比例。
    因此这里先用 Sobel 得到基准梯度 G0，再按论文 beta 分区规则进行像素级缩放。

    输出：
        G0_norm     : 基准梯度归一化到 [0,1]，用于后续 E 参数估计
        G_adapt_u8  : 自适应 beta 后的梯度图
        edge_binary : Otsu 二值边缘
    """
    src = gray.astype(np.float32)

    gx0 = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy0 = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    G0 = cv2.magnitude(gx0, gy0)

    gmin = float(np.min(G0))
    gmax = float(np.max(G0))
    gmean = float(np.mean(G0))

    F1 = 0.5 * (gmin + gmean)
    F2 = 0.5 * (gmax + gmean)

    beta_map = np.full_like(G0, 0.5, dtype=np.float32)

    mid = (G0 > F1) & (G0 <= F2)
    beta_map[mid] = 0.1 + 0.7 * (
        (G0[mid] - gmin) / (gmax - gmin + EPS)
    )
    beta_map[G0 > F2] = 0.1

    # alpha=1 时，M^(1,beta) = (1/(2 beta)) * Sobel kernel
    G_adapt = G0 / (2.0 * beta_map + EPS)

    G_adapt_u8 = cv2.normalize(
        G_adapt, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, edge = cv2.threshold(
        G_adapt_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 轻度闭运算，使中央圆轮廓更连续
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)

    G0_norm = cv2.normalize(G0, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return G0_norm.astype(np.float32), G_adapt_u8, edge


# ============================================================================
# 5. 中央圆/椭圆候选检测
# ============================================================================

def candidate_local_metrics(
    gray: np.ndarray,
    G0_norm: np.ndarray,
    ellipse: Tuple
) -> Optional[Tuple[float, float, float, float]]:
    """用实际 Schneider 模板几何给中央圆候选打分。"""
    rho = ellipse_rho_map(gray.shape, ellipse)

    m_center = (rho >= RHO_CENTER_HOLE) & (rho < RHO_CENTER_MAX)
    sep_max = max(
        RHO_SEPARATOR_MIN + 0.10,
        LAMBDA_IN - RHO_SEPARATOR_MARGIN_TO_CODE
    )
    m_sep = (rho >= RHO_SEPARATOR_MIN) & (rho <= sep_max)

    code_lo = LAMBDA_IN + 0.08
    code_hi = LAMBDA_OUT - 0.08
    m_code = (rho >= code_lo) & (rho <= code_hi)

    if np.count_nonzero(m_center) < 20:
        return None
    if np.count_nonzero(m_sep) < 30:
        return None
    if np.count_nonzero(m_code) < 50:
        return None

    mu_c = float(np.mean(gray[m_center]))
    mu_s = float(np.mean(gray[m_sep]))

    contrast = abs(mu_c - mu_s) / (mu_c + mu_s + EPS)
    code_std = float(np.std(gray[m_code]))

    edge_scores = []
    for r0 in (1.0, LAMBDA_IN, LAMBDA_OUT):
        m = np.abs(rho - r0) <= EDGE_BAND_HALF_WIDTH
        vals = G0_norm[m]
        if vals.size == 0:
            edge_scores.append(0.0)
        else:
            edge_scores.append(
                float(np.percentile(vals, EDGE_RESPONSE_PERCENTILE))
            )

    edge_support = float(np.mean(edge_scores))
    code_std_norm = float(np.clip(code_std / 100.0, 0.0, 1.0))
    score = 0.50 * contrast + 0.20 * code_std_norm + 0.30 * edge_support

    return contrast, code_std, edge_support, score


def detect_coarse_candidates(
    gray: np.ndarray,
    G0_norm: np.ndarray,
    edge_binary: np.ndarray
) -> List[Candidate]:
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    max_diameter = MAX_ELLIPSE_DIAMETER_RATIO * min_dim

    contours, _ = cv2.findContours(
        edge_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    raw: List[Candidate] = []

    for contour in contours:
        if len(contour) < MIN_CONTOUR_POINTS:
            continue

        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue

        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            continue

        (_, _), (A, B), _ = ellipse
        dmin = min(float(A), float(B))
        dmax = max(float(A), float(B))

        if dmin < MIN_ELLIPSE_DIAMETER_PX:
            continue
        if dmax > max_diameter:
            continue

        axis_ratio = dmin / (dmax + EPS)
        if axis_ratio < MIN_AXIS_RATIO:
            continue

        circ = circularity_of_contour(contour)
        if circ < MIN_CIRCULARITY:
            continue

        fit_residual = ellipse_fit_residual(contour, ellipse)
        if fit_residual > MAX_ELLIPSE_FIT_RESIDUAL:
            continue

        if not valid_outer_extent(ellipse, gray.shape):
            continue

        metrics = candidate_local_metrics(gray, G0_norm, ellipse)
        if metrics is None:
            continue

        contrast, code_std, edge_support, score = metrics

        if contrast < MIN_PHOTOMETRIC_CONTRAST:
            continue
        if code_std < MIN_CODE_RING_STD:
            continue
        if score < MIN_CANDIDATE_SCORE:
            continue

        raw.append(
            Candidate(
                ellipse=ellipse,
                contour_area=area,
                circularity=circ,
                axis_ratio=axis_ratio,
                ellipse_fit_residual=fit_residual,
                contrast_C=contrast,
                code_std=code_std,
                edge_support=edge_support,
                score=score,
            )
        )

    # 得分高的优先，随后按中心位置去重
    raw.sort(key=lambda c: c.score, reverse=True)

    selected: List[Candidate] = []
    for cand in raw:
        (cx, cy), _, _ = cand.ellipse
        rc = ellipse_mean_radius(cand.ellipse)

        duplicate = False
        for kept in selected:
            (kx, ky), _, _ = kept.ellipse
            rk = ellipse_mean_radius(kept.ellipse)
            d = math.hypot(float(cx) - float(kx), float(cy) - float(ky))
            if d < DEDUP_CENTER_FACTOR * min(rc, rk):
                duplicate = True
                break

        if not duplicate:
            selected.append(cand)

        if len(selected) >= MAX_TARGETS:
            break

    return selected


# ============================================================================
# 6. 核心：由 C 和 E 显式确定 alpha、beta
# ============================================================================

def estimate_alpha_beta(
    gray: np.ndarray,
    G0_norm: np.ndarray,
    ellipse: Tuple
) -> Dict[str, float]:
    """
    由靶标局部特征估计 C、E、alpha 与 beta_edge。

    C = |mu_center - mu_gap| / (mu_center + mu_gap + eps)
    E = G_T / (G_T + G_N + eps)
    alpha = 0.1 + 0.9*C

    beta_edge = beta_min + (beta_max-beta_min)*alpha*E

    真正参与精检测的是后续构造的 beta(x,y)。
    """
    rho = ellipse_rho_map(gray.shape, ellipse)

    m_center = (
        (rho >= RHO_CENTER_HOLE) &
        (rho < RHO_CENTER_MAX)
    )

    sep_max = max(
        RHO_SEPARATOR_MIN + 0.10,
        LAMBDA_IN - RHO_SEPARATOR_MARGIN_TO_CODE
    )
    m_sep = (
        (rho >= RHO_SEPARATOR_MIN) &
        (rho <= sep_max)
    )

    mu_center = float(np.mean(gray[m_center]))
    mu_separator = float(np.mean(gray[m_sep]))

    C = abs(mu_center - mu_separator) / (
        mu_center + mu_separator + EPS
    )
    C = float(np.clip(C, 0.0, 1.0))

    gt_components = []
    for r0 in (1.0, LAMBDA_IN, LAMBDA_OUT):
        mask = np.abs(rho - r0) <= EDGE_BAND_HALF_WIDTH
        vals = G0_norm[mask]
        if vals.size == 0:
            gt_components.append(0.0)
        else:
            gt_components.append(
                float(np.percentile(vals, EDGE_RESPONSE_PERCENTILE))
            )

    G_target = float(np.mean(gt_components))

    noise_hi = max(
        NOISE_RHO_MIN + 0.08,
        LAMBDA_IN - NOISE_MARGIN_TO_CODE
    )
    m_noise = (
        (rho >= NOISE_RHO_MIN) &
        (rho <= noise_hi)
    )
    noise_vals = G0_norm[m_noise]
    G_noise = float(np.mean(noise_vals)) if noise_vals.size else 0.0

    E = G_target / (G_target + G_noise + EPS)
    E = float(np.clip(E, 0.0, 1.0))

    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * C
    alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))

    beta_edge = (
        BETA_MIN +
        (BETA_MAX - BETA_MIN) * alpha * E
    )
    beta_edge = float(np.clip(beta_edge, BETA_MIN, BETA_MAX))

    return {
        "C": C,
        "E": E,
        "alpha": alpha,
        "beta": beta_edge,
        "mu_center": mu_center,
        "mu_separator": mu_separator,
        "G_target": G_target,
        "G_noise": G_noise,
        "G1": gt_components[0],
        "G2": gt_components[1],
        "G3": gt_components[2],
    }


# ============================================================================
# 7. DF3SOp 显式算子
# ============================================================================

def build_df3sop_kernels(
    alpha: float,
    beta: float,
    zero_sum: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据论文 Table 1 构造 3x3 DF3SOp。

    Mx = 1 / beta^alpha *
         [ -a/2, (1-a)/2,  1/2 ]
         [ -a,   (1-a),    1   ]
         [ -a/2, (1-a)/2,  1/2 ]

    My = 1 / beta^alpha *
         [  1/2,      1,      1/2 ]
         [ (1-a)/2, (1-a), (1-a)/2 ]
         [ -a/2,     -a,     -a/2 ]

    当 alpha=1, beta=1 时与经典 Sobel 仅差整体比例/方向约定。

    注意：原论文边缘检测通常固定 alpha=1，以确保零和性质。
    本研究允许 alpha!=1，因此这里提供一个“改进零和约束”：
        M_tilde = M - mean(M)
    使 sum(M_tilde)=0。

    返回：
        mx_raw, my_raw, mx_used, my_used
    """
    denom = max(beta ** alpha, EPS)

    mx_raw = np.array([
        [-alpha / 2.0, (1.0 - alpha) / 2.0, 0.5],
        [-alpha,       (1.0 - alpha),       1.0],
        [-alpha / 2.0, (1.0 - alpha) / 2.0, 0.5],
    ], dtype=np.float32) / denom

    my_raw = np.array([
        [0.5,                    1.0,               0.5],
        [(1.0 - alpha) / 2.0,   (1.0 - alpha),     (1.0 - alpha) / 2.0],
        [-alpha / 2.0,          -alpha,             -alpha / 2.0],
    ], dtype=np.float32) / denom

    if zero_sum:
        mx_used = mx_raw - np.mean(mx_raw)
        my_used = my_raw - np.mean(my_raw)
    else:
        mx_used = mx_raw.copy()
        my_used = my_raw.copy()

    return mx_raw, my_raw, mx_used.astype(np.float32), my_used.astype(np.float32)


def df3sop_edge(
    gray: np.ndarray,
    alpha: float,
    beta: float,
    zero_sum: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """使用显式 DF3SOp 得到精细梯度和二值边缘。"""
    mx_raw, my_raw, mx, my = build_df3sop_kernels(alpha, beta, zero_sum)

    src = gray.astype(np.float32)
    gx = cv2.filter2D(src, cv2.CV_32F, mx, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.filter2D(src, cv2.CV_32F, my, borderType=cv2.BORDER_REFLECT101)
    mag = cv2.magnitude(gx, gy)

    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge = cv2.threshold(
        mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    edge = cv2.morphologyEx(
        edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )

    return mag, mag_u8, edge, mx_raw, my_raw, mx, my


# ============================================================================
# 7.1 空间 beta(x,y) 参数场
# ============================================================================

def build_spatial_beta_map(
    rho: np.ndarray,
    alpha: float,
    E: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    构造 Schneider 几何引导的空间 beta(x,y)。

    beta_edge = beta_min + (beta_max-beta_min)*alpha*E

    W(rho) = max_i exp[-(rho-r_i)^2/(2*sigma_rho^2)]
    r_i in {1, lambda_in, lambda_out}

    beta(x,y) = beta_max - W(rho)*(beta_max-beta_edge)

    平坦区域 W≈0 -> beta≈beta_max；
    预期结构边缘 W≈1 -> beta≈beta_edge。
    """
    beta_edge = (
        BETA_MIN +
        (BETA_MAX - BETA_MIN) * alpha * E
    )
    beta_edge = float(np.clip(beta_edge, BETA_MIN, BETA_MAX))

    W = np.zeros_like(rho, dtype=np.float32)
    for r0 in (1.0, LAMBDA_IN, LAMBDA_OUT):
        Wi = np.exp(
            -0.5 * ((rho - float(r0)) / SPATIAL_BETA_SIGMA_RHO) ** 2
        ).astype(np.float32)
        W = np.maximum(W, Wi)

    beta_map = (
        BETA_MAX -
        W * (BETA_MAX - beta_edge)
    )
    beta_map = np.clip(
        beta_map, BETA_MIN, BETA_MAX
    ).astype(np.float32)

    return beta_map, W, beta_edge


def df3sop_edge_spatial(
    gray: np.ndarray,
    alpha: float,
    E: float,
    ellipse: Tuple,
    zero_sum: bool = True
):
    """
    空间 beta(x,y) 版 DF3SOp。

    先以 beta=1 构造只由 alpha 决定的零和基础核，
    再用 beta(x,y)^(-alpha) 对每个位置的梯度幅值做局部调制。
    """
    mx_raw, my_raw, mx, my = build_df3sop_kernels(
        alpha, 1.0, zero_sum
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
    mag_base = cv2.magnitude(gx, gy)

    rho = ellipse_rho_map(gray.shape, ellipse)
    beta_map, geometry_weight, beta_edge = build_spatial_beta_map(
        rho, alpha, E
    )

    local_gain = 1.0 / np.maximum(
        np.power(beta_map, alpha),
        EPS
    )
    mag = mag_base * local_gain

    mag_u8 = cv2.normalize(
        mag, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, edge = cv2.threshold(
        mag_u8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    edge = cv2.morphologyEx(
        edge, cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return (
        mag, mag_u8, edge,
        mx_raw, my_raw, mx, my,
        beta_map, geometry_weight, beta_edge
    )


# ============================================================================
# 8. 在局部精检测结果中重新拟合中央椭圆
# ============================================================================

def refine_center_ellipse(
    edge_binary: np.ndarray,
    coarse_ellipse: Tuple
) -> Tuple:
    """从最终 DF3SOp 边缘中，在粗中央圆附近重新选择最匹配的椭圆。"""
    contours, _ = cv2.findContours(
        edge_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    (cx0, cy0), (A0, B0), _ = coarse_ellipse
    r0 = ellipse_mean_radius(coarse_ellipse)

    best = None
    best_cost = float("inf")

    for c in contours:
        if len(c) < MIN_CONTOUR_POINTS:
            continue
        try:
            e = cv2.fitEllipse(c)
        except cv2.error:
            continue

        (cx, cy), (A, B), _ = e
        r = ellipse_mean_radius(e)

        center_shift = math.hypot(float(cx) - float(cx0), float(cy) - float(cy0))
        if center_shift > REFINE_CENTER_TOL_FACTOR * r0:
            continue

        if r < REFINE_SIZE_MIN_FACTOR * r0 or r > REFINE_SIZE_MAX_FACTOR * r0:
            continue

        axis_ratio = min(A, B) / (max(A, B) + EPS)
        if axis_ratio < MIN_AXIS_RATIO:
            continue

        # 中心偏移 + 尺寸变化 + 轮廓圆/椭圆质量的简单代价
        size_err = abs(r - r0) / (r0 + EPS)
        center_err = center_shift / (r0 + EPS)
        circ = circularity_of_contour(c)
        cost = center_err + 0.8 * size_err + 0.25 * max(0.0, 1.0 - circ)

        if cost < best_cost:
            best_cost = cost
            best = e

    return best if best is not None else coarse_ellipse


# ============================================================================
# 9. 12 bit 编码环采样
# ============================================================================

def circular_angle_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def get_local_binary_threshold(gray: np.ndarray, rho: np.ndarray) -> float:
    """在实际 CCT 支持区域内用 Otsu 求黑白阈值。"""
    vals = gray[rho <= (LAMBDA_OUT + 0.10)]
    if vals.size < 20:
        return float(np.mean(gray))

    vals_u8 = vals.astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(
        vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float(thr)


def decode_12bit(
    gray: np.ndarray,
    ellipse: Tuple,
    n_bits: int = 12
) -> Tuple[str, int, int, float, float]:
    """
    在 rho≈(lambda_in+lambda_out)/2 的编码带中进行 12 bit 区域采样。

    原论文定义白=1、黑=0。本程序遵循该约定。
    为避免采样落在两个码元交界处，在一个码元宽度内搜索 phase。

    返回：
        bits_string
        canonical_code          仅循环移位最小值（旋转不变）
        reversed_canonical_code 反向序列的循环最小值（诊断用）
        best_phase_deg
        confidence

    注意：canonical_code 不是 Schneider/Keith 官方 ID，正式 ID 需接入码表。
    """
    rho = ellipse_rho_map(gray.shape, ellipse)
    ang = ellipse_normalized_angle_map(gray.shape, ellipse)

    threshold = get_local_binary_threshold(gray, rho)

    sector = 360.0 / float(n_bits)
    half_width = 0.5 * sector * CODE_ANGULAR_FILL

    radial_mask = (
        (rho >= CODE_RHO_CENTER - CODE_RHO_HALF_WIDTH) &
        (rho <= CODE_RHO_CENTER + CODE_RHO_HALF_WIDTH)
    )

    best_conf = -1.0
    best_bits = None
    best_phase = 0.0

    phases = np.arange(0.0, sector, PHASE_STEP_DEG, dtype=np.float32)

    for phase in phases:
        means = []
        valid = True

        for k in range(n_bits):
            center_angle = (float(phase) + (k + 0.5) * sector) % 360.0
            da = circular_angle_distance_deg(ang, center_angle)
            mask = radial_mask & (da <= half_width)

            vals = gray[mask]
            if vals.size < 5:
                valid = False
                break

            # 中位数比单点/均值更耐局部噪声、反光和边缘污染
            means.append(float(np.median(vals)))

        if not valid:
            continue

        means_np = np.asarray(means, dtype=np.float32)
        bits = (means_np > threshold).astype(np.uint8)  # 白=1，黑=0

        # 平衡约束：排除近乎全黑/全白的伪编码相位
        n_white = int(np.sum(bits))
        if n_white < 2 or n_white > (n_bits - 2):
            continue

        # 码元远离阈值越多，置信度越高；按局部灰度动态范围归一化
        local_min = float(np.min(means_np))
        local_max = float(np.max(means_np))
        dynamic = max(local_max - local_min, 1.0)
        conf = float(np.mean(np.abs(means_np - threshold)) / dynamic * 2.0)
        conf = float(np.clip(conf, 0.0, 1.0))

        if conf > best_conf:
            best_conf = conf
            best_bits = bits.copy()
            best_phase = float(phase)

    if best_bits is None:
        return "", -1, -1, 0.0, 0.0

    bits_list = best_bits.tolist()
    bits_string = "".join(str(int(v)) for v in bits_list)

    def min_rotation_value(seq: List[int]) -> int:
        values = []
        n = len(seq)
        for shift in range(n):
            rot = seq[shift:] + seq[:shift]
            values.append(int("".join(str(int(v)) for v in rot), 2))
        return min(values)

    canonical = min_rotation_value(bits_list)
    reversed_canonical = min_rotation_value(list(reversed(bits_list)))

    return bits_string, canonical, reversed_canonical, best_phase, best_conf


# ============================================================================
# 9.1 编码采样可视化与逐码元诊断
# ============================================================================

def ellipse_point_from_rho_angle(
    ellipse: Tuple,
    rho_value: float,
    angle_deg: float,
) -> Tuple[float, float]:
    """把归一化椭圆坐标 (rho, angle) 转回原图像素坐标。"""
    (cx, cy), (A, B), theta_deg = ellipse
    a = max(float(A) / 2.0, EPS)
    b = max(float(B) / 2.0, EPS)

    phi = math.radians(float(angle_deg))
    theta = math.radians(float(theta_deg))

    # 椭圆自身坐标系
    xr = rho_value * a * math.cos(phi)
    yr = rho_value * b * math.sin(phi)

    # 旋回图像坐标系
    ct = math.cos(theta)
    st = math.sin(theta)
    dx = ct * xr - st * yr
    dy = st * xr + ct * yr

    return float(cx + dx), float(cy + dy)


def collect_decode_samples(
    gray: np.ndarray,
    ellipse: Tuple,
    phase_deg: float,
    n_bits: int = 12,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    按 decode_12bit 已经确定的最佳 phase，重新提取每个码元的诊断量。

    返回：
        threshold：局部 Otsu 黑白阈值
        samples：每个码元的序号、中心角、灰度中位数、bit、样本数等
    """
    rho = ellipse_rho_map(gray.shape, ellipse)
    ang = ellipse_normalized_angle_map(gray.shape, ellipse)
    threshold = get_local_binary_threshold(gray, rho)

    sector = 360.0 / float(n_bits)
    half_width = 0.5 * sector * CODE_ANGULAR_FILL
    radial_mask = (
        (rho >= CODE_RHO_CENTER - CODE_RHO_HALF_WIDTH) &
        (rho <= CODE_RHO_CENTER + CODE_RHO_HALF_WIDTH)
    )

    samples: List[Dict[str, float]] = []
    for k in range(n_bits):
        center_angle = (float(phase_deg) + (k + 0.5) * sector) % 360.0
        da = circular_angle_distance_deg(ang, center_angle)
        mask = radial_mask & (da <= half_width)
        vals = gray[mask]

        if vals.size == 0:
            median_gray = float('nan')
            mean_gray = float('nan')
            bit = -1
        else:
            median_gray = float(np.median(vals))
            mean_gray = float(np.mean(vals))
            bit = int(median_gray > threshold)  # 原文约定：白=1，黑=0

        x, y = ellipse_point_from_rho_angle(
            ellipse, CODE_RHO_CENTER, center_angle
        )

        samples.append({
            'bit_index': int(k),
            'angle_deg': float(center_angle),
            'median_gray': median_gray,
            'mean_gray': mean_gray,
            'bit': int(bit),
            'sample_count': int(vals.size),
            'center_x_px': float(x),
            'center_y_px': float(y),
        })

    return float(threshold), samples


def crop_around_cct(image: np.ndarray, ellipse: Tuple, scale: float = 3.45) -> np.ndarray:
    """按中央椭圆尺寸裁出完整 CCT，用于单目标诊断图。"""
    h, w = image.shape[:2]
    (cx, cy), (A, B), _ = ellipse
    half = 0.5 * scale * max(float(A), float(B))

    x0 = max(0, int(math.floor(float(cx) - half)))
    y0 = max(0, int(math.floor(float(cy) - half)))
    x1 = min(w, int(math.ceil(float(cx) + half)))
    y1 = min(h, int(math.ceil(float(cy) + half)))

    if x1 <= x0 or y1 <= y0:
        return image.copy()
    return image[y0:y1, x0:x1].copy()


def draw_decode_sampling_visualization(
    image_bgr: np.ndarray,
    gray: np.ndarray,
    ellipse: Tuple,
    phase_deg: float,
    bits: str,
    n_bits: int = 12,
) -> Tuple[np.ndarray, float, List[Dict[str, float]]]:
    """
    在原图上可视化 12 个编码采样扇区：
      - 中央圆 rho=1；
      - 编码采样环内/外边界；
      - 12 个实际参与判断的角向扇区；
      - 每个扇区的 bit 序号和值。

    返回裁剪后的单目标可视化图、灰度阈值和逐码元采样信息。
    """
    threshold, samples = collect_decode_samples(
        gray, ellipse, phase_deg, n_bits
    )

    vis = image_bgr.copy()
    overlay = vis.copy()

    # 中央圆（rho=1）
    (cx, cy), (A, B), theta = ellipse
    cv2.ellipse(vis, ellipse, (255, 180, 0), 2, cv2.LINE_AA)

    # 编码采样环的内外边界
    for rr, color in [
        (CODE_RHO_CENTER - CODE_RHO_HALF_WIDTH, (255, 255, 0)),
        (CODE_RHO_CENTER + CODE_RHO_HALF_WIDTH, (255, 255, 0)),
    ]:
        e = (
            (float(cx), float(cy)),
            (float(A) * rr, float(B) * rr),
            float(theta),
        )
        cv2.ellipse(vis, e, color, 1, cv2.LINE_AA)

    sector = 360.0 / float(n_bits)
    half_width = 0.5 * sector * CODE_ANGULAR_FILL
    rho_in = CODE_RHO_CENTER - CODE_RHO_HALF_WIDTH
    rho_out = CODE_RHO_CENTER + CODE_RHO_HALF_WIDTH

    for k, sample in enumerate(samples):
        center_angle = float(sample['angle_deg'])
        bit = int(sample['bit'])

        # 构造扇区多边形：外弧正向 + 内弧反向
        angles_outer = np.linspace(
            center_angle - half_width,
            center_angle + half_width,
            14,
        )
        angles_inner = angles_outer[::-1]

        pts = []
        for aa in angles_outer:
            pts.append(ellipse_point_from_rho_angle(ellipse, rho_out, float(aa)))
        for aa in angles_inner:
            pts.append(ellipse_point_from_rho_angle(ellipse, rho_in, float(aa)))

        poly = np.array(
            [[int(round(x)), int(round(y))] for x, y in pts],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        # 白码(1)用绿色，黑码(0)用红色；仅半透明填充，不改变原图信息
        color = (0, 190, 0) if bit == 1 else (0, 0, 220)
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(vis, [poly], True, color, 1, cv2.LINE_AA)

        sx = int(round(sample['center_x_px']))
        sy = int(round(sample['center_y_px']))
        cv2.circle(vis, (sx, sy), 3, color, -1, cv2.LINE_AA)

        # 标签放在 rho≈2.85 位置，形式“序号:bit”
        tx, ty = ellipse_point_from_rho_angle(ellipse, 2.88, center_angle)
        label = f"{k}:{bit}"
        cv2.putText(
            vis, label,
            (int(round(tx)) - 10, int(round(ty)) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            color, 1, cv2.LINE_AA,
        )

    # 半透明叠加扇区，再重新画边界和标签图层
    vis = cv2.addWeighted(overlay, 0.20, vis, 0.80, 0.0)

    # 重新画中央圆和中心
    cv2.ellipse(vis, ellipse, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(
        vis, (int(round(cx)), int(round(cy))), (255, 0, 255),
        markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2,
    )

    # 图像顶部写采样信息
    id_bits = bits if bits else 'decode failed'
    cv2.putText(
        vis,
        f"bits={id_bits}  phase={phase_deg:.1f} deg  threshold={threshold:.1f}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (255, 0, 255), 1, cv2.LINE_AA,
    )

    return crop_around_cct(vis, ellipse), threshold, samples


def make_unwrapped_code_strip(
    gray: np.ndarray,
    ellipse: Tuple,
    phase_deg: float,
    samples: List[Dict[str, float]],
    n_bits: int = 12,
    width: int = 720,
    height: int = 110,
) -> np.ndarray:
    """
    将 rho≈2.5 的编码环展开为矩形条带。

    横轴：从最佳 phase 开始的 0~360°；
    纵轴：编码环径向范围；
    每 30° 画一条分区线，并标注读取到的 0/1。
    """
    h, w = gray.shape[:2]
    (cx, cy), (A, B), theta_deg = ellipse
    a = max(float(A) / 2.0, EPS)
    b = max(float(B) / 2.0, EPS)
    theta = math.radians(float(theta_deg))
    ct = math.cos(theta)
    st = math.sin(theta)

    # 横向角度从最佳 phase 开始展开整圈；纵向覆盖编码采样环
    angles = float(phase_deg) + np.linspace(0.0, 360.0, width, endpoint=False)
    rhos = np.linspace(
        CODE_RHO_CENTER - CODE_RHO_HALF_WIDTH,
        CODE_RHO_CENTER + CODE_RHO_HALF_WIDTH,
        height,
    )
    phi = np.radians(angles)[None, :]
    rr = rhos[:, None]

    xr = rr * a * np.cos(phi)
    yr = rr * b * np.sin(phi)
    map_x = (float(cx) + ct * xr - st * yr).astype(np.float32)
    map_y = (float(cy) + st * xr + ct * yr).astype(np.float32)

    strip = cv2.remap(
        gray, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=127,
    )
    vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

    # 向下扩一个标注区，避免文字盖住条带
    label_h = 42
    canvas = np.full((height + label_h, width, 3), 255, dtype=np.uint8)
    canvas[:height] = vis

    sector_px = width / float(n_bits)
    for k in range(n_bits + 1):
        x = int(round(k * sector_px))
        x = min(max(x, 0), width - 1)
        cv2.line(canvas, (x, 0), (x, height), (0, 255, 255), 1)

    for k, sample in enumerate(samples):
        x0 = int(round((k + 0.5) * sector_px))
        bit = int(sample['bit'])
        med = float(sample['median_gray'])
        color = (0, 130, 0) if bit == 1 else (0, 0, 200)
        cv2.putText(
            canvas, f"{k}:{bit}",
            (max(0, x0 - 12), height + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"{med:.0f}",
            (max(0, x0 - 10), height + 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1, cv2.LINE_AA,
        )

    return canvas


def save_decode_samples_csv(
    path: str,
    threshold: float,
    samples: List[Dict[str, float]],
) -> None:
    """保存 12 个码元逐扇区采样信息，便于论文和调试。"""
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bit_index', 'angle_deg', 'median_gray', 'mean_gray',
            'threshold', 'bit', 'sample_count', 'center_x_px', 'center_y_px'
        ])
        for s in samples:
            writer.writerow([
                int(s['bit_index']),
                float(s['angle_deg']),
                float(s['median_gray']),
                float(s['mean_gray']),
                float(threshold),
                int(s['bit']),
                int(s['sample_count']),
                float(s['center_x_px']),
                float(s['center_y_px']),
            ])


# ============================================================================
# 10. 单个目标处理
# ============================================================================

def process_one_target(
    index: int,
    image_bgr: np.ndarray,
    gray: np.ndarray,
    G0_norm: np.ndarray,
    cand: Candidate,
    output_dir: str,
) -> TargetResult:
    coarse = cand.ellipse

    # 1) 估计 C / E / alpha / beta_edge
    p = estimate_alpha_beta(gray, G0_norm, coarse)
    alpha = p["alpha"]

    # 2) 空间 beta(x,y) DF3SOp
    (
        mag, mag_u8, edge,
        mx_raw, my_raw, mx, my,
        beta_map, geometry_weight, beta_edge
    ) = df3sop_edge_spatial(
        gray, alpha, p["E"], coarse, ZERO_SUM_CORRECTION
    )

    # 3) 精化中央椭圆
    refined = refine_center_ellipse(edge, coarse)

    # 4) 12 bit 编码
    bits, canonical, reversed_canonical, phase, conf = decode_12bit(
        gray, refined, N_BITS
    )

    # 5) 码表查询：先正向 canonical，再查询 reversed canonical
    schneider_id = CODEBOOK.get(canonical, -1)
    if schneider_id < 0:
        schneider_id = CODEBOOK.get(reversed_canonical, -1)

    # 6) 保存中间结果
    target_dir = os.path.join(output_dir, f"target_{index:02d}")
    os.makedirs(target_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(target_dir, "01_df3sop_spatial_gradient.png"),
        mag_u8
    )
    cv2.imwrite(
        os.path.join(target_dir, "02_df3sop_spatial_edge.png"),
        edge
    )

    beta_vis = cv2.normalize(
        beta_map, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    cv2.imwrite(
        os.path.join(target_dir, "03_beta_map.png"),
        beta_vis
    )

    W_vis = cv2.normalize(
        geometry_weight, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    cv2.imwrite(
        os.path.join(target_dir, "04_geometry_weight.png"),
        W_vis
    )

    sampling_vis, decode_threshold, decode_samples = draw_decode_sampling_visualization(
        image_bgr, gray, refined, phase, bits, N_BITS
    )
    cv2.imwrite(
        os.path.join(target_dir, "05_decode_sampling.png"),
        sampling_vis
    )

    unwrapped = make_unwrapped_code_strip(
        gray, refined, phase, decode_samples, N_BITS
    )
    cv2.imwrite(
        os.path.join(target_dir, "06_decode_unwrapped.png"),
        unwrapped
    )

    save_decode_samples_csv(
        os.path.join(target_dir, "decode_samples.csv"),
        decode_threshold, decode_samples
    )

    with open(
        os.path.join(target_dir, "parameters.txt"),
        "w", encoding="utf-8"
    ) as f:
        f.write("Schneider CCT alpha + spatial beta(x,y) result\n")
        f.write("=" * 55 + "\n")
        f.write(f"lambda_in = {LAMBDA_IN:.8f}\n")
        f.write(f"lambda_out = {LAMBDA_OUT:.8f}\n")
        f.write(f"code_rho_center = {CODE_RHO_CENTER:.8f}\n")
        f.write(f"C = {p['C']:.8f}\n")
        f.write(f"E = {p['E']:.8f}\n")
        f.write(f"alpha = {alpha:.8f}\n")
        f.write(f"beta_edge = {beta_edge:.8f}\n")
        f.write(f"beta_map_min = {float(np.min(beta_map)):.8f}\n")
        f.write(f"beta_map_max = {float(np.max(beta_map)):.8f}\n")
        f.write(f"mu_center = {p['mu_center']:.6f}\n")
        f.write(f"mu_separator = {p['mu_separator']:.6f}\n")
        f.write(f"G_target = {p['G_target']:.8f}\n")
        f.write(f"G_noise = {p['G_noise']:.8f}\n")
        f.write(f"G(rho=1) = {p['G1']:.8f}\n")
        f.write(f"G(rho=lambda_in) = {p['G2']:.8f}\n")
        f.write(f"G(rho=lambda_out) = {p['G3']:.8f}\n")

        f.write("\nSpatial beta formula:\n")
        f.write("beta_edge = beta_min + (beta_max-beta_min)*alpha*E\n")
        f.write(
            "W(rho)=max_i exp(-(rho-r_i)^2/(2*sigma_rho^2)), "
            "r_i={1,lambda_in,lambda_out}\n"
        )
        f.write(
            "beta(x,y)=beta_max-W(rho)*(beta_max-beta_edge)\n"
        )
        f.write("G_spatial=G_alpha / beta(x,y)^alpha\n")
        f.write(f"zero_sum_correction = {ZERO_SUM_CORRECTION}\n")

        f.write("\nRaw alpha-kernel Mx (beta=1):\n")
        f.write(np.array2string(mx_raw, precision=8, suppress_small=False))
        f.write(f"\nraw sum(Mx) = {float(mx_raw.sum()):.10f}\n")
        f.write("\nRaw alpha-kernel My (beta=1):\n")
        f.write(np.array2string(my_raw, precision=8, suppress_small=False))
        f.write(f"\nraw sum(My) = {float(my_raw.sum()):.10f}\n")
        f.write("\nUsed zero-sum Mx:\n")
        f.write(np.array2string(mx, precision=8, suppress_small=False))
        f.write(f"\nused sum(Mx) = {float(mx.sum()):.10f}\n")
        f.write("\nUsed zero-sum My:\n")
        f.write(np.array2string(my, precision=8, suppress_small=False))
        f.write(f"\nused sum(My) = {float(my.sum()):.10f}\n")

        f.write("\nDecode:\n")
        f.write(f"bits = {bits}\n")
        f.write(f"canonical_code = {canonical}\n")
        f.write(f"reversed_canonical_code = {reversed_canonical}\n")
        f.write(f"phase_deg = {phase:.4f}\n")
        f.write(f"decode_confidence = {conf:.6f}\n")
        f.write(f"decode_threshold = {decode_threshold:.6f}\n")
        f.write(f"schneider_id = {schneider_id}\n")

    return TargetResult(
        index=index,
        coarse_ellipse=coarse,
        refined_ellipse=refined,
        C=p["C"],
        E=p["E"],
        alpha=alpha,
        beta=beta_edge,
        beta_map_min=float(np.min(beta_map)),
        beta_map_max=float(np.max(beta_map)),
        mu_center=p["mu_center"],
        mu_separator=p["mu_separator"],
        G_target=p["G_target"],
        G_noise=p["G_noise"],
        bits=bits,
        canonical_code=canonical,
        reversed_canonical_code=reversed_canonical,
        phase_deg=phase,
        decode_confidence=conf,
        schneider_id=schneider_id,
    )


# ============================================================================
# 11. 可视化
# ============================================================================

def draw_results(image: np.ndarray, results: List[TargetResult]) -> np.ndarray:
    vis = image.copy()

    for r in results:
        # 粗椭圆：黄色
        ce = r.coarse_ellipse
        re = r.refined_ellipse

        cv2.ellipse(vis, ce, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.ellipse(vis, re, (0, 0, 255), 2, cv2.LINE_AA)

        (cx, cy), _, _ = re
        cx_i, cy_i = int(round(cx)), int(round(cy))
        cv2.drawMarker(
            vis, (cx_i, cy_i), (255, 0, 0),
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2
        )

        text1 = f"T{r.index}: a={r.alpha:.3f}, b_edge={r.beta:.3f}"
        id_text = str(r.schneider_id) if r.schneider_id >= 0 else "Unknown"
        text2 = f"ID={id_text}, code={r.canonical_code}, conf={r.decode_confidence:.2f}"

        x = max(5, cx_i - 85)
        y = max(20, cy_i - 45)

        cv2.putText(
            vis, text1, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            vis, text2, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1, cv2.LINE_AA
        )

    return vis


def draw_coarse_candidates(image: np.ndarray, candidates: List[Candidate]) -> np.ndarray:
    vis = image.copy()
    for i, c in enumerate(candidates, start=1):
        cv2.ellipse(vis, c.ellipse, (0, 255, 0), 2, cv2.LINE_AA)
        (cx, cy), _, _ = c.ellipse
        cv2.putText(
            vis,
            f"C{i} score={c.score:.2f} res={c.ellipse_fit_residual:.3f}",
            (int(cx) + 5, int(cy) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 120, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


# ============================================================================
# 12. 主程序
# ============================================================================

def main() -> None:
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(
            "没有找到输入图像。请修改程序顶部 IMAGE_PATH：\n"
            f"当前 IMAGE_PATH = {IMAGE_PATH}"
        )

    output_dir = ensure_output_dir(IMAGE_PATH, OUTPUT_DIR)

    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取图像：{IMAGE_PATH}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if USE_GAUSSIAN_BLUR:
        k = GAUSSIAN_KSIZE
        if k % 2 == 0:
            k += 1
        gray_work = cv2.GaussianBlur(gray, (k, k), GAUSSIAN_SIGMA)
    else:
        gray_work = gray.copy()

    cv2.imwrite(os.path.join(output_dir, "01_gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "02_gray_work.png"), gray_work)

    # ----------------------------------------------------------------------
    # Stage A：粗检测，只运行一次
    # ----------------------------------------------------------------------
    G0_norm, coarse_grad_u8, coarse_edge = coarse_adaptive_edge(gray_work)
    cv2.imwrite(os.path.join(output_dir, "03_coarse_gradient.png"), coarse_grad_u8)
    cv2.imwrite(os.path.join(output_dir, "04_coarse_edge.png"), coarse_edge)

    candidates = detect_coarse_candidates(gray_work, G0_norm, coarse_edge)

    coarse_vis = draw_coarse_candidates(image, candidates)
    cv2.imwrite(os.path.join(output_dir, "05_coarse_candidates.png"), coarse_vis)

    print("=" * 78)
    print("Schneider CCT - alpha + spatial beta(x,y)")
    print("=" * 78)
    print(f"输入图像 : {IMAGE_PATH}")
    print(f"输出目录 : {output_dir}")
    print(f"粗候选数 : {len(candidates)}")

    if len(candidates) == 0:
        print("\n没有检测到中央圆候选。")
        print("建议优先调节：MIN_AXIS_RATIO、MIN_CIRCULARITY、MIN_CANDIDATE_SCORE。")
        return

    # ----------------------------------------------------------------------
    # Stage B：每个中央圆直接根据公式得到唯一 alpha,beta，然后只精检测一次
    # ----------------------------------------------------------------------
    results: List[TargetResult] = []

    for i, cand in enumerate(candidates, start=1):
        result = process_one_target(
            i, image, gray_work, G0_norm, cand, output_dir
        )
        results.append(result)

        (cx, cy), _, _ = result.refined_ellipse
        print("-" * 78)
        print(f"Target {i}")
        print(f"  center             = ({cx:.3f}, {cy:.3f}) px")
        print(f"  C                  = {result.C:.6f}")
        print(f"  E                  = {result.E:.6f}")
        print(f"  alpha              = {result.alpha:.6f}")
        print(f"  beta_edge          = {result.beta:.6f}")
        print(f"  beta_map range     = [{result.beta_map_min:.6f}, {result.beta_map_max:.6f}]")
        print(f"  bits               = {result.bits}")
        print(f"  canonical code     = {result.canonical_code}")
        print(f"  Schneider ID       = {result.schneider_id if result.schneider_id >= 0 else 'Unknown'}")
        print(f"  decode confidence  = {result.decode_confidence:.4f}")

    final_vis = draw_results(image, results)
    final_path = os.path.join(output_dir, "06_final_detection.png")
    cv2.imwrite(final_path, final_vis)

    # CSV 汇总
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target", "center_x_px", "center_y_px",
            "C", "E", "alpha", "beta_edge", "beta_map_min", "beta_map_max",
            "mu_center", "mu_separator", "G_target", "G_noise",
            "bits", "canonical_code", "reversed_canonical_code",
            "schneider_id", "phase_deg", "decode_confidence"
        ])
        for r in results:
            (cx, cy), _, _ = r.refined_ellipse
            writer.writerow([
                r.index, cx, cy,
                r.C, r.E, r.alpha, r.beta, r.beta_map_min, r.beta_map_max,
                r.mu_center, r.mu_separator, r.G_target, r.G_noise,
                r.bits, r.canonical_code, r.reversed_canonical_code,
                r.schneider_id, r.phase_deg, r.decode_confidence
            ])

    print("=" * 78)
    print(f"处理完成。最终结果：{final_path}")
    print(f"参数汇总：{csv_path}")
    print("=" * 78)

    if SHOW_WINDOWS:
        cv2.imshow("Coarse candidates", coarse_vis)
        cv2.imshow("Final Schneider CCT detection", final_vis)
        print("按任意键关闭窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
