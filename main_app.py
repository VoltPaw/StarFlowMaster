# -*- coding: utf-8 -*-
"""
StarFlow Master (全中文·深空毛玻璃尊享版)
一站式极客星轨合成引擎
"""
from __future__ import annotations

import os
import re
import sys
import time
import math
import traceback
import threading
import multiprocessing
import concurrent.futures
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import random

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None  # type: ignore

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:
    imageio = None  # type: ignore

from PyQt5 import QtCore, QtGui, QtWidgets


# =============================================================================
# 核心工具与 I/O 防错
# =============================================================================

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _fmt_sec(sec: float) -> str:
    sec = float(sec)
    if not math.isfinite(sec) or sec < 0: return "--"
    if sec < 60: return f"{sec:.1f}秒"
    if sec < 3600: return f"{sec/60:.1f}分钟"
    return f"{sec/3600:.2f}小时"

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def collect_images_from_folder(folder: Union[str, Path], recursive: bool = False) -> List[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir(): return []
    files: List[Path] = []
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for p in iterator:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS: files.append(p)
    files.sort(key=lambda p: natural_key(str(p)))
    return files

def dedup_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        ps = str(p.resolve())
        if ps in seen: continue
        seen.add(ps)
        out.append(p)
    return out

def ensure_cv2() -> None:
    if cv2 is None: raise RuntimeError("OpenCV(cv2) 未安装。请执行: pip install opencv-python")

def ensure_imageio() -> None:
    if imageio is None: raise RuntimeError("imageio 未安装。请执行: pip install imageio imageio-ffmpeg")

def imread_unicode(path: Union[str, Path], flags: int = None) -> Optional[np.ndarray]:
    ensure_cv2()
    p = str(path)
    if flags is None: flags = cv2.IMREAD_COLOR
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0: return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None

def imwrite_unicode(path: Union[str, Path], img: np.ndarray) -> bool:
    ensure_cv2()
    p = str(path)
    try:
        ext = Path(p).suffix.lower()
        if not ext: ext = ".jpg"
        success, data = cv2.imencode(ext, img)
        if success:
            with open(p, "wb") as f: f.write(data.tobytes())
            return True
        return False
    except Exception:
        return False

class TaskCanceled(RuntimeError): pass

class CancelToken:
    def __init__(self): self._ev = threading.Event()
    def cancel(self): self._ev.set()
    def is_cancelled(self) -> bool: return self._ev.is_set()
    def raise_if_cancelled(self):
        if self.is_cancelled(): raise TaskCanceled("任务已被用户取消。")


# =============================================================================
# 算法逻辑封装区
# =============================================================================

def _read_bgr(path: Union[str, Path]) -> Tuple[str, Optional[np.ndarray]]:
    ensure_cv2()
    return str(path), imread_unicode(path, cv2.IMREAD_COLOR)

def make_star_trail_video_decay_maxhold(
    input_paths: List[Union[str, Path]], output_path: Union[str, Path],
    decay: float, fps: int, io_workers: int, prefetch: int, crf: int, preset: str,
    *, cancel: Optional[CancelToken] = None,
    progress_cb: Optional[Callable[[str, int, int, Dict[str, str]], None]] = None, log_cb: Optional[Callable[[str], None]] = None,
) -> None:
    ensure_cv2(); ensure_imageio()
    cancel = cancel or CancelToken()
    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    try: cv2.setNumThreads(0)
    except: pass

    n = len(input_paths)
    t0 = time.time()
    def log(msg: str):
        if log_cb: log_cb(msg)
    def prog(done: int, extra: Optional[Dict[str, str]] = None):
        if progress_cb: progress_cb("视频编码", done, n, extra or {})

    log(f"[残影延时] 开始：{n}张 | 线程数={io_workers}")
    cancel.raise_if_cancelled()

    p0, img0 = _read_bgr(input_paths[0])
    if img0 is None: raise FileNotFoundError(f"无法读取首帧: {p0}")

    H, W = img0.shape[:2]
    result = img0.astype(np.float32)
    result_L = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32)

    writer = None
    try:
        writer = imageio.get_writer(
            str(output_path), fps=fps, codec="libx264", format="ffmpeg",
            ffmpeg_params=["-loglevel", "error", "-pix_fmt", "yuv420p", "-crf", str(int(crf)), "-preset", str(preset)]
        )
    except Exception as e:
        raise RuntimeError("无法打开视频写入器（缺少 FFmpeg 后端）。") from e

    processed = 0
    try:
        writer.append_data(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        processed = 1
        prog(processed)

        futures: List[Optional[object]] = [None] * n
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as ex:
            warm_end = min(n, 1 + max(1, int(prefetch)))
            for i in range(1, warm_end): futures[i] = ex.submit(_read_bgr, input_paths[i])

            for i in range(1, n):
                while True:
                    cancel.raise_if_cancelled()
                    try:
                        p_str, img = futures[i].result(timeout=0.5)  # type: ignore
                        futures[i] = None # 立即释放内存
                        break
                    except concurrent.futures.TimeoutError:
                        continue

                result *= float(decay)
                result_L *= float(decay)

                if img is not None:
                    if img.shape[:2] != (H, W): raise ValueError(f"尺寸不匹配: {Path(p_str).name}")
                    img_L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    idx = img_L > result_L
                    if np.any(idx):
                        result[idx] = img[idx].astype(np.float32)
                        result_L[idx] = img_L[idx]

                out_u8 = np.clip(result, 0, 255).astype(np.uint8)
                writer.append_data(cv2.cvtColor(out_u8, cv2.COLOR_BGR2RGB))

                processed += 1
                prog(processed)

                j = i + max(1, int(prefetch))
                if j < n and futures[j] is None: futures[j] = ex.submit(_read_bgr, input_paths[j])
    finally:
        try:
            if writer is not None: writer.close()
        except Exception: pass
        writer = None
        gc.collect()
    log(f"[残影延时] 完毕 | 耗时={_fmt_sec(time.time()-t0)}")

def composite_max_fullframe(
    image_paths: List[Union[str, Path]], out_path: Union[str, Path], io_workers: int, prefetch: int,
    *, cancel: Optional[CancelToken] = None, progress_cb: Optional[Callable[[str, int, int, Dict[str, str]], None]] = None, log_cb: Optional[Callable[[str], None]] = None,
) -> str:
    ensure_cv2()
    cancel = cancel or CancelToken()
    paths = [Path(p) for p in image_paths]
    try: cv2.setNumThreads(0)
    except: pass
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(paths)
    t0 = time.time()

    def log(msg: str):
        if log_cb: log_cb(msg)
    def prog(done: int, extra: Optional[Dict[str, str]] = None):
        if progress_cb: progress_cb("全局堆栈", done, total, extra or {})

    log(f"[全局堆栈] 开始：输入 {total} 张 | 线程数={io_workers}")
    cancel.raise_if_cancelled()

    first = imread_unicode(paths[0], cv2.IMREAD_COLOR)
    if first is None: raise FileNotFoundError(f"无法读取文件: {paths[0]}")
    max_img = first.copy()
    prog(1)

    rest = paths[1:]
    if not rest:
        imwrite_unicode(out_path, max_img)
        return str(out_path)

    processed = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as ex:
        in_flight = set()
        def submit_one(p: Path): return ex.submit(lambda pp: (str(pp), imread_unicode(pp, cv2.IMREAD_COLOR)), p)
        it = iter(rest)
        for _ in range(min(max(1, int(prefetch)), len(rest))):
            p = next(it, None)
            if p: in_flight.add(submit_one(p))

        while in_flight:
            cancel.raise_if_cancelled()
            done_set, in_flight = concurrent.futures.wait(in_flight, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done_set:
                p_str, img = fut.result()
                processed += 1
                if img is not None: np.maximum(max_img, img, out=max_img)
                prog(processed)
                p = next(it, None)
                if p: in_flight.add(submit_one(p))

    imwrite_unicode(out_path, max_img)
    log(f"[全局堆栈] 完毕 | 耗时={_fmt_sec(time.time()-t0)}")
    return str(out_path)


def load_detection_mask(mask_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    读取掩膜图片，返回二值化后的掩膜矩阵。
    掩膜逻辑：图片中亮度 > 127 的区域为【屏蔽区】(Mask=255)，其余为【检测区】(Mask=0)。
    返回: np.ndarray (uint8), 0=valid, 255=invalid. 
    如果路径无效或读取失败，返回 None。
    """
    ensure_cv2()
    # 强制以 grayscale 读取
    img = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None # 或者抛出异常，视调用方而定
    
    # 阈值判定：> 127 为屏蔽区(255)，<= 127 为有效区(0)
    # THRESH_BINARY: src > thresh ? maxval : 0
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask

def _detect_one_sep_worker(args: Tuple[str, Optional[np.ndarray], float, int, float]) -> Tuple[str, np.ndarray]:
    try:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        
        ensure_cv2()
        cv2.setNumThreads(0)
        
        import sep  # type: ignore
        path_str, mask_img, thresh_sigma, minarea, mask_dilation_pct = args
        img = imread_unicode(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None: return path_str, np.zeros((0, 2), dtype=np.float32)

        # 1. 预处理：将非检测范围（地景）填充为星空背景亮度的均值
        #    目的：平滑地景与星空的交界处，降低高对比度纹理对 sep 算法背景估计的干扰，减少误检。
        process_img = img
        filt_mask = None

        if mask_img is not None:
            H, W = img.shape
            MH, MW = mask_img.shape
            # 只有尺寸匹配时才能进行填充操作
            if H == MH and W == MW:
                # 复制一份以免修改原图缓存（虽然这里是局部变量读取的，但稳妥起见）
                process_img = img.copy()
                
                # mask_img 中 0 是有效检测区，255 是屏蔽区
                # [步骤A] 计算有效区域均值并填充屏蔽区 (抹平地景细节)
                valid_mask = (mask_img == 0)
                invalid_mask = (mask_img == 255)
                
                if np.any(valid_mask):
                    mean_val = np.mean(img[valid_mask])
                else:
                    mean_val = 0
                
                process_img[invalid_mask] = mean_val

                # [步骤B] 生成更严格的筛选掩膜 (边缘收缩)
                # 目的：消除交界处因压缩伪影、锐化或亮度溢出导致的边缘噪点。
                # 原理：对屏蔽区进行形态学膨胀，使其向星空区域扩张指定距离。
                # 扩张距离 = 短边长度 * 百分比 (mask_dilation_pct)
                if mask_dilation_pct > 0.001:
                    expand_dist = int(min(H, W) * (mask_dilation_pct / 100.0)) 
                    if expand_dist < 1: expand_dist = 1
                    
                    # 核大小 k = 2 * dist + 1，确保边缘向外扩张 dist 像素
                    k_size = 2 * expand_dist + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                    
                    # 膨胀屏蔽区(255) -> 进一步侵蚀有效检测区(0)
                    filt_mask = cv2.dilate(mask_img, kernel, iterations=1) 
                else:
                    filt_mask = mask_img # 不做收缩 

        # 2. 对处理后的图像进行星点检测

        # 2. 对处理后的图像进行星点检测
        data = np.ascontiguousarray(process_img.astype(np.float32))
        try:
            bkg = sep.Background(data)
        except Exception:
            # 极少数情况下 sep.Background 可能失败 (如全黑/全白图)
            return path_str, np.zeros((0, 2), dtype=np.float32)
            
        data_sub = data - bkg.back()
        thresh = float(thresh_sigma) * float(bkg.globalrms)
        objects = sep.extract(data_sub, thresh, minarea=int(minarea))
        
        if len(objects) == 0: return path_str, np.zeros((0, 2), dtype=np.float32)

        xs = objects["x"]
        ys = objects["y"]

        # 3. 掩膜后处理：使用收缩后的掩膜剔除无效检测点
        # 这里使用 filt_mask (可能经过了边缘膨胀) 来二次校验坐标
        final_mask = filt_mask if filt_mask is not None else mask_img

        if final_mask is not None:
            H, W = img.shape
            MH, MW = final_mask.shape
            
            # 使用查表法快速筛选落在屏蔽区(mask=255)的星点
            # 兼容性处理：防止尺寸不匹配导致的越界
            ixs = np.rint(xs).astype(np.int32)
            iys = np.rint(ys).astype(np.int32)
            
            valid_coord_mask = (ixs >= 0) & (ixs < MW) & (iys >= 0) & (iys < MH)
            keep_mask = np.ones(len(xs), dtype=bool)
            keep_mask[~valid_coord_mask] = False
            
            valid_indices = np.where(valid_coord_mask)[0]
            if len(valid_indices) > 0:
                pixel_vals = final_mask[iys[valid_indices], ixs[valid_indices]]
                is_valid_star = (pixel_vals == 0) # 0为有效检测区
                keep_mask[valid_indices] = is_valid_star
            
            xs = xs[keep_mask]
            ys = ys[keep_mask]
            
        if len(xs) == 0: return path_str, np.zeros((0, 2), dtype=np.float32)
        return path_str, np.stack([xs, ys], axis=1).astype(np.float32)

    except Exception:
        # 捕获并忽略单张图片的错误，防止单个任务挂掉整个进程池
        return args[0], np.zeros((0, 2), dtype=np.float32)

def detect_stars_sep(
    paths: List[Path], mask_img: Optional[np.ndarray], max_workers: int, thresh_sigma: float, minarea: int, prefetch: int, mask_dilation_pct: float,
    *, cancel: Optional[CancelToken] = None, progress_cb: Optional[Callable[[str, int, int, Dict[str, str]], None]] = None, log_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray]:
    ensure_cv2()
    cancel = cancel or CancelToken()
    try: import sep  # noqa: F401
    except Exception as e: raise RuntimeError("星点检测依赖 `sep`。请执行: pip install sep") from e

    t0 = time.time()
    # 构造多进程任务参数
    # 注：直接传递 mask_img (uint8) 到子进程。Windows spawn 模式下会有序列化开销，但在接受范围内。
    tasks = [(str(p), mask_img, float(thresh_sigma), int(minarea), float(mask_dilation_pct)) for p in paths]
    total = len(tasks)

    def log(msg: str):
        if log_cb: log_cb(msg)
    def prog(done: int):
        if progress_cb: progress_cb("星点检测", done, total, {})

    log(f"[星点检测] 开始：{total}张 | 线程数={max_workers}")
    cancel.raise_if_cancelled()

    results: Dict[str, np.ndarray] = {}
    done_n = 0

    if max_workers <= 1:
        for t in tasks:
            cancel.raise_if_cancelled()
            p_str, xy = _detect_one_sep_worker(t)
            results[p_str] = xy
            done_n += 1
            prog(done_n)
        return results

    try:
        # 并行计算 (ProcessPoolExecutor)
        chunk_size = max(int(max_workers), int(prefetch))
        task_iter = iter(tasks)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(max_workers)) as ex:
            in_flight = set()
            
            # 初始填充
            for _ in range(chunk_size):
                t = next(task_iter, None)
                if t: in_flight.add(ex.submit(_detect_one_sep_worker, t))
            
            while in_flight:
                cancel.raise_if_cancelled()
                
                # 等待至少一个完成，超时时间短一点以便检查 cancel
                done_set, in_flight = concurrent.futures.wait(in_flight, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for fut in done_set:
                    p_str, xy = fut.result()
                    results[p_str] = xy
                    done_n += 1
                    prog(done_n)
                    
                    # 完成一个补一个，保持队列充盈但不过载
                    t = next(task_iter, None)
                    if t: in_flight.add(ex.submit(_detect_one_sep_worker, t))
                    
    except TaskCanceled: 
        log("[星点检测] 检测到取消信号，正在终止剩余进程...")
        raise
    except Exception as e:
        log(f"[星点检测] 多进程执行失败，降级为多线程：{e}")
        # 多线程回退模式 (同样使用流式提交，避免取消卡顿)
        chunk_size = max(int(max_workers), int(prefetch))
        task_iter = iter(tasks)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            in_flight = set()
            for _ in range(chunk_size):
                t = next(task_iter, None)
                if t: in_flight.add(ex.submit(_detect_one_sep_worker, t))
            
            while in_flight:
                cancel.raise_if_cancelled()
                done_set, in_flight = concurrent.futures.wait(in_flight, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done_set:
                    p_str, xy = fut.result()
                    results[p_str] = xy
                    done_n += 1
                    prog(done_n)
                    t = next(task_iter, None)
                    if t: in_flight.add(ex.submit(_detect_one_sep_worker, t))

    log(f"[星点检测] 完毕 | 耗时={_fmt_sec(time.time()-t0)}")
    return results

def composite_max_stars(
    results: Dict[str, np.ndarray], image_paths: List[Path], base_image_path: Union[str, Path],
    out_path: Union[str, Path], star_radius: int, io_workers: int, prefetch: int,
    *, cancel: Optional[CancelToken] = None, progress_cb: Optional[Callable[[str, int, int, Dict[str, str]], None]] = None, log_cb: Optional[Callable[[str], None]] = None,
) -> str:
    ensure_cv2()
    cancel = cancel or CancelToken()
    t0 = time.time()
    
    base_img = imread_unicode(base_image_path, cv2.IMREAD_COLOR)
    if base_img is None: raise FileNotFoundError(f"无法读取基础底图: {base_image_path}")

    H, W, _ = base_img.shape
    all_xy = [xy for xy in results.values() if xy is not None and xy.size > 0]
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if log_cb: log_cb(msg)
    def prog(done: int):
        if progress_cb: progress_cb("星点区域堆栈", done, len(image_paths), {})

    cancel.raise_if_cancelled()
    if not all_xy:
        imwrite_unicode(out_path, base_img)
        log(f"[星点堆栈] 未检测到星点，直接输出底图。")
        return str(out_path)

    xy_stars_all = np.vstack(all_xy)
    pts = np.rint(xy_stars_all).astype(np.int32)
    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[ys, xs] = 255
    if int(star_radius) > 0:
        k = 2 * int(star_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask_bool = mask > 0
    max_star_pixels = base_img[mask_bool].copy()
    done_n = 0
    prog(0)

    def read_and_extract(path_str: str) -> Optional[np.ndarray]:
        img = imread_unicode(path_str, cv2.IMREAD_COLOR)
        if img is None: return None
        return img[mask_bool]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as ex:
        in_flight = set()
        it = iter(image_paths)
        for _ in range(min(max(1, int(prefetch)), len(image_paths))):
            p = next(it, None)
            if p: in_flight.add(ex.submit(read_and_extract, str(p)))

        while in_flight:
            cancel.raise_if_cancelled()
            done_set, in_flight = concurrent.futures.wait(in_flight, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done_set:
                arr = fut.result()
                done_n += 1
                if arr is not None: np.maximum(max_star_pixels, arr, out=max_star_pixels)
                prog(done_n)
                p = next(it, None)
                if p: in_flight.add(ex.submit(read_and_extract, str(p)))

    final_img = base_img.copy()
    final_img[mask_bool] = max_star_pixels
    imwrite_unicode(out_path, final_img)
    log(f"[星点堆栈] 完毕 | 耗时={_fmt_sec(time.time()-t0)}")
    return str(out_path)


# =============================================================================
# 自定义 UI 组件 & 毛玻璃深空渲染
# =============================================================================

APP_NAME = "StarFlow Master"

def build_app_icon(size: int = 256) -> QtGui.QIcon:
    """绘制高定北斗七星图标"""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    
    # 底部星云渐变圈
    grad = QtGui.QRadialGradient(size/2, size/2, size*0.48)
    grad.setColorAt(0.0, QtGui.QColor(30, 60, 110, 240))
    grad.setColorAt(0.6, QtGui.QColor(15, 25, 50, 180))
    grad.setColorAt(1.0, QtGui.QColor(5, 10, 25, 0))
    p.setBrush(QtGui.QBrush(grad))
    p.setPen(QtCore.Qt.NoPen)
    p.drawRoundedRect(0, 0, size, size, size*0.2, size*0.2)
    
    # 北斗七星坐标 (0~1)
    dipper = [(0.85, 0.25), (0.70, 0.35), (0.55, 0.45), (0.40, 0.55), (0.25, 0.80), (0.50, 0.85), (0.65, 0.60)]
    qpts = [QtCore.QPointF(x*size, y*size) for x, y in dipper]
    
    # 连线
    pen = QtGui.QPen(QtGui.QColor(100, 200, 255, 120), size*0.02)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    p.setPen(pen)
    for i in range(3): p.drawLine(qpts[i], qpts[i+1])
    p.drawLine(qpts[3], qpts[4]); p.drawLine(qpts[4], qpts[5])
    p.drawLine(qpts[5], qpts[6]); p.drawLine(qpts[6], qpts[3])
    
    # 星星本体与发光
    p.setPen(QtCore.Qt.NoPen)
    for pt in qpts:
        p.setBrush(QtGui.QColor(0, 229, 255, 80))
        p.drawEllipse(pt, size*0.07, size*0.07)
        p.setBrush(QtGui.QColor(255, 255, 255, 255))
        p.drawEllipse(pt, size*0.035, size*0.035)
    
    p.end()
    return QtGui.QIcon(pm)

class StarryBackgroundWidget(QtWidgets.QWidget):
    """深空动态渐变+繁星渲染组件 (拟真旋转星空版)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stars = []
        self.num_stars = 16000  # 增加星星数量
        
        # 预计算一些随机星点数据
        # 结构: [angle(弧度), dist(距离中心归一化), size, speed_factor, color_type, flicker_phase, flicker_speed]
        # dist: 0.0 -> center, 1.4 -> corner (cover full screen rotation)
        for _ in range(self.num_stars):
            self.stars.append(self._random_star_init())
            
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_stars)
        self.timer.start(30)  # ~30fps

    def _random_star_init(self):
        # 极坐标角度: 0 ~ 2pi
        angle = random.uniform(0, 2 * math.pi)
        
        # 距离: 偏向于均匀分布在面积上，而不是半径上 (sqrt(random))
        # 覆盖屏幕对角线以上，确保旋转时不会留黑角
        dist = math.sqrt(random.uniform(0, 2.2)) 
        
        # 大小分布: 大部分是小星，极少大星
        # 0: micro (pixel), 1: small (soft), 2: medium (glow), 3: bright (spikes)
        rand_val = random.random()
        if rand_val > 0.99: size_type = 3     # 1% 特亮星
        elif rand_val > 0.94: size_type = 2   # 5% 亮星
        elif rand_val > 0.70: size_type = 1   # 24% 普通星
        else: size_type = 0                   # 70% 微星Background
        
        # 旋转速度因子 (视差效果: 远的/边缘的转稍微慢一点? 
        # 其实真实星空是角速度一致的，但为了艺术效果，稍微加一点点层次差异)
        speed_factor = random.uniform(0.9, 1.1)

        # 颜色倾向: 0=白, 1=微蓝, 2=微黄
        c_rand = random.random()
        color_type = 1 if c_rand > 0.7 else (2 if c_rand > 0.9 else 0)
        
        flicker_phase = random.uniform(0, 2 * math.pi)
        flicker_speed = random.uniform(0.05, 0.15) if size_type > 1 else 0.0
        
        return [angle, dist, size_type, speed_factor, color_type, flicker_phase, flicker_speed]

    def update_stars(self):
        # 基础旋转速度
        base_rotation = 0.00077
        
        for i in range(len(self.stars)):
            # star: [angle, dist, size_type, speed, color, f_phase, f_speed]
            star = self.stars[i]
            
            # 更新角度 (改为减法，实现逆时针旋转)
            # 在屏幕坐标系下(Y向下)，顺时针是增加角度，逆时针是减小角度
            star[0] -= base_rotation * star[3]
            
            # 保证角度在 0-2pi 循环
            if star[0] < 0:
                star[0] += 2 * math.pi
                
            # 更新闪烁
            if star[2] > 0:
                star[5] += star[6]

        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        max_dim = max(w, h)
        
        # --- 1. 背景: 深空渐变 (保持深邃感) ---
        grad = QtGui.QRadialGradient(cx, cy, max_dim * 0.8)
        grad.setColorAt(0.0, QtGui.QColor(15, 20, 30))   # 中心微亮
        grad.setColorAt(0.6, QtGui.QColor(5, 8, 15))     #这也是一种深蓝黑
        grad.setColorAt(1.0, QtGui.QColor(0, 0, 5))      # 边缘接近纯黑
        p.fillRect(self.rect(), grad)

        p.setPen(QtCore.Qt.NoPen)
        
        # 预定义颜色
        colors = [
            QtGui.QColor(255, 255, 255),    # 白
            QtGui.QColor(200, 230, 255),    # 蓝白
            QtGui.QColor(255, 245, 200)     # 暖白
        ]

        # --- 2. 绘制旋转星空 ---
        for star in self.stars:
            angle, dist, size_type, speed, c_type, f_phase, f_speed = star
            
            # 极坐标 -> 笛卡尔坐标
            # r = dist * max_dim * 0.8 (缩放因子)
            r = dist * max_dim * 0.7
            sx = cx + math.cos(angle) * r
            sy = cy + math.sin(angle) * r
            
            # 视口裁剪 (简单的边界检查，稍微放宽一点以免边缘闪烁)
            if not (-50 < sx < w + 50 and -50 < sy < h + 50):
                continue

            # 基础透明度闪烁
            flicker = 1.0
            if size_type > 0:
                flicker = 0.7 + 0.3 * math.sin(f_phase)
            
            base_color = colors[c_type]
            alpha = int(255 * flicker * (0.4 if size_type == 0 else 0.8)) # 小星星暗一点
            
            if size_type == 0:
                # Type 0: 微星 (单像素点)
                p.setPen(QtGui.QColor(base_color.red(), base_color.green(), base_color.blue(), alpha))
                p.drawPoint(int(sx), int(sy))
                p.setPen(QtCore.Qt.NoPen) # Reset
                
            elif size_type == 1:
                # Type 1:同时也柔和一点的小圆点
                p.setBrush(QtGui.QColor(base_color.red(), base_color.green(), base_color.blue(), alpha))
                p.drawEllipse(QtCore.QPointF(sx, sy), 1.0, 1.0)
                
            elif size_type == 2:
                # Type 2: 亮星 (带微弱光晕)
                # 核心
                p.setBrush(QtGui.QColor(255, 255, 255, alpha))
                p.drawEllipse(QtCore.QPointF(sx, sy), 1.5, 1.5)
                # 光晕
                p.setBrush(QtGui.QColor(base_color.red(), base_color.green(), base_color.blue(), int(alpha * 0.3)))
                p.drawEllipse(QtCore.QPointF(sx, sy), 3.0, 3.0)
                
            elif size_type == 3:
                # Type 3: 特亮星 (带十字星芒)
                # 大光晕
                p.setBrush(QtGui.QColor(base_color.red(), base_color.green(), base_color.blue(), int(alpha * 0.2)))
                p.drawEllipse(QtCore.QPointF(sx, sy), 5.0, 5.0)
                # 核心
                p.setBrush(QtGui.QColor(255, 255, 255, alpha))
                p.drawEllipse(QtCore.QPointF(sx, sy), 2.0, 2.0)
                
                # 十字芒 (画两条细线)
                # 随时间缓慢旋转一点点，增加动感? 或者固定角度
                spike_len = 5.0 + 2.0 * flicker # 芒刺长度随闪烁伸缩
                p.setPen(QtGui.QColor(255, 255, 255, int(alpha * 0.6)))
                # 横向
                p.drawLine(QtCore.QPointF(sx - spike_len, sy), QtCore.QPointF(sx + spike_len, sy))
                # 纵向
                p.drawLine(QtCore.QPointF(sx, sy - spike_len), QtCore.QPointF(sx, sy + spike_len))
                p.setPen(QtCore.Qt.NoPen)

class DropListWidget(QtWidgets.QListWidget):
    """
    修改后的文件列表：
    1. 移除左侧原本较丑的三角箭头 (item indicator)
    2. 设置更好的 selection 样式
    """
    filesDropped = QtCore.pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # 隐藏默认左侧小三角
        self.setStyleSheet("""
            QListWidget::item { padding: 4px; margin: 2px; }
            QListWidget::item:selected { 
                background: rgba(0, 229, 255, 40); 
                border: 1px solid rgba(0, 229, 255, 120); 
                border-radius: 4px; 
            }
        """)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event: QtGui.QDropEvent):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths: self.filesDropped.emit(paths)

@dataclass
class AlgorithmConfig:
    name: str
    stages: List[str]

class TaskSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(dict)
    log = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    canceled = QtCore.pyqtSignal(str)

class TaskWorker(QtCore.QObject):
    def __init__(self, task_name: str, fn: Callable[[], str], cfg: AlgorithmConfig, cancel: CancelToken):
        super().__init__()
        self.task_name = task_name
        self.fn = fn
        self.cfg = cfg
        self.cancel = cancel
        self.signals = TaskSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.signals.log.emit(f"== {self.task_name} ==")
            out = self.fn()
            self.signals.finished.emit({"task": self.task_name, "output_path": out})
        except TaskCanceled:
            self.signals.canceled.emit(self.task_name)
        except Exception:
            self.signals.error.emit(traceback.format_exc())

class TaskMonitor(QtWidgets.QFrame):
    cancelClicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        layout = QtWidgets.QVBoxLayout(self)
        
        top = QtWidgets.QHBoxLayout()
        self.lblTitle = QtWidgets.QLabel("状态监控")
        self.lblTitle.setObjectName("MonitorTitle")
        self.lblSub = QtWidgets.QLabel("空闲")
        self.lblSub.setObjectName("MonitorSubTitle")
        top.addWidget(self.lblTitle); top.addStretch(1); top.addWidget(self.lblSub)
        layout.addLayout(top)

        self.prog_layout = QtWidgets.QVBoxLayout()
        
        self.grp1 = QtWidgets.QWidget()
        l1 = QtWidgets.QHBoxLayout(self.grp1); l1.setContentsMargins(0,0,0,0); l1.setSpacing(15) # 增加文本与进度条的间距
        self.lblProg1 = QtWidgets.QLabel("任务进度"); self.lblProg1.setFixedWidth(140) # 再次增加文字区域宽度，防止被遮挡
        self.prog1 = QtWidgets.QProgressBar(); self.prog1.setRange(0, 100); self.prog1.setValue(0)
        l1.addWidget(self.lblProg1); l1.addWidget(self.prog1)
        self.prog_layout.addWidget(self.grp1)

        self.grp2 = QtWidgets.QWidget()
        l2 = QtWidgets.QHBoxLayout(self.grp2); l2.setContentsMargins(0,0,0,0); l2.setSpacing(15) # 增加文本与进度条的间距
        self.lblProg2 = QtWidgets.QLabel("局部堆栈"); self.lblProg2.setFixedWidth(140)
        self.prog2 = QtWidgets.QProgressBar(); self.prog2.setRange(0, 100); self.prog2.setValue(0)
        l2.addWidget(self.lblProg2); l2.addWidget(self.prog2)
        self.prog_layout.addWidget(self.grp2)
        
        layout.addLayout(self.prog_layout)

        metrics = QtWidgets.QHBoxLayout()
        self.chipCount = QtWidgets.QLabel("进度: --")
        self.chipElapsed = QtWidgets.QLabel("已用: --")
        self.chipETA = QtWidgets.QLabel("剩余: --")
        for w in (self.chipCount, self.chipElapsed, self.chipETA):
            w.setObjectName("Chip")
            metrics.addWidget(w)
        
        metrics.addStretch(1)
        self.btnCancel = QtWidgets.QPushButton("取消任务")
        self.btnCancel.setObjectName("BtnDanger")
        self.btnCancel.setEnabled(False)
        self.btnCancel.clicked.connect(self.cancelClicked.emit)
        metrics.addWidget(self.btnCancel)
        layout.addLayout(metrics)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("LogView")
        layout.addWidget(self.log, 1)
        
        self.set_dual_mode(False)

    def set_dual_mode(self, dual: bool, task_type: str = ""):
        if dual:
            # 星点堆栈双模式
            self.lblProg1.setText("星点检测")
            self.lblProg2.setText("局部堆栈")
            self.prog1.setValue(0); self.prog2.setValue(0)
            self.grp2.show()
        else:
            # 单模式根据任务类型设置文案
            if task_type == "afterimage_video":
                self.lblProg1.setText("视频合成")
            elif task_type == "global_stack":
                self.lblProg1.setText("全局堆栈")
            else:
                self.lblProg1.setText("任务进度")
                
            self.prog1.setValue(0)
            self.grp2.hide()

    def append_log(self, msg: str):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def set_running(self, running: bool):
        self.btnCancel.setEnabled(running)
        if not running: self.lblSub.setText("空闲")

    def update_progress(self, payload: dict):
        self.lblSub.setText(payload.get("task", ""))
        stage = payload.get("stage", "")
        
        if payload.get("is_dual"):
            # 双进度条逻辑：下方文本显示当前阶段的详情
            if stage == "星点检测":
                self.prog1.setValue(int(payload.get("stage_pct", 0)))
            elif stage == "星点区域堆栈":
                self.prog1.setValue(100) 
                self.prog2.setValue(int(payload.get("stage_pct", 0)))
        else:
            self.prog1.setValue(int(payload.get("overall_pct", 0)))

        self.chipCount.setText(f"进度: {payload.get('done', 0)}/{payload.get('total', 0)}")
        
        # 显示当前阶段的耗时与剩余时间
        self.chipElapsed.setText(f"已用: {_fmt_sec(payload.get('stage_elapsed', 0.0))}")
        self.chipETA.setText(f"剩余: {_fmt_sec(payload.get('stage_eta', float('inf')))}")


class InputPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        self._paths: List[Path] = []
        layout = QtWidgets.QVBoxLayout(self)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.btnAddFiles = QtWidgets.QPushButton("添加文件"); self.btnAddFiles.setObjectName("BtnPrimary")
        self.btnAddFolder = QtWidgets.QPushButton("添加文件夹"); self.btnAddFolder.setObjectName("BtnPrimary")
        self.btnRemove = QtWidgets.QPushButton("移除选中"); self.btnRemove.setObjectName("BtnSecondary")
        self.btnClear = QtWidgets.QPushButton("清空"); self.btnClear.setObjectName("BtnSecondary")
        btn_row.addWidget(self.btnAddFiles); btn_row.addWidget(self.btnAddFolder)
        btn_row.addWidget(self.btnRemove); btn_row.addWidget(self.btnClear)
        layout.addLayout(btn_row)

        self.list = DropListWidget()
        self.list.setObjectName("FileList")
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setToolTip("可直接将文件或文件夹拖拽到此处。双击可打开图片。")
        layout.addWidget(self.list, 1)
        
        info = QtWidgets.QHBoxLayout()
        self.lblCount = QtWidgets.QLabel("0 张图片")
        self.lblCount.setObjectName("HintText")
        self.lblSize = QtWidgets.QLabel("尺寸: --")
        self.lblSize.setObjectName("HintText")
        info.addWidget(self.lblCount); info.addStretch(1); info.addWidget(self.lblSize)
        layout.addLayout(info)

        self.btnAddFiles.clicked.connect(self.add_files)
        self.btnAddFolder.clicked.connect(self.add_folder)
        self.btnRemove.clicked.connect(self.remove_selected)
        self.btnClear.clicked.connect(self.clear)
        self.list.filesDropped.connect(self._on_drop)
        self.list.itemDoubleClicked.connect(self._on_double_click)

    def paths(self) -> List[Path]: return list(self._paths)

    def set_enabled(self, enabled: bool):
        for w in [self.btnAddFiles, self.btnAddFolder, self.btnRemove, self.btnClear, self.list]: w.setEnabled(enabled)

    def _update_ui(self):
        self.list.clear()
        for p in self._paths:
            item = QtWidgets.QListWidgetItem(p.name)
            item.setData(QtCore.Qt.UserRole, str(p))
            self.list.addItem(item)
        self.lblCount.setText(f"{len(self._paths)} 张图片")
        
        if self._paths and cv2 is not None:
            try:
                img = imread_unicode(self._paths[0], cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    self.lblSize.setText(f"尺寸: {w}×{h}")
                else: self.lblSize.setText("尺寸: --")
            except: self.lblSize.setText("尺寸: --")
        else: self.lblSize.setText("尺寸: --")

    def _on_drop(self, paths: List[Path]):
        files: List[Path] = []
        for p in paths:
            if p.is_dir(): files.extend(collect_images_from_folder(p, False))
            elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS: files.append(p)
        if files:
            self._paths = dedup_paths(self._paths + files)
            self._paths.sort(key=lambda p: natural_key(str(p)))
            self._update_ui()

    def add_files(self):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dlg.exec_():
            files = [Path(p) for p in dlg.selectedFiles() if Path(p).suffix.lower() in SUPPORTED_EXTS]
            self._on_drop(files)

    def add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder: self._on_drop([Path(folder)])

    def remove_selected(self):
        remove_set = {Path(i.data(QtCore.Qt.UserRole)).resolve() for i in self.list.selectedItems()}
        self._paths = [p for p in self._paths if p.resolve() not in remove_set]
        self._update_ui()

    def clear(self):
        self._paths = []
        self._update_ui()

    def _on_double_click(self, item):
        path = item.data(QtCore.Qt.UserRole)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))


class OutputPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        layout = QtWidgets.QHBoxLayout(self)
        self.edtDir = QtWidgets.QLineEdit()
        self.edtDir.setPlaceholderText("输出文件夹路径...")
        # 增加按钮最小宽度，使其更大
        self.btnBrowse = QtWidgets.QPushButton("浏览"); self.btnBrowse.setObjectName("BtnPrimary"); self.btnBrowse.setMinimumWidth(80)
        self.btnOpen = QtWidgets.QPushButton("打开"); self.btnOpen.setObjectName("BtnSecondary"); self.btnOpen.setMinimumWidth(80)
        
        layout.addWidget(QtWidgets.QLabel("导出文件夹:"))
        layout.addWidget(self.edtDir, 1) # 输入框铺满
        layout.addWidget(self.btnBrowse)
        layout.addWidget(self.btnOpen)
        self.btnBrowse.clicked.connect(self._browse)
        self.btnOpen.clicked.connect(self._open_dir)
        self.edtDir.setText(str(Path.home() / "StarFlowMaster_Output"))

    def output_dir(self) -> Path: return Path(self.edtDir.text()).expanduser()
    def _browse(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出文件夹", self.edtDir.text())
        if folder: self.edtDir.setText(folder)
    def _open_dir(self):
        p = self.output_dir()
        p.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))


class AlgoTabBase(QtWidgets.QWidget):
    runRequested = QtCore.pyqtSignal(str)
    def set_enabled(self, enabled: bool):
        for w in self.findChildren(QtWidgets.QWidget):
            if isinstance(w, QtWidgets.QPushButton) and w.text() == "运行":
                w.setEnabled(enabled)
            elif w.objectName() != "DontDisable":
                w.setEnabled(enabled)

class TabGlobalStack(AlgoTabBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        g = QtWidgets.QGridLayout()
        
        self.edtName = QtWidgets.QLineEdit("最大值全局堆栈")
        name_lay = QtWidgets.QHBoxLayout()
        name_lay.setContentsMargins(0,0,0,0)
        name_lay.addWidget(self.edtName); name_lay.addWidget(QtWidgets.QLabel(".jpg"))

        self.spinWorkers = QtWidgets.QSpinBox(); self.spinWorkers.setRange(1, 64); self.spinWorkers.setValue(8); self.spinWorkers.setSingleStep(2)
        self.spinPrefetch = QtWidgets.QSpinBox(); self.spinPrefetch.setRange(1, 512); self.spinPrefetch.setValue(16); self.spinPrefetch.setSingleStep(8)
        
        g.addWidget(QtWidgets.QLabel("输出文件名"), 0, 0); g.addLayout(name_lay, 0, 1, 1, 3)
        g.addWidget(QtWidgets.QLabel("并发线程数"), 1, 0); g.addWidget(self.spinWorkers, 1, 1)
        g.addWidget(QtWidgets.QLabel("缓存图片数"), 1, 2); g.addWidget(self.spinPrefetch, 1, 3)
        layout.addLayout(g)
        
        self.btnRun = QtWidgets.QPushButton("运行"); self.btnRun.setObjectName("BtnNeon")
        self.btnRun.clicked.connect(lambda: self.runRequested.emit("global_stack"))
        layout.addWidget(self.btnRun)
        layout.addStretch(1)

    def params(self):
        return {"output_name": self.edtName.text().strip() + ".jpg", "io_workers": self.spinWorkers.value(), "prefetch": self.spinPrefetch.value()}


class TabAfterimageVideo(AlgoTabBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        g = QtWidgets.QGridLayout()
        
        self.edtName = QtWidgets.QLineEdit("残影延时")
        name_lay = QtWidgets.QHBoxLayout()
        name_lay.setContentsMargins(0,0,0,0)
        name_lay.addWidget(self.edtName); name_lay.addWidget(QtWidgets.QLabel(".mp4"))

        self.spinDecay = QtWidgets.QDoubleSpinBox(); self.spinDecay.setRange(0.0001, 1.0); self.spinDecay.setValue(0.9880); self.spinDecay.setDecimals(4); self.spinDecay.setSingleStep(0.0025)
        self.spinFps = QtWidgets.QSpinBox(); self.spinFps.setRange(1, 120); self.spinFps.setValue(25); self.spinFps.setSingleStep(5)
        self.spinWorkers = QtWidgets.QSpinBox(); self.spinWorkers.setRange(1, 64); self.spinWorkers.setValue(8); self.spinWorkers.setSingleStep(2)
        self.spinPrefetch = QtWidgets.QSpinBox(); self.spinPrefetch.setRange(1, 256); self.spinPrefetch.setValue(16); self.spinPrefetch.setSingleStep(8)
        self.spinCrf = QtWidgets.QSpinBox(); self.spinCrf.setRange(0, 51); self.spinCrf.setValue(20)
        
        self.cmbPreset = QtWidgets.QComboBox()
        self.cmbPreset.addItems(["极速", "较快", "快速", "中等", "较慢"])
        self.cmbPreset.setCurrentText("快速")
        
        g.addWidget(QtWidgets.QLabel("输出文件名"), 0, 0); g.addLayout(name_lay, 0, 1, 1, 3)
        g.addWidget(QtWidgets.QLabel("拖尾衰减系数"), 1, 0); g.addWidget(self.spinDecay, 1, 1)
        g.addWidget(QtWidgets.QLabel("视频帧率"), 1, 2); g.addWidget(self.spinFps, 1, 3)
        g.addWidget(QtWidgets.QLabel("并发线程数"), 2, 0); g.addWidget(self.spinWorkers, 2, 1)
        g.addWidget(QtWidgets.QLabel("缓存图片数"), 2, 2); g.addWidget(self.spinPrefetch, 2, 3)
        g.addWidget(QtWidgets.QLabel("视频清晰度"), 3, 0); g.addWidget(self.spinCrf, 3, 1)
        g.addWidget(QtWidgets.QLabel("编码速度"), 3, 2); g.addWidget(self.cmbPreset, 3, 3)
        layout.addLayout(g)

        self.btnRun = QtWidgets.QPushButton("运行"); self.btnRun.setObjectName("BtnNeon")
        self.btnRun.clicked.connect(lambda: self.runRequested.emit("afterimage_video"))
        layout.addWidget(self.btnRun)
        layout.addStretch(1)

    def params(self):
        preset_map = {"极速": "ultrafast", "较快": "superfast", "快速": "fast", "中等": "medium", "较慢": "slow"}
        return {"output_name": self.edtName.text().strip() + ".mp4", "decay": self.spinDecay.value(), "fps": self.spinFps.value(),
                "io_workers": self.spinWorkers.value(), "prefetch": self.spinPrefetch.value(), "crf": self.spinCrf.value(), 
                "preset": preset_map.get(self.cmbPreset.currentText(), "fast")}


class TabStarStack(AlgoTabBase):
    testRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        g = QtWidgets.QGridLayout()
        
        self.edtName = QtWidgets.QLineEdit("最大值星点堆栈")
        name_lay = QtWidgets.QHBoxLayout(); name_lay.setContentsMargins(0,0,0,0)
        name_lay.addWidget(self.edtName); name_lay.addWidget(QtWidgets.QLabel(".jpg"))
        
        self.edtMask = QtWidgets.QLineEdit(""); self.edtMask.setPlaceholderText("可选：不选择则全局检测")
        self.btnMask = QtWidgets.QPushButton("浏览"); self.btnMask.setObjectName("BtnSecondary"); self.btnMask.setMinimumWidth(80)
        self.btnMask.clicked.connect(self._browse)
        
        self.spinThresh = QtWidgets.QDoubleSpinBox(); self.spinThresh.setRange(0.1, 1000.0); self.spinThresh.setValue(15)
        self.spinMinArea = QtWidgets.QSpinBox(); self.spinMinArea.setRange(1, 999); self.spinMinArea.setValue(5)
        self.spinRadius = QtWidgets.QSpinBox(); self.spinRadius.setRange(0, 50); self.spinRadius.setValue(3)
        self.spinDilation = QtWidgets.QDoubleSpinBox(); self.spinDilation.setRange(0.0, 50.0); self.spinDilation.setValue(1.0); self.spinDilation.setSingleStep(0.5); self.spinDilation.setSuffix("%")

        self.cmbBase = QtWidgets.QComboBox(); self.cmbBase.addItems(["最后一张", "第一张", "自定义"])
        self.edtCustomBase = QtWidgets.QLineEdit(""); self.edtCustomBase.setPlaceholderText("选择自定义底图...")
        self.btnCustomBase = QtWidgets.QPushButton("浏览"); self.btnCustomBase.setObjectName("BtnSecondary"); self.btnCustomBase.setMinimumWidth(80)
        
        self.customBaseWidget = QtWidgets.QWidget()
        cb_lay = QtWidgets.QHBoxLayout(self.customBaseWidget); cb_lay.setContentsMargins(0,0,0,0)
        cb_lay.addWidget(self.edtCustomBase); cb_lay.addWidget(self.btnCustomBase, 0, QtCore.Qt.AlignRight)
        self.customBaseWidget.hide()
        
        self.cmbBase.currentIndexChanged.connect(self._on_base_changed)
        self.btnCustomBase.clicked.connect(self._browse_custom_base)

        self.spinWorkers = QtWidgets.QSpinBox(); self.spinWorkers.setRange(1, 64); self.spinWorkers.setValue(8); self.spinWorkers.setSingleStep(2)
        self.spinPrefetch = QtWidgets.QSpinBox(); self.spinPrefetch.setRange(1, 512); self.spinPrefetch.setValue(16); self.spinPrefetch.setSingleStep(8)

        r=0
        g.addWidget(QtWidgets.QLabel("输出文件名"), r, 0); g.addLayout(name_lay, r, 1, 1, 3); r+=1
        
        # 修复布局：输入框铺满，浏览按钮靠右且变大
        mask_lay = QtWidgets.QHBoxLayout(); mask_lay.setContentsMargins(0,0,0,0)
        mask_lay.addWidget(self.edtMask); mask_lay.addWidget(self.btnMask, 0, QtCore.Qt.AlignRight)
        g.addWidget(QtWidgets.QLabel("星点检测范围"), r, 0); g.addLayout(mask_lay, r, 1, 1, 3); r+=1
        
        g.addWidget(QtWidgets.QLabel("星点亮度阈值"), r, 0); g.addWidget(self.spinThresh, r, 1)
        g.addWidget(QtWidgets.QLabel("星点最小面积"), r, 2); g.addWidget(self.spinMinArea, r, 3); r+=1
        g.addWidget(QtWidgets.QLabel("星点扩张范围"), r, 0); g.addWidget(self.spinRadius, r, 1)
        g.addWidget(QtWidgets.QLabel("非检测膨胀"), r, 2); g.addWidget(self.spinDilation, r, 3); r+=1
        
        g.addWidget(QtWidgets.QLabel("非星区域底图"), r, 0); g.addWidget(self.cmbBase, r, 1)
        # 自定义底图布局也修复
        g.addWidget(self.customBaseWidget, r, 2, 1, 2); r+=1
        
        g.addWidget(QtWidgets.QLabel("并发线程数"), r, 0); g.addWidget(self.spinWorkers, r, 1)
        g.addWidget(QtWidgets.QLabel("缓存图片数"), r, 2); g.addWidget(self.spinPrefetch, r, 3); r+=1

        layout.addLayout(g)

        btn_box = QtWidgets.QHBoxLayout()
        self.btnTest = QtWidgets.QPushButton("测试星点检测"); self.btnTest.setObjectName("BtnGhost")
        self.btnTest.clicked.connect(self.testRequested.emit)
        self.btnRun = QtWidgets.QPushButton("运行"); self.btnRun.setObjectName("BtnNeon")
        self.btnRun.clicked.connect(lambda: self.runRequested.emit("star_stack"))
        btn_box.addWidget(self.btnTest); btn_box.addWidget(self.btnRun, 1)
        
        layout.addLayout(btn_box)
        layout.addStretch(1)

    def _browse(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择蒙版图片", "", "Images (*.jpg *.png *.bmp)")
        if f: self.edtMask.setText(f)

    def _on_base_changed(self):
        if self.cmbBase.currentText() == "自定义": self.customBaseWidget.show()
        else: self.customBaseWidget.hide()

    def _browse_custom_base(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择底图", "", "Images (*.jpg *.png *.bmp *.tif *.webp)")
        if f: self.edtCustomBase.setText(f)

    def params(self):
        return {"output_name": self.edtName.text().strip() + ".jpg", "mask_path": self.edtMask.text().strip(),
                "thresh_sigma": self.spinThresh.value(), "minarea": self.spinMinArea.value(), "star_radius": self.spinRadius.value(),
                "io_workers": self.spinWorkers.value(), "prefetch": self.spinPrefetch.value(), 
                "base_mode": self.cmbBase.currentText(), "custom_base": self.edtCustomBase.text().strip(),
                "mask_dilation_pct": self.spinDilation.value()}


class MainWindow(QtWidgets.QMainWindow):
    progressSignal = QtCore.pyqtSignal(dict)
    logSignal = QtCore.pyqtSignal(str) # 新增日志信号

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(build_app_icon())
        self.resize(1150, 720)
        self.cancel_token = None; self._thread = None; self._worker = None

        # 挂载深空背景
        self.bgWidget = StarryBackgroundWidget()
        self.setCentralWidget(self.bgWidget)
        root = QtWidgets.QVBoxLayout(self.bgWidget)
        root.setContentsMargins(18, 18, 18, 18)

        split = QtWidgets.QHBoxLayout()
        self.inputPanel = InputPanel()
        split.addWidget(self.inputPanel, 4)
        
        right = QtWidgets.QVBoxLayout()
        self.outputPanel = OutputPanel()
        right.addWidget(self.outputPanel, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("AlgoTabs")
        self.tabGlobal = TabGlobalStack()
        self.tabAfter = TabAfterimageVideo()
        self.tabStar = TabStarStack()
        self.tabs.addTab(self.tabGlobal, "最大值全局堆栈")
        self.tabs.addTab(self.tabAfter, "残影延时视频")
        self.tabs.addTab(self.tabStar, "最大值星点堆栈")
        right.addWidget(self.tabs, 1)

        self.monitor = TaskMonitor()
        right.addWidget(self.monitor, 1)
        split.addLayout(right, 7)
        root.addLayout(split, 1)

        self.tabGlobal.runRequested.connect(self._run_algo)
        self.tabAfter.runRequested.connect(self._run_algo)
        self.tabStar.runRequested.connect(self._run_algo)
        self.tabStar.testRequested.connect(self._test_star_detection)
        self.monitor.cancelClicked.connect(self._cancel)
        self.progressSignal.connect(self._forward_progress)
        self.logSignal.connect(self.monitor.append_log)

    def _set_ui_running(self, running: bool):
        if running:
            self._last_stage = None
            self._stage_t0 = time.time()
            self._overall_t0 = time.time()
        self.inputPanel.set_enabled(not running)
        self.outputPanel.setEnabled(not running)
        self.tabs.setEnabled(not running)
        self.monitor.set_running(running)

    def _cancel(self):
        if self.cancel_token:
            self.cancel_token.cancel()
            self.monitor.append_log("用户请求取消...")

    def _validate_image_sizes(self, img_path: Union[str, Path], mask_path: str, base_path: Union[str, Path]) -> bool:
        """校验图片尺寸一致性"""
        ensure_cv2()
        
        # 读取主图尺寸
        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # QtWidgets.QMessageBox.warning(self, "错误", f"无法读取待处理图片:\n{img_path}")
            # 这里不弹窗了，因为后面流程还会检查
            return True 
        h, w = img.shape[:2]
        
        # 校验掩膜
        if mask_path:
            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mh, mw = mask.shape
                if (mh, mw) != (h, w):
                    msg = f"检测到尺寸不匹配！\n\n主图尺寸: {w}x{h}\n掩膜尺寸: {mw}x{mh}\n\n这可能导致掩膜错位或运行失败。建议调整图片尺寸使其一致。"
                    QtWidgets.QMessageBox.warning(self, "尺寸不匹配警告", msg)
                    # 按照用户要求，只弹窗提醒，不强制 return False？
                    # 用户原话是“弹窗提醒”，通常意味着警告。但尺寸不一致会导致后续 Numpy 索引越界报错。
                    # 所以还是应该 return False 阻止运行，或者询问用户是否继续？
                    # 考虑到后续代码里我做了 mask_display = cv2.resize，其实显示部分不会崩。
                    # 但 detect_stars_sep 里的向量化筛选部分：
                    # valid_coord_mask = ... (iys < MH)
                    # 会自动过滤掉越界的点，或者如果掩膜太小，会导致大量点被误过滤。
                    # 为了安全，这里还是返回 False 阻止运行比较好，强制用户修正。
                    return False

        # 校验底图
        if base_path:
            # 只有自定义底图才需要强校验，若是第一张/最后一张本身就是主图之一，无需校验
            # 但这里 base_path 可能是主图列表里的。
            # 为了避免重复读取大图，可以简单判断路径字符串
            if str(base_path) != str(img_path):
                base = imread_unicode(base_path, cv2.IMREAD_COLOR)
                if base is not None:
                    bh, bw = base.shape[:2]
                    if (bh, bw) != (h, w):
                         msg = f"检测到尺寸不匹配！\n\n主图尺寸: {w}x{h}\n底图尺寸: {bw}x{bh}\n\n合成结果可能会被裁切或报错。"
                         QtWidgets.QMessageBox.warning(self, "尺寸不匹配警告", msg)
                         return False
            
        return True

    def _test_star_detection(self):
        imgs = self.inputPanel.paths()
        p = self.tabStar.params()
        mask_path = p["mask_path"]
        
        test_img_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择一张照片测试星点检测", "", "Images (*.jpg *.png *.bmp *.tif)")
        if not test_img_path: return

        # 校验尺寸
        if not self._validate_image_sizes(test_img_path, mask_path, ""):
            return

        mask_img = None
        if mask_path:
            try: mask_img = load_detection_mask(mask_path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "掩膜错误", f"无法读取检测范围掩膜，请检查路径:\n{e}")
                return
        
        # ...原有逻辑继续...
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            ensure_cv2()
            img = imread_unicode(test_img_path, cv2.IMREAD_COLOR)
            if img is None: raise FileNotFoundError("无法读取照片")
            
            mask_dilation_pct = p.get("mask_dilation_pct", 1.0)
            _, xy_stars = _detect_one_sep_worker((test_img_path, mask_img, p["thresh_sigma"], p["minarea"], mask_dilation_pct))
            
            pts = np.rint(xy_stars).astype(np.int32)
            
            # 1. 验证掩膜对齐情况，绘制半透明遮罩
            if mask_img is not None:
                H, W = img.shape[:2]
                MH, MW = mask_img.shape
                
                # 若尺寸不匹配，简单拉伸用于可视化
                mask_display = mask_img
                if MH != H or MW != W:
                    mask_display = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

                # 生成膨胀后的掩膜用于可视化
                filt_mask_display = mask_display
                if mask_dilation_pct > 0.001:
                    expand_dist = int(min(H, W) * (mask_dilation_pct / 100.0)) 
                    if expand_dist < 1: expand_dist = 1
                    k_size = 2 * expand_dist + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                    filt_mask_display = cv2.dilate(mask_display, kernel, iterations=1)

                # 绘制可视化遮罩
                overlay = img.copy()
                
                # 深蓝色区域：原始掩膜屏蔽区
                base_invalid = mask_display > 127
                overlay[base_invalid] = (255, 100, 50) 
                
                # 淡紫色区域：边缘膨胀产生的额外过滤带 (高亮显示以辅助调节参数)
                if mask_dilation_pct > 0.001:
                    dilated_invalid = filt_mask_display > 127
                    diff_region = dilated_invalid & (~base_invalid)
                    overlay[diff_region] = (250, 150, 200)

                # 叠加显示
                cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

            # 2. 绘制检测到的有效星点 (红色圆圈)
            for x, y in pts: cv2.circle(img, (x, y), 8, (0, 0, 255), 2)
            
            out_dir = self.outputPanel.output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_test = out_dir / "测试星点_标记.jpg"
            imwrite_unicode(out_test, img)
            
            self.monitor.append_log(f"[测试] 在该图检测到 {len(pts)} 个有效星点，导出为: {out_test.name}")
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(out_test)))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "测试失败", f"星点检测测试失败:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _run_algo(self, algo_key: str):
        if self._thread is not None: return
        imgs = self.inputPanel.paths()
        if not imgs: return QtWidgets.QMessageBox.warning(self, "提示", "请先导入照片！")
        
        out_dir = self.outputPanel.output_dir()
        try: out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e: return QtWidgets.QMessageBox.critical(self, "错误", f"无法创建导出文件夹:\n{e}")
        
        try: ensure_cv2()
        except Exception as e: return QtWidgets.QMessageBox.critical(self, "环境缺失", str(e))

        cancel = CancelToken()
        self.cancel_token = cancel
        
        # 传递具体的任务键，用于 set_dual_mode 判断显示的文本
        is_star_stack = (algo_key == "star_stack")
        self.monitor.set_dual_mode(is_star_stack, task_type=algo_key)

        if algo_key == "global_stack":
            p = self.tabGlobal.params()
            out_path = out_dir / p["output_name"]
            cfg = AlgorithmConfig("最大值全局堆栈", ["全局堆栈"])
            def fn(): return composite_max_fullframe(imgs, out_path, p["io_workers"], p["prefetch"], cancel=cancel, progress_cb=self._progress_bridge(cfg), log_cb=self.logSignal.emit)
        
        elif algo_key == "afterimage_video":
            p = self.tabAfter.params()
            out_path = out_dir / p["output_name"]
            cfg = AlgorithmConfig("残影延时视频", ["视频编码"])
            def fn():
                make_star_trail_video_decay_maxhold(imgs, out_path, p["decay"], p["fps"], p["io_workers"], p["prefetch"], p["crf"], p["preset"], cancel=cancel, progress_cb=self._progress_bridge(cfg), log_cb=self.logSignal.emit)
                return str(out_path)
                
        elif algo_key == "star_stack":
            p = self.tabStar.params()
            out_path = out_dir / p["output_name"]
            cfg = AlgorithmConfig("最大值星点堆栈", ["星点检测", "星点区域堆栈"])
            
            mask_path = p["mask_path"]
            mask_img = None
            if mask_path:
                try: mask_img = load_detection_mask(mask_path)
                except Exception as e:
                    return QtWidgets.QMessageBox.warning(self, "掩膜错误", f"无法读取星点检测范围掩膜:\n{e}")
            
            base_mode = p["base_mode"]
            if base_mode == "自定义":
                if not p["custom_base"]: 
                    return QtWidgets.QMessageBox.warning(self, "错误", "选择了自定义底图但未指定路径！")
                base_path = Path(p["custom_base"])
                if not base_path.exists():
                    return QtWidgets.QMessageBox.warning(self, "错误", "指定的底图路径不存在！")
            else:
                base_path = imgs[-1] if "最后一张" in base_mode else imgs[0]

            # 校验尺寸 consistency (主图与掩膜、底图)
            if not self._validate_image_sizes(imgs[0], mask_path, base_path):
                return
            
            def fn():
                # 使用 logSignal.emit 替代直接调用 append_log，解决跨线程/多进程回调时的 QObject::connect 类型注册问题
                res = detect_stars_sep(imgs, mask_img, p["io_workers"], p["thresh_sigma"], p["minarea"], p["prefetch"], p["mask_dilation_pct"], cancel=cancel, progress_cb=self._progress_bridge(cfg), log_cb=self.logSignal.emit)
                return composite_max_stars(res, imgs, base_path, out_path, p["star_radius"], p["io_workers"], p["prefetch"], cancel=cancel, progress_cb=self._progress_bridge(cfg), log_cb=self.logSignal.emit)

        self._thread = QtCore.QThread(self)
        self._worker = TaskWorker(cfg.name, fn, cfg, cancel)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.signals.log.connect(self.monitor.append_log)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.error.connect(self._on_error)
        self._worker.signals.canceled.connect(self._on_canceled)
        for sig in (self._worker.signals.finished, self._worker.signals.error, self._worker.signals.canceled):
            sig.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._set_ui_running(True)
        self._thread.start()

    def _progress_bridge(self, cfg: AlgorithmConfig):
        return lambda s, d, t, e: self.progressSignal.emit({"stage": s, "done": d, "total": t, "extra": e, "cfg": cfg})

    @QtCore.pyqtSlot(dict)
    def _forward_progress(self, payload: dict):
        if not self._worker: return
        cfg = payload["cfg"]
        if self._worker.cfg.name != cfg.name: return 
        
        stage_name = payload["stage"]
        # 阶段计时逻辑：如果阶段变更，更新基准时间
        if getattr(self, "_last_stage", None) != stage_name:
            self._last_stage = stage_name
            self._stage_t0 = time.time()
            if not hasattr(self, "_overall_t0"): self._overall_t0 = time.time()

        stage_frac = min(1.0, max(0.0, payload["done"] / max(1, payload["total"])))
        stage_el = time.time() - (getattr(self, "_stage_t0", 0) or time.time())
        stage_eta = (stage_el / stage_frac - stage_el) if stage_frac > 0.001 else float("inf")
        
        idx = cfg.stages.index(stage_name) if stage_name in cfg.stages else 0
        ov_frac = (idx + stage_frac) / max(1, len(cfg.stages))
        
        # 为了兼容单进度模式显示总体剩余时间，也计算一下总体 ETA
        ov_el = time.time() - (getattr(self, "_overall_t0", 0) or time.time())
        ov_eta = (ov_el / ov_frac - ov_el) if ov_frac > 0.001 else float("inf")
        
        # 如果是双进度条模式，优先展示当前阶段的计时信息；如果是单进度条，展示总体
        is_dual = len(cfg.stages) > 1
        disp_el = stage_el if is_dual else ov_el
        disp_eta = stage_eta if is_dual else ov_eta

        self.monitor.update_progress({
            "task": cfg.name, "stage": stage_name, "done": payload["done"], "total": payload["total"],
            "stage_pct": int(stage_frac * 100), "overall_pct": int(ov_frac * 100), 
            "stage_elapsed": disp_el, "stage_eta": disp_eta, "extra": payload["extra"],
            "is_dual": is_dual
        })

    def _on_finished(self, payload: dict):
        self._set_ui_running(False)
        out_path = payload.get('output_path')
        self.monitor.append_log(f"[完成] 路径: {out_path}")
        
        def show_msg():
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("处理完毕")
            msg.setText(f"文件已成功保存:\n{out_path}")
            msg.setIcon(QtWidgets.QMessageBox.Information)
            
            # 添加按钮: 打开文件、打开文件夹、确定
            btn_open_file = msg.addButton("打开文件", QtWidgets.QMessageBox.ActionRole)
            btn_open_dir = msg.addButton("打开文件夹", QtWidgets.QMessageBox.ActionRole)
            btn_ok = msg.addButton("确定", QtWidgets.QMessageBox.AcceptRole)
            
            msg.exec_()
            
            clicked = msg.clickedButton()
            if clicked == btn_open_file:
                try:
                    os.startfile(out_path)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "错误", f"无法打开文件:\n{e}")
            elif clicked == btn_open_dir:
                try:
                    folder = os.path.dirname(out_path)
                    os.startfile(folder)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "错误", f"无法打开文件夹:\n{e}")

        QtCore.QTimer.singleShot(100, show_msg)

    def _on_canceled(self, name: str):
        self._set_ui_running(False)
        self.monitor.append_log("[取消] 用户手动终止了任务")
        self.monitor.chipETA.setText("剩余: 已取消")

    def _on_error(self, err: str):
        self._set_ui_running(False)
        self.monitor.append_log(f"[错误]\n{err}")
        def show_err():
            QtWidgets.QMessageBox.critical(self, "异常中断", f"处理过程中发生错误:\n{err}")
        QtCore.QTimer.singleShot(100, show_err)

    def _cleanup_thread(self):
        if self._worker: self._worker.deleteLater()
        if self._thread: self._thread.deleteLater()
        self._worker = self._thread = self.cancel_token = None


STYLESHEET = r"""
* { font-family: "Microsoft YaHei"; font-size: 10.5pt; color: #E6EEF7; outline: none; } /* outline: none 去除选中时的虚线框/红框 */

/* 毛玻璃半透明面板 */
#GlassPanel, #AlgoTabs::pane, #FileList, #LogView { 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    border-radius: 12px; 
    background: rgba(20, 25, 40, 0.55); 
}

#AlgoTabs::pane { top: -1px; }
#AlgoTabs QTabBar::tab { 
    background: rgba(20, 25, 40, 0.4); 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    padding: 8px 16px; margin-right: 4px; 
    border-top-left-radius: 8px; border-top-right-radius: 8px; 
    color: #9FB2D0; 
}
#AlgoTabs QTabBar::tab:selected { 
    color: #FFF; 
    border: 1px solid rgba(0, 229, 255, 0.5); 
    background: rgba(15, 23, 42, 0.85); 
    font-weight: bold; 
}

/* 输入框组合 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { 
    background: rgba(10, 15, 30, 0.6); 
    border: 1px solid rgba(255, 255, 255, 0.15); 
    border-radius: 6px; padding: 6px 8px; 
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid rgba(0, 229, 255, 0.5); /* 选中时显示科技蓝边框，覆盖默认红色/橙色焦点框 */
    background: rgba(10, 15, 30, 0.8);
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid rgba(255, 255, 255, 0.1);
}
QComboBox::down-arrow {
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #E6EEF7;
    image: none;
    margin-right: 2px; /* 微调位置 */
}
QComboBox QAbstractItemView {
    background: #141928;
    color: #E6EEF7;
    selection-background-color: rgba(0, 229, 255, 80);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

/* 修复 QMessageBox 白底白字问题 */
QMessageBox {
    background-color: #141928;
}
QMessageBox QLabel {
    color: #E6EEF7;
    background-color: transparent;
}
QMessageBox QPushButton {
    min-width: 80px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #FFF;
}
QMessageBox QPushButton:hover {
    background: rgba(255, 255, 255, 0.2);
}
QMessageBox QPushButton:pressed {
    background: rgba(0, 0, 0, 0.3);
}

/* 修复 QSpinBox 加减按钮按压反馈 */
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 20px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 255, 255, 0.05), stop:1 rgba(255, 255, 255, 0.15));
    border-left: 1px solid rgba(255, 255, 255, 0.1);
    subcontrol-origin: border;
}
QSpinBox::up-button { subcontrol-position: top right; }
QSpinBox::down-button { subcontrol-position: bottom right; }

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: rgba(255, 255, 255, 0.25);
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: rgba(0, 0, 0, 0.3);
    padding-top: 1px;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 8px; height: 8px;
    border-radius: 4px;
    background: #E6EEF7;
    image: none;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 8px; height: 8px;
    border-radius: 4px;
    background: #E6EEF7;
    image: none;
}

QPushButton { 
    border-radius: 6px; padding: 7px 12px; 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    background: rgba(255, 255, 255, 0.1); 
}
QPushButton:hover { background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(0, 229, 255, 120); }
QPushButton:pressed { 
    background: rgba(0, 0, 0, 0.3); 
    padding-top: 8px; padding-bottom: 6px; /* 按压下陷效果 */
    border: 1px solid rgba(0, 229, 255, 80); 
}

/* 高级发光主按钮 */
#BtnNeon { 
    font-weight: bold; font-size: 11pt; color: #FFF;
    border: 1px solid rgba(0, 229, 255, 150); 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 229, 255, 120), stop:1 rgba(0, 102, 255, 120)); 
}
#BtnNeon:hover { border: 1px solid rgba(0, 229, 255, 255); background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 229, 255, 160), stop:1 rgba(0, 102, 255, 160)); }
#BtnNeon:pressed { background: rgba(0, 102, 255, 100); padding-top: 8px; padding-bottom: 6px; }

/* 幽灵测试按钮 */
#BtnGhost { background: rgba(255, 255, 255, 0.05); color: #A0B0C0; }
#BtnGhost:hover { background: rgba(255, 255, 255, 0.1); color: #FFF; }
#BtnGhost:pressed { background: rgba(0, 0, 0, 0.2); padding-top: 8px; padding-bottom: 6px; }

#BtnDanger { border: 1px solid rgba(255, 77, 99, 120); background: rgba(255, 77, 99, 0.1); color: #FF8899; font-weight: bold; }
#BtnDanger:hover { background: rgba(255, 77, 99, 0.2); border: 1px solid rgba(255, 77, 99, 200); }
#BtnDanger:pressed { background: rgba(255, 77, 99, 0.05); padding-top: 8px; padding-bottom: 6px; }

/* 进度条 */
QProgressBar { border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px; text-align: center; background: rgba(10, 15, 30, 0.6); height: 14px; }
QProgressBar::chunk { border-radius: 6px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 229, 255, 180), stop:1 rgba(0, 102, 255, 180)); }

#HintText { color: #8A9BB8; font-size: 9.5pt; }
#MonitorTitle { font-size: 12pt; font-weight: bold; }
#MonitorSubTitle { color: #00E5FF; }
"""

def main():
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(STYLESHEET)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()