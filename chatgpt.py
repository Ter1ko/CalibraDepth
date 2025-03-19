import cv2
import numpy as np
import os
from PIL import Image, ExifTags

# ===== 用户可调参数 =====
# 标定板参数
chessboard_size = (9, 6)        # 内部角点数 (列数-1, 行数-1)
square_size_m = 0.03          # 方格实际边长 (单位：米)（此为A4纸打印所提供棋盘格数值，请根据情况自行修改）

# 相机传感器参数（用户可自行搜索确认后修改此参数）
sensor_size_mm = ( , )     # 传感器物理尺寸 (宽×高，单位：毫米)
focal_length_mm =           # 默认物理焦距 (单位：毫米)，若EXIF中有焦距信息，则会优先使用EXIF

# 图片参数
image_extension = ".jpg"        # 图片格式 (如 .jpg, .png)
# ==========================

def get_exif_data(image_path):
    """利用PIL提取图片EXIF数据"""
    try:
        img = Image.open(image_path)
        exif_data = {}
        if hasattr(img, '_getexif'):
            info = img._getexif()
            if info:
                for tag, value in info.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data[decoded] = value
        return exif_data
    except Exception as e:
        print(f"读取EXIF数据时出错: {e}")
        return {}

def get_camera_params_from_exif(image_path):
    """
    从EXIF中提取焦距信息，如果EXIF中存在焦距，则使用EXIF中的焦距（单位：毫米）。
    否则，返回默认焦距；传感器尺寸由用户自行设置，不从EXIF中读取。
    """
    exif = get_exif_data(image_path)
    focal = focal_length_mm  # 默认焦距
    if 'FocalLength' in exif:
        focal_value = exif['FocalLength']
        if isinstance(focal_value, tuple):
            focal = focal_value[0] / focal_value[1]
        else:
            focal = focal_value
        print(f"EXIF检测到焦距: {focal} mm")
    else:
        print("EXIF中未找到焦距，采用默认焦距。")
    return focal, sensor_size_mm

def main():
    # 自动获取程序所在目录，并定位图片文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "images")
    
    if not os.path.exists(image_dir):
        print(f"错误：图片文件夹不存在 {image_dir}")
        return
    
    # 自动读取所有指定格式图片
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
              if f.lower().endswith(image_extension)]
    if not images:
        print(f"错误：图片文件夹中未找到 {image_extension} 文件")
        return

    # 读取第一张图片获取EXIF参数和图像分辨率
    focal_mm, sensor_mm = get_camera_params_from_exif(images[0])
    sample_img = cv2.imread(images[0])
    if sample_img is None:
        print("错误：无法读取第一张图片")
        return
    image_height, image_width = sample_img.shape[:2]
    print(f"图片分辨率: {image_width} x {image_height} 像素")

    # 生成棋盘格在世界坐标系中的3D点 (假设棋盘在 z=0 平面)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size_m

    objpoints, imgpoints = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 遍历每张图片进行角点检测
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"读取失败: {os.path.basename(fname)}")
            continue
        
        # 为加速处理，先将高分辨率图像缩小后检测角点
        scale = 0.25
        small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        gray_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(
            gray_small, chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        )
        if not ret:
            # 尝试交换棋盘尺寸顺序
            ret, corners = cv2.findChessboardCorners(
                gray_small, chessboard_size[::-1],
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
            )
        
        if ret:
            # 将检测到的角点映射回原图尺度，并使用cornerSubPix进行精细化
            corners_original = corners / scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners_refined = cv2.cornerSubPix(gray, corners_original.astype(np.float32), (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print(f"成功检测到角点: {os.path.basename(fname)}")
        else:
            print(f"未能检测到角点: {os.path.basename(fname)}")

    if len(objpoints) < 5:
        print(f"错误：有效标定图像不足 ({len(objpoints)}张)，至少需要5张！")
        return

    # 进行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (image_width, image_height), None, None
    )

    print("\n=== 相机参数 ===")
    print(f"标定焦距 fx: {mtx[0,0]:.1f}px  fy: {mtx[1,1]:.1f}px")
    
    # 根据自动读取的图像宽度和用户设置的传感器参数计算理论焦距
    fx_theoretical = (image_width / sensor_mm[0]) * focal_mm
    print(f"理论焦距: {fx_theoretical:.1f}px (基于 {focal_mm}mm 物理焦距)")

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\n重投影误差: {mean_error/len(objpoints):.3f} 像素")

    # 输出各图像相机到棋盘格的距离
    print("\n=== 相机到棋盘格距离 ===")
    distances = []
    for i, tvec in enumerate(tvecs):
        distance = np.linalg.norm(tvec)
        distances.append(distance)
        print(f"图像 {i+1}: {distance*100:.1f}cm")
        
    # 计算加权中位数：权重为重投影误差的倒数（防止除零）
    errors = []
    for i in range(len(distances)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(1 / (error + 1e-6))
    
    sorted_indices = np.argsort(distances)
    sorted_distances = np.array(distances)[sorted_indices]
    sorted_weights = np.array(errors)[sorted_indices]
    cum_weights = np.cumsum(sorted_weights)
    median_idx = np.argmax(cum_weights >= 0.5 * cum_weights[-1])
    final_distance = sorted_distances[median_idx]

    print("\n=== 最终聚合距离 ===")
    print(f"有效测量样本数: {len(distances)}")
    print(f"加权中位数: {final_distance*100:.1f}cm")
    print(f"简单平均值: {np.mean(distances)*100:.1f}cm")
    print(f"标准差: {np.std(distances)*100:.1f}cm")

if __name__ == "__main__":
    main()
