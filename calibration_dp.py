import cv2
import numpy as np
import os

# ============== 用户可调参数 ==============
# 标定板参数
chessboard_size = (9, 6)        # 内部角点数 (列数-1, 行数-1)
square_size_m = 0.03          # 方格实际边长 (单位：米)（此为A4纸打印所提供棋盘格数值，请根据情况自行修改）

# 相机传感器参数
sensor_size_mm = ( , )     # 传感器物理尺寸 (宽×高，毫米)
focal_length_mm =           # 已知焦距 (毫米)

# 图像参数
image_extension = ".jpg"        # 图片格式 (如 .jpg, .png)
# ========================================

def main():
    # 自动获取程序所在路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "images")
    
    # 检查图片文件夹是否存在
    if not os.path.exists(image_dir):
        print(f"错误：图片文件夹不存在 {image_dir}")
        return
    
    # 自动读取所有图片
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
              if f.lower().endswith(image_extension)]
    if not images:
        print(f"错误：图片文件夹中未找到 {image_extension} 文件")
        return

    # 生成3D世界坐标点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size_m

    objpoints, imgpoints = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 处理所有图片
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"读取失败: {os.path.basename(fname)}")
            continue
        
        # 高分辨率优化：缩小检测
        scale = 0.25
        small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        gray_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        # 检测角点
        ret, corners = cv2.findChessboardCorners(gray_small, chessboard_size, None)
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray_small, chessboard_size[::-1], None)
        
        if ret:
            # 映射到原图并精细化
            corners_original = corners / scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners_refined = cv2.cornerSubPix(gray, corners_original.astype(np.float32), (25,25), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print(f"成功: {os.path.basename(fname)}")
        else:
            print(f"失败: {os.path.basename(fname)}")

    # 检查有效数据
    if len(objpoints) < 5:
        print(f"错误：有效标定图像不足 ({len(objpoints)}张)，至少需要5张！")
        return

    # 标定相机
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None
    )

    # ============== 输出结果 ==============
    print("\n=== 相机参数 ===")
    print(f"标定焦距 fx: {mtx[0,0]:.1f}px  fy: {mtx[1,1]:.1f}px")
    
    # 计算理论焦距
    image_width_px = img.shape[1]
    sensor_width_mm = sensor_size_mm[0]
    fx_theoretical = (image_width_px / sensor_width_mm) * focal_length_mm
    print(f"理论焦距: {fx_theoretical:.1f}px (基于{focal_length_mm}mm物理焦距)")

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\n重投影误差: {mean_error/len(objpoints):.3f} 像素")

    # 输出所有图像的距离
    print("\n=== 相机到棋盘格距离 ===")
    for i, tvec in enumerate(tvecs):
        distance = np.linalg.norm(tvec)
        print(f"图像 {i+1}: {distance*100:.1f}cm")
        
    # 提取所有距离值（单位：米）
    distances = [np.linalg.norm(tvec) for tvec in tvecs]

    # 计算加权中位数（权重=重投影误差的倒数）
    errors = []
    for i in range(len(distances)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(1 / (error + 1e-6))  # 避免除零

    # 按距离排序并计算加权中位数
    sorted_indices = np.argsort(distances)
    sorted_distances = np.array(distances)[sorted_indices]
    sorted_weights = np.array(errors)[sorted_indices]
    cum_weights = np.cumsum(sorted_weights)
    median_idx = np.argmax(cum_weights >= 0.5 * cum_weights[-1])
    final_distance = sorted_distances[median_idx]

    # 输出统计结果
    print("\n=== 最终聚合距离 ===")
    print(f"有效测量样本数: {len(distances)}")
    print(f"加权中位数: {final_distance*100:.1f}cm")
    print(f"简单平均值: {np.mean(distances)*100:.1f}cm")
    print(f"标准差: {np.std(distances)*100:.1f}cm")

if __name__ == "__main__":
    main()
