import cv2
import numpy as np
import random,os
from collections import defaultdict
def unpack_name(image_name):
    """
    从文件名中提取病人的信息。
    文件名格式: Patient's ID_sex_gestational age (GA)_birth weight (BW)_postconceptual age (PA)_diagnosis code (DG)_plus-form (PF)_device(D)_serie (S)_image number.jpg
    """
    parts = image_name.split('_')
    patient_info = {
        'patient_id': parts[0],
        'sex': parts[1],
        'gestational_age': int(parts[2][2:]),  # GA后面的数字
        'birth_weight': int(parts[3][2:]),  # BW后面的数字
        'postconceptual_age': int(parts[4][2:]),  # PA后面的数字
        'diagnosis_code': int(parts[5][2:]),  # DG后面的数字
        'plus_form': int(parts[6][2:]),  # PF后面的数字
        'device': int(parts[7][2]),  # 设备型号
        'series': int(parts[8][1:]),  # S后面的数字
        'image_number': int(parts[9].split('.')[0])  # 图片编号
    }
    return patient_info

def split_data(data_path, num_splits=5):
    """
    Splits the data into training and testing sets based on the specified rules.
    """
    image_dir = os.path.join(data_path, 'images')
    image_name_list = sorted(os.listdir(image_dir))

    # Collect all image data
    images = []
    for image_name in image_name_list:
        data_info = unpack_name(image_name)
        image_path = os.path.join(image_dir, image_name)
        data_info.update({
            'image_path': image_path,
            'image_name': image_name
        })
        images.append(data_info)

    splits = []
    for split_num in range(num_splits):
        # Shuffle images to ensure randomness in each split
        random.shuffle(images)

        # Group images by diagnosis_code
        diagnosis_groups = defaultdict(list)
        for img in images:
            diagnosis_groups[img['diagnosis_code']].append(img)

        training_set = []
        testing_set = []

        for diagnosis_code, group_images in diagnosis_groups.items():
            # Get unique patient IDs in this diagnosis code
            patient_ids = list(set(img['patient_id'] for img in group_images))

            if len(patient_ids) == 1:
                # Only one patient in this class, split by series
                series_groups = defaultdict(list)
                for img in group_images:
                    series_groups[img['series']].append(img)
                series_list = list(series_groups.keys())
                random.shuffle(series_list)
                num_series = len(series_list)
                split_point = max(1, int(num_series * 0.8))  # Ensure at least one series in test set
                training_series = series_list[:split_point]
                testing_series = series_list[split_point:]
                for series in training_series:
                    training_set.extend(series_groups[series])
                for series in testing_series:
                    testing_set.extend(series_groups[series])
            else:
                # Multiple patients in this class, split by patient IDs
                random.shuffle(patient_ids)
                num_patients = len(patient_ids)
                split_point = max(1, int(num_patients * 0.8))  # Ensure at least one patient in test set
                training_patient_ids = patient_ids[:split_point]
                testing_patient_ids = patient_ids[split_point:]
                for img in group_images:
                    if img['patient_id'] in training_patient_ids:
                        training_set.append(img)
                    else:
                        testing_set.append(img)

        # Ensure overall split is approximately 4:1
        total_images = len(training_set) + len(testing_set)
        training_ratio = len(training_set) / total_images
        testing_ratio = len(testing_set) / total_images
        print(f"Split {split_num+1}: Training images = {len(training_set)} ({training_ratio:.2f}), "
              f"Testing images = {len(testing_set)} ({testing_ratio:.2f})")

        splits.append({
            'training_set': training_set,
            'testing_set': testing_set
        })

    return splits
def sift_feature_matching(src, tar, bbox, threshold=0.75):
    """
    使用SIFT从src中的给定bbox区域提取特征，在tar中寻找最相似的区域

    参数:
    src (numpy.ndarray): 源图像（从中提取指定bbox区域的特征）
    tar (numpy.ndarray): 目标图像（在其中寻找匹配区域）
    bbox (tuple): 包含 (xmin, xmax, ymin, ymax) 表示src中指定的区域
    threshold (float): 特征匹配的过滤阈值（默认值为0.75）

    返回:
    (tuple) 返回在tar图像中匹配区域的bbox (xmin, xmax, ymin, ymax) 
    或 False（如果没有找到匹配区域）
    """

    # 初始化 SIFT 特征提取器
    sift = cv2.SIFT_create()

    # 从源图像中提取指定 bbox 区域
    xmin, xmax, ymin, ymax = bbox
    src_roi = src[ymin:ymax, xmin:xmax]

    # 在源区域和目标图像中检测SIFT关键点和描述符
    kp_src, des_src = sift.detectAndCompute(src_roi, None)
    kp_tar, des_tar = sift.detectAndCompute(tar, None)

    # 使用暴力匹配器匹配描述符
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_src, des_tar, k=2)

    # 进行最近邻匹配，过滤掉低质量匹配
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    # 如果找到的匹配点数足够多，进行定位
    if len(good_matches) > 4:
        # 获取源图像和目标图像中的匹配关键点的坐标
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        tar_pts = np.float32([kp_tar[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # 计算单应矩阵，使用RANSAC去除错误匹配
        H, mask = cv2.findHomography(src_pts, tar_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # 使用单应矩阵将源图像的区域映射到目标图像中，获得匹配区域
            h, w = src_roi.shape[:2]
            bbox_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            bbox_tar = cv2.perspectiveTransform(bbox_src, H)

            # 计算匹配区域的最小外接矩形
            xmin, ymin = np.int32(bbox_tar.min(axis=0).ravel())
            xmax, ymax = np.int32(bbox_tar.max(axis=0).ravel())

            return (xmin, xmax, ymin, ymax)
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    import json
    image_name="001_F_GA41_BW2905_PA44_DG11_PF0_D1_S01_2.jpg"
    output_json=unpack_name(image_name)
    print(json.dumps(output_json, indent=4))