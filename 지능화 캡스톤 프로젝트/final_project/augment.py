import cv2
import numpy as np
import os
import albumentations as A
import glob
import random


def load_labels(label_path):
    labels = []
    try:
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                labels.append([float(x) for x in line.strip().split()])
    except Exception as e:
        print(f"Error loading labels from {label_path}: {e}")
    return labels


def save_labels(label_path, labels):
    try:
        with open(label_path, 'w') as file:
            for label in labels:
                file.write(' '.join([str(x) for x in label]) + '\n')
    except Exception as e:
        print(f"Error saving labels to {label_path}: {e}")


def copy_paste(image, labels, paste_image, paste_labels):
    for label in paste_labels:
        class_id, x_center, y_center, width, height = label
        x_center = int(x_center * paste_image.shape[1])
        y_center = int(y_center * paste_image.shape[0])
        width = int(width * paste_image.shape[1])
        height = int(height * paste_image.shape[0])

        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2

        paste_object = paste_image[y1:y2, x1:x2]
        x_offset = np.random.randint(0, image.shape[1] - width)
        y_offset = np.random.randint(0, image.shape[0] - height)

        image[y_offset:y_offset + height, x_offset:x_offset + width] = paste_object

        new_x_center = (x_offset + width // 2) / image.shape[1]
        new_y_center = (y_offset + height // 2) / image.shape[0]
        new_width = width / image.shape[1]
        new_height = height / image.shape[0]

        labels.append([class_id, new_x_center, new_y_center, new_width, new_height])
    return image, labels


def mixup(image1, labels1, image2, labels2, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    mixed_image = lam * image1 + (1 - lam) * image2
    mixed_labels = labels1 + labels2  # 라벨을 합칩니다.
    return mixed_image.astype(np.uint8), mixed_labels


def clip_bbox(bbox):
    class_id, x_center, y_center, width, height = bbox
    x_min = max(x_center - width / 2, 0)
    y_min = max(y_center - height / 2, 0)
    x_max = min(x_center + width / 2, 1)
    y_max = min(y_center + height / 2, 1)
    if x_max <= x_min or y_max <= y_min:
        return None
    return [class_id, x_min, y_min, x_max, y_max]


def convert_bbox_to_yolo(bbox, rows, cols):
    class_id, x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / cols
    y_center = (y_min + y_max) / 2 / rows
    width = (x_max - x_min) / cols
    height = (y_max - y_min) / rows
    return [class_id, x_center, y_center, width, height]


def augment_data(image_folder, label_folder, output_image_folder, output_label_folder, num_augmentations=5):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    video_folders = glob.glob(os.path.join(image_folder, "*"))
    if not video_folders:
        print(f"No video folders found in folder: {image_folder}")
        return

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.CLAHE(p=0.5),
        A.RandomGamma(p=0.2),
        A.ToGray(p=0.2),
        A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=0.2)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    for video_folder in video_folders:
        print(f"Processing video folder: {video_folder}")
        image_paths = glob.glob(os.path.join(video_folder, "*.jpg"))
        if not image_paths:
            print(f"No images found in video folder: {video_folder}")
            continue

        random.shuffle(image_paths)

        for image_path in image_paths:
            print(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            h, w, _ = image.shape
            video_folder_name = os.path.basename(video_folder)
            label_path = os.path.join(label_folder, video_folder_name,
                                      os.path.basename(image_path).replace(".jpg", ".txt"))
            print(f"Loading labels from: {label_path}")
            labels = load_labels(label_path)

            if not labels:
                print(f"No labels found for image: {image_path}")
                continue

            # YOLO 형식의 바운딩 박스를 Albumentations가 인식하는 Pascal VOC 형식으로 변환
            pascal_labels = [
                clip_bbox([label[0], label[1] * w, label[2] * h, (label[1] + label[3]) * w, (label[2] + label[4]) * h])
                for label in labels]
            pascal_labels = [label for label in pascal_labels if label is not None]
            class_labels = [label[0] for label in pascal_labels]

            for i in range(num_augmentations):
                # 기본 증강 기법 적용
                augmented = transform(image=image, bboxes=pascal_labels, class_labels=class_labels)
                augmented_image = augmented['image']
                augmented_labels = augmented['bboxes']

                # Copy & Paste 적용
                paste_image_path = random.choice(image_paths)
                paste_image = cv2.imread(paste_image_path)
                if paste_image is None:
                    print(f"Failed to load paste image: {paste_image_path}")
                    continue

                paste_label_path = os.path.join(label_folder, video_folder_name,
                                                os.path.basename(paste_image_path).replace(".jpg", ".txt"))
                paste_labels = load_labels(paste_label_path)
                pascal_paste_labels = [clip_bbox(
                    [label[0], label[1] * w, label[2] * h, (label[1] + label[3]) * w, (label[2] + label[4]) * h]) for
                                       label in paste_labels]
                pascal_paste_labels = [label for label in pascal_paste_labels if label is not None]

                augmented_image, augmented_labels = copy_paste(augmented_image, augmented_labels, paste_image,
                                                               pascal_paste_labels)

                # Mix-up 적용
                mixup_image_path = random.choice(image_paths)
                mixup_image = cv2.imread(mixup_image_path)
                if mixup_image is None:
                    print(f"Failed to load mixup image: {mixup_image_path}")
                    continue

                mixup_label_path = os.path.join(label_folder, video_folder_name,
                                                os.path.basename(mixup_image_path).replace(".jpg", ".txt"))
                mixup_labels = load_labels(mixup_label_path)
                pascal_mixup_labels = [clip_bbox(
                    [label[0], label[1] * w, label[2] * h, (label[1] + label[3]) * w, (label[2] + label[4]) * h]) for
                                       label in mixup_labels]
                pascal_mixup_labels = [label for label in pascal_mixup_labels if label is not None]

                augmented_image, augmented_labels = mixup(augmented_image, augmented_labels, mixup_image,
                                                          pascal_mixup_labels)

                # 바운딩 박스를 YOLO 형식으로 다시 변환
                yolo_labels = [convert_bbox_to_yolo(bbox, h, w) for bbox in augmented_labels]

                # 증강된 이미지와 라벨 저장
                video_output_folder = os.path.join(output_image_folder, video_folder_name)
                if not os.path.exists(video_output_folder):
                    os.makedirs(video_output_folder)

                augmented_image_path = os.path.join(video_output_folder,
                                                    os.path.basename(image_path).replace(".jpg", f"_aug_{i}.jpg"))
                augmented_label_path = os.path.join(output_label_folder, video_folder_name,
                                                    os.path.basename(label_path).replace(".txt", f"_aug_{i}.txt"))

                if not os.path.exists(os.path.dirname(augmented_label_path)):
                    os.makedirs(os.path.dirname(augmented_label_path))

                cv2.imwrite(augmented_image_path, augmented_image)
                save_labels(augmented_label_path, yolo_labels)

                print(f"Saved augmented image: {augmented_image_path}")
                print(f"Saved augmented label: {augmented_label_path}")


# 이미지 폴더와 라벨 폴더 경로 설정
image_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\frame\\frame_Onboard"
label_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\Label\\label_Onboard"

# 증강된 데이터를 저장할 폴더 설정
output_image_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\augmented_images"
output_label_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\augmented_labels"

# 데이터 증강 수행
augment_data(image_folder, label_folder, output_image_folder, output_label_folder, num_augmentations=5)
