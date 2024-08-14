import scipy.io
import os
import numpy as np

# .mat 파일에서 라벨 정보를 추출하는 함수
def mat_to_yolo_labels(mat_file, output_folder):
    mat_data = scipy.io.loadmat(mat_file)
    struct_xml = mat_data['structXML']

    class_mapping = {
        'Buoy': 0,
        'Vessel/ship': 1,
        'Boat': 2,
        'Sail Boat': 3,
        'Kayak': 4,
        'Ferry': 5,
        'Other': 6
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, element in enumerate(struct_xml[0]):
        for j in range(len(element[4])):  # 각 객체에 대해 반복
            if len(element[4][j]) == 0:  # 빈 배열 무시
                continue

            class_name = element[4][j][0]
            class_name = class_name.item()  # class_name을 문자열로 변환
            if class_name not in class_mapping:
                print(f"Unknown class: {class_name}")
                continue

            class_id = class_mapping[class_name]

            bbox_array = element[6][j]
            if len(bbox_array) == 0:
                continue

            x_center = bbox_array[0] / 1920  # 너비로 정규화 (예: 1920)
            y_center = bbox_array[1] / 1080  # 높이로 정규화 (예: 1080)
            width = bbox_array[2] / 1920
            height = bbox_array[3] / 1080

            label_file = os.path.join(output_folder, f"frame_{i:06d}.txt")
            with open(label_file, 'a') as f:  # append 모드로 파일 열기
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# .mat 파일이 있는 폴더 경로
mat_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\VIS_Onboard\\ObjectGT"
output_base_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\Label\\label_Onboard"

# 모든 .mat 파일 목록 가져오기
mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]

# 각 .mat 파일에 대해 라벨 추출 및 저장 작업 수행
for mat_file in mat_files:
    mat_path = os.path.join(mat_folder, mat_file)
    mat_output_folder = os.path.join(output_base_folder, os.path.splitext(mat_file)[0])  # 각 .mat 파일 이름의 폴더 생성

    # .mat 파일에서 라벨 추출
    mat_to_yolo_labels(mat_path, mat_output_folder)