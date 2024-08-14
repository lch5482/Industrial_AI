import cv2
import os


def video_to_images(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

# 비디오 파일이 있는 폴더 경로
video_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\VIS_Onboard\\Videos"

# 모든 비디오 파일 목록 가져오기
video_files = [f for f in os.listdir(video_folder) if f.endswith('.avi')]

# 각 비디오 파일에 대해 변환 작업 수행
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    output_folder = os.path.join("C:\\Users\\PC\\Downloads\\smd_plus\\frame_Onboard", os.path.splitext(video_file)[0])  # 각 비디오 파일 이름의 폴더 생성
    video_to_images(video_path, output_folder)import cv2
import os


def video_to_images(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

# 비디오 파일이 있는 폴더 경로
video_folder = "C:\\Users\\PC\\Downloads\\smd_plus\\VIS_Onboard\\Videos"

# 모든 비디오 파일 목록 가져오기
video_files = [f for f in os.listdir(video_folder) if f.endswith('.avi')]

# 각 비디오 파일에 대해 변환 작업 수행
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    output_folder = os.path.join("C:\\Users\\PC\\Downloads\\smd_plus\\frame_Onboard", os.path.splitext(video_file)[0])  # 각 비디오 파일 이름의 폴더 생성
    video_to_images(video_path, output_folder)