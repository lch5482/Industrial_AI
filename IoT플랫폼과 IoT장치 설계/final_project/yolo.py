import torch
import os
from ultralytics import YOLO

def main():
    # CUDA 사용 가능 여부 확인
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 현재 CUDA 장치 확인
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    # 프로젝트 디렉토리 경로 설정
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(project_dir, 'data.yaml')

    # 하이퍼파라미터 설정
    lr = 0.01
    momentum = 0.937
    weight_decay = 0.0005
    img_size = 640
    batch_size = 16
    epochs = 30

    # YOLOv8 모델 로드
    model = YOLO('yolov8m.pt')  # YOLOv8 small 모델 사용

    # 학습 설정
    overrides = {
        'lr0': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'imgsz': img_size,
        'batch': batch_size
    }

    try:
        # 모델 학습
        model.train(data=data_yaml_path, epochs=epochs, **overrides, device=0)
    except Exception as e:
        print(f"Training failed with exception: {e}")
        raise

    try:
        # 모델 평가 및 검증 성능 반환
        results = model.val(data=data_yaml_path, imgsz=img_size, device=0)
    except Exception as e:
        print(f"Validation failed with exception: {e}")
        raise


if __name__ == '__main__':
    main()