# cv-study

Computer Vision 학습을 위한 개인 레포지토리

## Topics

- **Object Detection**: YOLO를 활용한 객체 탐지
- **VLM (Vision Language Model)**: CLIP, LLaVA 등 멀티모달 모델

## Setup

```bash
# 의존성 설치
uv sync

# Jupyter 실행
uv run jupyter notebook
```

## Structure

```
cv-study/
├── notebooks/
│   ├── 01_object_detection/   # YOLO 기반 객체 탐지
│   └── 02_vlm/                # Vision-Language 모델
├── src/cv_pkg/                # 공통 유틸리티
│   └── utils/
│       ├── visualization.py   # 시각화 (draw_bbox, show_images)
│       └── data.py            # 데이터 로딩 (load_image, preprocess_image)
├── data/                      # 데이터셋 (gitignore)
├── models/                    # 모델 체크포인트 (gitignore)
└── outputs/                   # 실험 결과물 (gitignore)
```

## Usage

```python
from cv_pkg.utils import draw_bbox, show_images, load_image

# 이미지 로드 및 시각화
img = load_image('path/to/image.jpg')
img_with_boxes = draw_bbox(img, boxes=[(x1, y1, x2, y2)], labels=['object'])
show_images([img_with_boxes])
```