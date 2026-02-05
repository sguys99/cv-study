# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer Vision 학습용 Python 프로젝트. YOLO 기반 객체 탐지와 Vision-Language 모델(CLIP) 실험을 다룬다.

## Development Commands

```bash
# 의존성 설치
uv sync

# Jupyter 노트북 실행
uv run jupyter notebook

# 단일 Python 스크립트 실행
uv run python script.py
```

## Architecture

- **notebooks/**: 학습 노트북
  - `01_object_detection/`: YOLO 기반 객체 탐지
  - `02_vlm/`: Vision-Language 모델 (CLIP)
  - `hf_tutorial/`: Hugging Face transformers 튜토리얼
- **src/cv_pkg/**: 공용 유틸리티 패키지
  - `utils/visualization.py`: `draw_bbox()`, `show_images()` - 시각화 (BGR→RGB 자동 변환)
  - `utils/data.py`: `load_image()`, `preprocess_image()` - 이미지 로딩/전처리
- **data/**, **models/**, **outputs/**: gitignore 처리된 데이터/모델/결과 디렉토리

## Key Dependencies

- **ultralytics**: YOLO 객체 탐지
- **transformers**: CLIP 등 Hugging Face 모델
- **timm**: PyTorch 이미지 모델 라이브러리
- **opencv-python**: 이미지 처리
- **superb-ai-***: Superb AI 플랫폼 연동 (데이터 라벨링)

## Notes

- Python 3.12.12 사용 (`uv`로 관리)
- 노트북에서 cv_pkg 사용 시 `sys.path.insert(0, '../../src')` 필요
- 이미지는 OpenCV BGR 포맷으로 처리됨
