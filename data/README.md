# Data Directory

이 폴더에 학습/테스트용 데이터셋을 저장합니다.

## 권장 데이터셋

### Object Detection
- **COCO**: `https://cocodataset.org/`
- **Pascal VOC**: `http://host.robots.ox.ac.uk/pascal/VOC/`

### VLM
- **Flickr30k**: `https://shannon.cs.illinois.edu/DenotationGraph/`
- **MSCOCO Captions**: `https://cocodataset.org/#captions-2015`

## 폴더 구조 예시

```
data/
├── coco/
│   ├── images/
│   └── annotations/
├── custom/
│   ├── train/
│   └── val/
└── samples/
    └── test_images/
```

## 주의사항

- 이 폴더의 내용물은 `.gitignore`에 의해 Git에서 제외됩니다.
- 대용량 데이터셋은 로컬에만 저장하세요.
