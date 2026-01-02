---
layout: post
title:  "[Studying] YOLO를 활용한 Object Detection 기초"
date: 2026-01-02 15:28:44 +0900
categories: AI YOLO Object_Detection 2학년
---
제목을 눌러 본문을 확인하세요.  

# 서론
---
서화 누나와 태민이와 함께 G-RISE 2025 지역 산업군 PoC 연구 수행 과제에 참여하게 되었다.  
아직 주제를 확정짓진 못했지만, 일단은 우리 연구실이 Computer Vision 분야에 자부심이 있어  
연구 수행 과제도 Computer Vision 분야 중 하나인 Object Detection으로 진행하기로 하였다.

하지만 아직 Object Detection에 대해 잘 알지 못하여 아주 빠르게 기초 개념을 공부해보았다.

# 기본 설정
---
```py
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('yolov8n.pt')
img = cv2.imread('./bus.jpg')
```

`cv2`는 OpenCV 라이브러리로, 이미지 및 비디오 처리에 주로 사용된다.  
난 YOLO 모델 중 가장 작은 버전인 `yolov8n.pt`를 사용해 볼 것이다.

# 예측
---
```py
results = model.predict(source='./bus.jpg', save=True)
result = results[0]
```
`result[0]`는 이미지에 대한 예측 결과(Result Object)를 담고 있다.
`result.boxes`는 모델이 찾은 모든 경계 상자(Bounding Boxes) 정보를 포함하며,  
경계 상자의 좌표는 `xyxy`, 신뢰도 점수는 `conf`, class ID는 `cls` 속성을 통해 접근할 수 있다.

# 경계 상자 출력
---
```py
for box in result.boxes:
    coords = box.xyxy[0].cpu().numpy() # 경계상자 좌표를 (x1, y1, x2, y2) 형태로 반환
    x1, y1, x2, y2 = map(int, coords) # 좌표를 정수형으로 변환

    cls = int(box.cls[0].item()) 
    conf = box.conf[0].item() * 100  # 신뢰도 점수를 백분율로 변환

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"ID: {cls} Conf: {conf:.2f}"
    cv2.putText(img, f"{cls}: {conf:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
cv2.imshow('Detection Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
`cv2.putText(이미지, 출력할 텍스트, 텍스트 출력 좌표, 폰트, 폰트 크기, 색상, 두께)`  
`cv2.imshow()`는 이미지를 화면에 표시한다.  
`cv2.waitKey(0)`는 키 입력을 대기한다.  
`cv2.destroyAllWindows()`는 모든 OpenCV 창을 닫는다.
