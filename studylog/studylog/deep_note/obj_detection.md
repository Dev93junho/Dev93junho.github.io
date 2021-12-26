# Object Detection

- 영역 추정
- Detection을 위한 deep learning 네트워크 구성
- Detection 을 구성하는 기타요소

# Object Detection 모델의 구성요소
### Backbone
- ResNet 

### Neck
- Feature Pyramid Network

### Head
- Classification 
- Bounding box regression

### object Localization
Annotation -> Feature Extractor -> Feature Map -> Fully Connect layer -> Softmax

### Sliding Window Method
Window 를 왼쪽 상단에서 부터 오른쪽하단으로 이동시키며 Object를 Detection 하는 방식
1. 다양한 shape의 window를 각각 sliding 시키는 방식
2. Window Scale을 고정하고 Image Scale을 변경한 여러 이미지를 사용하는 방식

- Obj detection의 초기 기법으로 활용
- Object 없는 영역도 무조건 슬라이딩 하여 비효율적임


### Selective Search
#### Region Proposal(영역추정)
"Object가 있을 만한 후보영역의 탐색"
원본 이미지 -select-> 후보 Bounding Box 선택 -(최종후보 도출)-> 최종 Object Detection