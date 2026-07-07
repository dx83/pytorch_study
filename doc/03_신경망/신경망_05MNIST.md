# MNIST

## 실험 시 주의 사항
1. GPU 가용성 확인
    - .to(device)를 사용해 모델과 데이터를 GPU로 이동시켜 학습 속도를 향상

2. 데이터 정규화 확인
    - Normalize((0.1307,), (0.3081,))을 적용하여 평균과 표준 편차를 조정

3. 과적합 방지
    - 훈련 데이터에만 높은 성능을 보이고, 테스트 성능이 낮을 경우 과적합

4. 학습률 설정
    - 너무 크면 학습이 불안정하고, 너무 작으면 수렴 속도가 느리다.
    - lr=0.001이 일반적으로 무난한 값, 다양한 값으로 실험 권장

5. 모델 저장 및 복원
    - 저장 : torch.save(model.state_dict(), '파일명.pth')
    - 복원 : model.load_state_dict(torch.laod('파일명.pth'))

<br>

## 추가 실험 및 권장 사항
1. Dropout 추가
    - 과적합 방지
2. Batch Normalization 적용
    - 초기 학습 안정성을 향상
3. Optimizer 변경 실험
    - Adam, SGD, RMSprop 등 다양한 옵티마이저로 성능 비교
4. Batch Size 변경 실험
    - 속도 및 성능 변화 확인
5. 정확도 평가 및 혼동 행렬 분석
    - sklearn.metrics.classification_report, confusion_matrix 로 예측 성능 분석 
