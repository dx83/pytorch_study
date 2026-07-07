# DataLoader 활용 및 배치 구성
- DataLoader는 파이토치에서 데이터셋을 배치(batch) 단위로 나누고, 셔플(shuffle)이나 멀티프로세싱(Multiprocessing) 등 다양한 옵션을 제공하여 학습 속도를 향상시키는 데 도움을 준다.

<br>

### 배치 처리 (Batch Processing)
- 한 번에 여러 샘플을 처리함으로써 GPU나 CPU의 병렬 연산을 극대화

<br>

### 데이터 셔플 (Shuffling)
- 학습 중 에포크마다 데이터 순서를 무작위로 섞어 모델이 데이터 순서에 의존하지 않고 일반화 성능을 높일 수 있도록 한다.

<br>

### 멀티프로세싱 (Multiprocessing)
- num_workers 옵션을 통해 데이터를 여러 프로세스에서 동시에 불러올 수 있어 I/O 병목을 줄이고 학습 시간을 단축시킨다. 이를 통해 데이터 로딩 속도를 개선할 수 있다.

<br>

```python
from torch.utils.data import DataLoader

# 학습할 때 매 epoch마다 데이터를 64개씩 랜덤한 순서로 꺼내도록 설정
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
data_iter = iter(train_loader)      # train_loader를 반복자(iterator)로 바꿉니다.
# DataLoader는 (입력 데이터, 정답 라벨) 형태로 데이터를 반환한다.
images, labels = next(data_iter)    # next() 첫 번째 배치를 꺼냅니다.

print(f"배치 이미지의 shape: {images.shape}")   # [배치크기, 채널수, 높이, 너비]
```

<br>
