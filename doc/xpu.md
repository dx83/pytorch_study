`torch.xpu.synchronize()` : XPU에서 아직 끝나지 않은 연산이 모두 끝날 때까지 Python 코드 실행을 기다리게 하는 함수

### XPU 사용
```python
if torch.xpu.is_available():
    print("xpu 사용 가능!")
else:
    print("cpu 모드로만 진행")
```

<br>

### 텐서를 XPU로 이동
```python
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"사용할 디바이스: {device}")             # xpu

x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(f"x가 위치한 디바이스: {x.device}")       # xpu:0
```

<br>

### CPU <=> XPU
```python
x_cpu = x.cpu()
print(f'x_cpu가 위치한 디바이스: {x_cpu.device}')
```

<br>

### 해제
- 연산이 많았던 경우 필수
```python
import gc
import torch

for name in ["a_cpu", "b_cpu", "a_xpu", "b_xpu"]: # 해제할 변수 모두 넣으면 됨
    if name in globals():
        del globals()[name]

gc.collect()

if torch.xpu.is_available():
    torch.xpu.synchronize()
    torch.xpu.memory.empty_cache()
```

<br>