import torch
import numpy as np
import time
# 돌리기전에 메모리(RAM) 확인 요망!!!!
#
a = torch.arange(1, 6)
b = torch.arange(6, 11)
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")

#
c = torch.tensor([[1], [2], [3]])   # shape (3, 1)
d = torch.tensor([10, 20, 30])      # shape (3,)
print(f"브로드캐스팅 c + d:\n {c + d}")

#
np_a = a.numpy()
np_b = b.numpy()
np_result = np_a + np_b
pt_result = a + b
print(f"NumPy 덧셈 결과: {np_result}")
print(f"PyTorch 덧셈 결과: {pt_result}")

#
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
x_cpu = torch.randn((30000, 30000))
y_cpu = torch.randn((30000, 30000))

start = time.time()
_ = torch.mm(x_cpu, y_cpu)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f}sec")

if torch.xpu.is_available():
    x_xpu = x_cpu.to(device)
    y_xpu = y_cpu.to(device)

    torch.xpu.synchronize()
    start = time.time()
    _ = torch.mm(x_xpu, y_xpu)
    torch.xpu.synchronize()
    xpu_time = time.time() - start
    print(f"XPU time: {xpu_time:.4f}sec")

print(f"속도 향상: {cpu_time / xpu_time:.1f}배 빠름")
