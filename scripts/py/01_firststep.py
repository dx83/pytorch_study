import torch
import numpy as np
import time

def step1_basic_tensor():
    # 텐서 기초
    print('=' * 50)
    print('1단계: 텐서가 뭔가요?')
    print('=' * 50)

    # 0차원 텐서
    scalar = torch.tensor(5.0)
    print(f'스칼라 (0차원): {scalar}, 모양: {scalar.shape}')

    # 1차원 텐서
    vector = torch.tensor([1.0, 2.0, 3.0])
    print(f'벡터 (1차원): {vector}, 모양: {vector.shape}')

    # 2차원 텐서
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f'행렬 (2차원):\n{matrix}\n모양: {matrix.shape}')

    print('텐서는 숫자들을 n차원 배열로 정리한 것이다.\n')

def step2_tensor_operations():
    # 텐서 연산
    print('=' * 50)
    print('2단계: 텐서로 계산하기')
    print('=' * 50)

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f'a = {a}')
    print(f'b = {b}')
    print(f'덧셈: a + b = {a+b}')
    print(f'곱셈: a * b = {a*b}')
    print(f'내적: a · b = {torch.dot(a, b)}')

    # 브로드캐스팅
    c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    d = torch.tensor([10.0, 20.0])
    print(f'\n행렬:\n{c}')
    print(f'벡터: {d}')
    print(f'브로드캐스팅 결과:\n{c+d}')
    print('크기가 다른 텐서도 자동으로 맞춰서 계산됩니다!\n')

def step3_xpu_magic():
    # XPU 활용
    print('=' * 50)
    print('3단계: XPU로 가속하기')
    print('=' * 50)

    # XPU 사용 가능 여부 확인
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f'사용 중인 장치: {device}')
    
    if torch.xpu.is_available():
        print(f'XPU 이름: {torch.xpu.get_device_name(0)}')
        print(f'XPU 메모리: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # 큰 행렬로 속도 비교
    # CPU 연산
    a_cpu = torch.randn(10000, 10000)
    b_cpu = torch.randn(10000, 10000)

    start_time = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f'CPU 연산 시간: {cpu_time:.4f}초')

    if torch.xpu.is_available():
        # XPU 연산
        a_xpu = a_cpu.to(device)
        b_xpu = b_cpu.to(device)

        start_time = time.time()
        c_xpu = torch.matmul(a_xpu, b_xpu)
        torch.xpu.synchronize() # XPU 연산 완료 대기
        xpu_time = time.time() - start_time
        print(f'XPU 연산 시간: {xpu_time:.4f}초')
        print(f'속도 향상: {cpu_time / xpu_time:.1f}배 빨라짐!')
    else:
        print('XPU가 없어서 CPU로만 실행됩니다.')

def step4_dynamic_graph():
    print('=' * 50)
    print('4단계: 동적 그래프의 마법')
    print('=' * 50)

    # 자동 미분 활성화
    x = torch.tensor(2.0, requires_grad=True)
    print(f'입력값 x = {x}')
    # 계산 과정
    y = x**2 + 3*x + 1
    print('함수: y = x² + 3x + 1')
    print(f'x=2일 때 y = {y}')
    # 자동으로 미분 계산!
    y.backward()
    print(f'dy/dx = {x.grad}')
    print('수학적으로 계산하면: dy/dx = 2x + 3 = 2*2 + 3 = 7')
    print('파이토치가 자동으로 미분을 계산했습니다!\n')

def main():
    print('파이토치 첫걸음을 시작합니다!')
    print('=' * 50)
    print(torch.device('cpu'))
    #step1_basic_tensor()
    #step2_tensor_operations()
    #step3_xpu_magic()
    step4_dynamic_graph()

    print('1장 실습 완료!!')

main()

