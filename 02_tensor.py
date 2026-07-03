import torch

#tensor1d = torch.tensor([1,2,3])
#print(f'1차원 텐서: {tensor1d}')

#tensor2d = torch.tensor([[1,2,3],
#                         [4,5,6]])
#print(f'2차원 텐서:\n {tensor2d}')

#===================================
#zeros_tensor = torch.zeros((3,3))
#ones_tensor = torch.ones((2,4))
#full_tensor = torch.full((2,2),7)

#arange_tensor = torch.arange(0, 10, step=2)
arange_tensor = torch.arange(0, 10, step=8)
#linspace_tensor = torch.linspace(0, 1, steps=5)
#linspace_tensor = torch.linspace(10, 100, steps=5)

#print(f'zeros:\n {zeros_tensor}')
#print(f'ones:\n {ones_tensor}')
#print(f'full:\n {full_tensor}')
print(f'arange:\n {arange_tensor}')
#print(f'linspace:\n {linspace_tensor}')

#===================================
#rand_tensor = torch.rand((3,3))     # 0,1 범위의 균일 분포
#randn_tensor = torch.randn((2,2))   # 평균 0, 표준 편차 1의 정규 분포

#print(f'rand:\n {rand_tensor}')
#print(f'randn:\n {randn_tensor}')

#===================================
#float_tensor = torch.tensor([1,2,3], dtype=torch.float32)
#int_tensor = torch.tensor([1,2,3], dtype=torch.int64)

#print(f'float: {float_tensor.dtype}')
#print(f'int: {int_tensor.dtype}')

#===================================
#int_tensor = torch.tensor([1,2,3], dtype=torch.int32)
#print(f'\nint_tensor: {int_tensor}')
#print(f'int_tensor dtype: {int_tensor.dtype}')

#float_tensor = int_tensor.float()
#print(f'\nfloat_tensor: {float_tensor}')
#print(f'float_tensor dtype: {float_tensor.dtype}')

#long_tensor = int_tensor.long()
#print(f'\nlong_tensor: {long_tensor}')
#print(f'long_tensor dtype: {long_tensor.dtype}')

#double_tensor = int_tensor.double()
#print(f'\ndouble_tensor: {double_tensor}')
#print(f'double_tensor dtype: {double_tensor.dtype}')

#===================================
def explore_tensor_dimensions():
    print("텐서 차원별 탐험을 시작합니다!")
    print("=" * 50)

    loss_value = torch.tensor(0.245)
    print(f"0차원 (스칼라): {loss_value}")
    print("   실제 의미: 딥러닝에서 손실값 등")
    print(f"   모양: {loss_value.shape}")
    print(f"   차원 수: {loss_value.dim()}\n")

    word_embedding = torch.tensor([0.2, -0.1, 0.8, 0.5])
    print(f"1차원 (벡터): {word_embedding}")
    print("   실제 의미: 단어 임베딩, 특성 벡터 등")
    print(f"   모양: {word_embedding.shape}")
    print(f"   차원 수: {word_embedding.dim()}\n")

    grayscale_image = torch.randint(0, 256, (4,4))
    print(f"2차원 (행렬):\n{grayscale_image}")
    print("   실제 의미: 이미지, 가중치 행렬")
    print(f"   모양: {grayscale_image.shape}")
    print(f"   차원 수: {grayscale_image.dim()}\n")

    color_image = torch.randint(0, 256, (3, 4, 4))
    print(f"3차원 (컬러 이미지): 모양 {color_image.shape}")
    print("   실제 의미: 컬러 이미지(높이x너비x채널)")
    print("   해석: 3개 채널 x 4높이 x 4너비")
    print(f"   차원 수: {color_image.dim()}\n")

    batch_images = torch.randint(0, 256, (2, 3, 4, 4))
    print(f"4차원 (이미지 배치): 모양 {batch_images.shape}")
    print("   실제 의미: 배치x채널x높이x너비")
    print("   해석: 2배치 x 3채널 x 4높이 x 4너비")
    print(f"   차원 수: {batch_images.dim()}\n")

    print("딥러닝에서 각 차원의 의미:")
    print("    - 배치 차원: 한 번에 처리할 데이터 개수")
    print("    - 특성 차원: 데이터의 특징 개수")
    print("    - 시퀀스 차원: 시계열 데이터의 길이")
    print("    - 공간 차원: 이미지의 높이, 너비")

explore_tensor_dimensions()