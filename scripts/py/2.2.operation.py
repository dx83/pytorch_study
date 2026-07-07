import torch

#a = torch.tensor([1,2,3])
#b = torch.tensor([4,5,6])

#print(f"a + b = {a + b}")
#print(f"a - b = {b - a}")
#print(f"a * b = {a * b}")
#print(f"a / b = {b / a}")

#==================================
#print(f"a + 10 = {a + 10}")
#print(f"b * 2 = {b * 2}")

#==================================
#a = torch.tensor([1,2,3])   # shape (3,)
#b = torch.tensor([10])      # shape (1,)
#print(f"a + b = {a + b}")   # b가 자동으로 [10, 10, 10]으로 확장

#==================================
#tensor2d = torch.tensor([[10, 20, 30],
#                         [40, 50, 60],
#                         [70, 80, 90]])
#print(f"tensor2d[0, 1] = {tensor2d[0, 1]}")
#print(f'tensor2d[1:, :] =\n {tensor2d[1:, :]}')

#==================================
#mask = tensor2d > 50
#print(f'mask:\n {mask}')
#print(f'50보다 큰 값들: {tensor2d[mask]}')

#==================================
#tensor_flat = torch.arange(6)
#reshaped = tensor_flat.view(2, 3)
#print(f'reshaped:\n {reshaped}')

#==================================
#t1 = torch.tensor([1,2,3])
#t2 = torch.tensor([4,5,6])

#cat_result = torch.cat((t1, t2), dim=0)
#print(f'cat_result: {cat_result}')

#stack_result = torch.stack((t1, t2), dim=0)
#print(f'stack_result:\n {stack_result}')

#==================================
#big_tensor = torch.arange(10)

#split_result = torch.split(big_tensor, split_size_or_sections=3)
#print(f'split_result: {split_result}')

#chunk_result = torch.chunk(big_tensor, chunks=3)
#print(f'chunk_result: {chunk_result}')

#==================================
#matrix = torch.tensor([[1,2,3],
#                      [4,5,6]])
#transposed = matrix.t()
#print(f'transposed:\n {transposed}')

tensor_3d = torch.randn(2, 3, 4)
permute_result = tensor_3d.permute(2, 1, 0)
print(f'tensor_3d:\n {tensor_3d}')
print(f'permute:\n {permute_result}')

x = torch.tensor([[1,2,3]])
squeeze = x.squeeze()
unsqueezed = squeeze.unsqueeze(0)
print(f'unsqueezed:\n {unsqueezed}')