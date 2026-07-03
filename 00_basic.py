import torch
import tensorflow as tf

#x = tf.Variable(2.0, name="x")
#with tf.GradientTape() as tape:
#    y = x**2 + 3*x + 4
#    grad = tape.gradient(y, x)

#print(f'y 값: {y.numpy()}')
#print(f'x에 대한 기울기: {grad.numpy()}')

#===========================================
#x = torch.tensor(2.0, requires_grad=True)
#y = x**2 + 3*x + 4

#print(f"y 값: {y.item()}")

#y.backward()
#print(f"x에 대한 기울기: {x.grad.item()}")

#===========================================
#x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
#y = x * 2
#z = y.sum()
#z.backward()
#print(f"x.grad: {x.grad}")

#===========================================
#def main():
#    x = torch.tensor([1.0, 2.0, 3.0])
#    print('Hello, PyTorch!')
#    print(f'Tensor x: {x}')

#main()

# xpu :  Intel(R) Arc(TM) Graphics 사용 가능 여부
#print(tf.config.list_physical_devices("GPU"))
#print(torch.xpu.is_available())
#print(torch.xpu.get_device_name(0))

#device = torch.device("xpu")
#x = torch.randn(3, 3).to(device)
#print(x)
