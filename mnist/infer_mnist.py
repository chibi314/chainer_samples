from mnist_network import *
from chainer import serializers
import matplotlib.pyplot as plt
from chainer.datasets import mnist
from chainer.cuda import to_cpu

infer_net = MLP()
serializers.load_npz('my_mnist.model', infer_net)

gpu_id = 0

if gpu_id >= 0:
    infer_net.to_gpu(gpu_id)

train, test = mnist.get_mnist(withlabel = True, ndim=1)

x, t = test[0]

plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()

print('元の形:', x.shape, end=' -> ')
x = x[None, ...]

print('ミニバッチの形にしたあと：', x.shape)

x = infer_net.xp.asarray(x)

y = infer_net(x)

y = y.array

y = to_cpu(y)

pred_label = y.argmax(axis=1)

print('ネットワークの予測：', pred_label[0])
