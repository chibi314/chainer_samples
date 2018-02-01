from chainer.datasets import mnist
import matplotlib.pyplot as plt
from chainer import iterators
import random
import numpy as np
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
from chainer import serializers
from mnist_network import *
import chainer

train, test = mnist.get_mnist(withlabel=True, ndim=1)

'''
x, t = train[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
print('label:', t)

'''

batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

random.seed(0)
np.random.seed(0)

if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(0)

gpu_id = 1

net = MLP()

if gpu_id >= 0:
    net.to_gpu(gpu_id)

'''
print('1つ目の全結合相のバイアスパラメータの形は、', net.l1.b.shape)
print('初期化直後のその値は、', net.l1.b.data)
'''

optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(net)

max_epoch = 10

while train_iter.epoch < max_epoch:
    train_batch = train_iter.next()
    x, t = concat_examples(train_batch, gpu_id)

    y = net(x)

    loss = F.softmax_cross_entropy(y, t)

    net.cleargrads()
    loss.backward()

    optimizer.update()

    if train_iter.is_new_epoch:

        print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float(to_cpu(loss.data))), end='')

        test_losses = []
        test_accuracies = []

        while True:
            test_batch = test_iter.next()
            x_test, t_test = concat_examples(test_batch, gpu_id)

            y_test = net(x_test)

            loss_test = F.softmax_cross_entropy(y_test, t_test)
            test_losses.append(to_cpu(loss_test.data))

            accuracy = F.accuracy(y_test, t_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.data)

            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(np.mean(test_losses), np.mean(test_accuracies)))

serializers.save_npz('my_mnist.model', net)
