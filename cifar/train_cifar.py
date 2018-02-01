from cifar_net import *
from chainer.datasets import cifar
from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer.training import extensions
from chainer.cuda import to_cpu

def train(network_object,
          batchsize=128,
          gpu_id=0,
          max_epoch=20,
          train_dataset=None,
          test_dataset=None,
          postfix='',
          base_lr=0.01,
          lr_decay=None):

    #1. Dataset
    if train_dataset is None and test_dataset is None:
        train, test = cifar.get_cifar10()
    else:
        train, test = train_dataset, test_dataset

    #2. Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)

    #3. Model
    net = L.Classifier(network_object)

    #4. Optimizer
    optimizer = optimizers.MomentumSGD(lr=base_lr)
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    #5. Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    #6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_cifar10_{}result'.format(network_object.__class__.__name__, postfix))

    #7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(test_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    if lr_decay is not None:
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_decay)
    trainer.run()
    del trainer

    return net

if __name__ == "__main__":
    model = train(DeepCNN(10), gpu_id=1, max_epoch=100, base_lr=0.1, lr_decay=(30, 'epoch'))
