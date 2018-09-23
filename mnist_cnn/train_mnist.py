#!/usr/bin/env python

from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainercv
import numpy as np

# Network definition
class CNN(chainer.Chain):
    def __init__(self, n_out = 10):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = chainercv.links.Conv2DBNActiv(None, 64, 3, stride=1, pad=1)
            self.conv1_2 = chainercv.links.Conv2DBNActiv(None, 64, 3, stride=1, pad=1)
            self.conv1_3 = chainercv.links.Conv2DBNActiv(None, 64, 3, stride=1, pad=1)
            self.conv2_1 = chainercv.links.Conv2DBNActiv(None, 128, 3, stride=1, pad=1)
            self.conv2_2 = chainercv.links.Conv2DBNActiv(None, 128, 3, stride=1, pad=1)
            self.conv2_3 = chainercv.links.Conv2DBNActiv(None, 128, 3, stride=1, pad=1)
            self.conv3_1 = chainercv.links.Conv2DBNActiv(None, 256, 3, stride=1, pad=1)
            self.conv3_2 = chainercv.links.Conv2DBNActiv(None, 256, 3, stride=1, pad=1)
            self.conv3_3 = chainercv.links.Conv2DBNActiv(None, 256, 3, stride=1, pad=1)
            self.l1 = L.Linear(None, 100)
            self.l2 = L.Linear(None, n_out)

    def __call__(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(x)
        h = self.conv1_3(x)
        h = F.average_pooling_2d(h, 2)
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.conv2_3(h)
        h = F.average_pooling_2d(h, 2)
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = F.average_pooling_2d(h, 7)
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h

def transform(in_data):
    img, label = in_data
    img = img.reshape(28, 28)
    img = img[np.newaxis]
    return img, label

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(CNN())
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    train = chainer.datasets.TransformDataset(train, transform)
    test = chainer.datasets.TransformDataset(test, transform)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
