import chainer
from chainer.datasets import tuple_dataset
import numpy as np
from model import *
from chainer import optimizers
from iterator import *
from chainer import report
import matplotlib.pyplot as plt

N_data  = 100
N_Loop  = 3
t = np.linspace(0., 2*np.pi*N_Loop, num=N_data)

X = 0.8*np.sin(2.0*t)

# データセット
N_train = int(N_data*0.8)
N_test  = int(N_data*0.2)

tmp_DataSet_X= np.array(X).astype(np.float32)

x_train, x_test = np.array(tmp_DataSet_X[:N_train]),np.array(tmp_DataSet_X[N_train:])

train = tuple_dataset.TupleDataset(x_train)
test  = tuple_dataset.TupleDataset(x_test)

model = LossFuncL(MLP())
optimizer = optimizers.Adam()
optimizer.setup(model)

train_iter = LSTM_test_Iterator(train, batch_size = 10, seq_len = 10)
test_iter  = LSTM_test_Iterator(test,  batch_size = 10, seq_len = 10, repeat = False)

max_epoch = 100

while train_iter.epoch < max_epoch:

    batch = np.array(train_iter.__next__()).astype(np.float32)
    x, t  = batch[:,0].reshape((-1,1)), batch[:,1].reshape((-1,1))

    loss = model(chainer.Variable(x), chainer.Variable(t))

    model.cleargrads()
    loss.backward()
    optimizer.update()
    model.predictor.reset_state()

    if train_iter.is_new_epoch:  # 1 epochが終わったら

        # ロスの表示
        print('epoch:{:02d} train_loss:{:.04f} \n'.format(train_iter.epoch, loss.data), end='')

presteps = 10
model.predictor.reset_state()

for i in range(presteps):
    y = model.predictor(chainer.Variable(np.roll(x_train,i).reshape((-1,1))))

print(y)
#plt.plot(t[:N_train],np.roll(y.data,-presteps))
plt.plot(y.data)
plt.show()
