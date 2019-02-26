import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import report


class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__()

        with self.init_scope():
            self.fc1 = L.Linear(1, 5)
            self.lstm = L.LSTM(5, 5)
            self.fc2 = L.Linear(5, 1)

    def __call__(self, x):
        h = self.fc1(x)
        h = self.lstm(h)
        return self.fc2(h)

    def reset_state(self):
        self.lstm.reset_state()

class LossFuncL(chainer.Chain):

    def __init__(self, predictor):
        super(LossFuncL, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        x.data = x.data.reshape((-1, 1)).astype(np.float32)
        t.data = t.data.reshape((-1, 1)).astype(np.float32)

        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        report({'loss':loss}, self)
        return loss
