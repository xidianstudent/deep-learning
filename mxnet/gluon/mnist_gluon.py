import mxnet as mx
from mxnet.gluon import nn, loss as gloss
from mxboard import SummaryWriter
from mxnet import autograd, nd, init
import numpy as np

mnist_data = mx.test_utils.get_mnist()
mnist_train_data = mnist_data['train_data']
mnist_train_label= mnist_data['train_label']
mnist_test_data = mnist_data['test_data']
mnist_test_label = mnist_data['test_label']

batchSize = 100
lr = 0.01
epoch_num = 5

# train_data_iter = mx.io.NDArrayIter(mnist_train_data, label=mnist_train_label, batch_size=batchSize)
# test_data_iter = mx.io.NDArrayIter(mnist_test_data, label=mnist_test_label)
train_data = mx.gluon.data.ArrayDataset(mnist_train_data, mnist_train_label)
train_data_iter = mx.gluon.data.DataLoader(train_data,batch_size=batchSize,shuffle=True)

test_data = mx.gluon.data.ArrayDataset(mnist_test_data, mnist_test_label)
test_data_iter = mx.gluon.data.DataLoader(test_data)

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=64, kernel_size=5),
    nn.Activation('relu'),
    nn.Conv2D(channels=32, kernel_size=3),
    nn.Activation('relu'),
    nn.Dense(10)
)

net.initialize()

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = mx.gluon.Trainer(net.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': lr})

for epoch in range(epoch_num):
    for X, Y in train_data_iter:
        with autograd.record():
            output = net(X)
            l = loss(output, Y)
        l.backward()
        trainer.step(batchSize)
    yy = loss(net(nd.array(mnist_test_data)), nd.array(mnist_test_label))
    print('epoch = %d, loss = %.2f'%(epoch, yy.mean().asnumpy()))

yy = net(nd.array(mnist_test_data))
b = np.argmax(yy[0].asnumpy())
print(b)
print(mnist_test_label[0])