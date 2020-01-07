import mxnet as mx
import os
from mxnet import autograd, nd
from mxboard import SummaryWriter
import random

train_data = nd.random.uniform(-1, 1, shape=(1000, 2))
true_w = nd.array([[5.3, 6.5]])
true_b = nd.array([[8.6]])
train_label = nd.dot(train_data, nd.transpose(true_w)) + true_b

# print(train_label)

weight = nd.random.normal(scale=1.0, shape=(1, 2))
bias = nd.zeros(shape=(1,1))

# print(weight)
# print(bias)

def data_iter(datas, labels, batchsize):
    data_len = len(datas)
    indices = list(range(data_len))
    random.shuffle(indices)
    for i in range(0, data_len, batchsize):
        j = nd.array(indices[i: min(i + batchsize, data_len)])
        yield (datas.take(j), labels.take(j))

def mlp(x, w, b):
    return nd.dot(x, nd.transpose(w)) + b

def square_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

def sgd(params, lr, batchsize):
    for param in params:
        param[:] = param - lr * param.grad / batchsize

def acc(y_hat, y):
    return 1.0 - (nd.sum(nd.abs(y_hat - y)) / len(y[0]))

batchsz = 10
lr = 0.01
epoch_num = 50
loss = square_loss
optimizer = sgd

weight.attach_grad()
bias.attach_grad()

sw = SummaryWriter('./logs', flush_secs=1)
for epoch in range(epoch_num):
    for x, y in data_iter(train_data, train_label, batchsz):
        with autograd.record():
            l = loss(mlp(x, weight, bias), y)
        l.backward()
        optimizer([weight, bias], lr, batchsz)
    sw.add_scalar('weight1', weight[0][0].asscalar(), global_step= epoch)
    sw.add_scalar('weight2', weight[0][1].asscalar(), global_step= epoch)
    sw.add_scalar('bias', bias[0][0].asscalar(), global_step=epoch)

    yy  = loss(mlp(train_data, weight, bias), train_label)
    yy1 = acc(mlp(train_data, weight, bias), train_label)
    sw.add_scalar('loss', yy.mean().asscalar(), global_step=epoch)
    sw.add_scalar('acc', yy1.asscalar(), global_step=epoch)

    print('epoch = %d, loss = %.2f, acc = %.2f'%(epoch, yy.mean().asnumpy(), yy1.asnumpy()))
    
print(weight)
print(bias)