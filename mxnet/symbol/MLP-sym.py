import mxnet as mx
from mxnet import autograd, nd
from mxboard import SummaryWriter
import os

input_samples = 1000
feat_num = 2

train_data = nd.random.uniform(-1, 1, shape=(input_samples, feat_num))
true_w = nd.array([[4.3, 6.3]])
true_b = nd.array([[8.3]])
train_label = nd.dot(train_data, nd.transpose(true_w)) + true_b

eval_data = nd.random.uniform(-1, 1, shape=(10, 2))
eval_label = nd.dot(eval_data, nd.transpose(true_w)) + true_b

# print(train_label)
print(eval_label)

batchsz = 10
epoch_num = 50
lr = 0.01

train_data_iter = mx.io.NDArrayIter(train_data, label=train_label, batch_size=batchsz, shuffle=True, data_name='input_x', label_name='input_y')
eval_data_iter = mx.io.NDArrayIter(eval_data, label=eval_label)

net = mx.sym.Variable('input_x')
y = mx.sym.Variable('input_y')
# weight = mx.sym.Variable('weight')
# bias = mx.sym.Variable('bias')
net = mx.sym.FullyConnected(data=net, weight=None, bias= None ,num_hidden=1, name='fc1')
net = mx.sym.LinearRegressionOutput(data=net, label=y,name='lroutput')

# mx.viz.plot_network(net).view()
nd_weight = nd.random.normal(scale=0.5, shape=(1, feat_num))
nd_bias = nd.zeros(shape=(1,1))

mod = mx.mod.Module(symbol=net, data_names=['input_x'], label_names=['input_y'])
# mod.fit(train_data_iter,num_epoch=epoch_num)

mod.bind(train_data_iter.provide_data, label_shapes=train_data_iter.provide_label, for_training=True)
mod.init_params(mx.init.Uniform(scale=1.5))
mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': lr})
metrics = mx.metric.create('acc')

for epoch in range(epoch_num):
    train_data_iter.reset()
    metrics.reset()
    for batch in train_data_iter:
        mod.forward(batch, is_train=True)
        mod.backward()
        mod.update_metric(metrics, batch.label)
        mod.update()
    print('epoch = %d, acc = %s'%(epoch, metrics.get()))

yy = mod.predict(eval_data_iter)
print('++++++++++++++++++++++')
print(yy)

print(mod.get_params())