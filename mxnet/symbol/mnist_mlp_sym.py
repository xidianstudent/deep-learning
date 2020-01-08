import mxnet as mx
from mxnet import autograd, nd
from mxboard import SummaryWriter
import numpy as np

mnist_data = mx.test_utils.get_mnist()
# print(mnist_data)

train_data = mnist_data['train_data']
train_label= mnist_data['train_label']
test_data = mnist_data['test_data']
test_label = mnist_data['test_label']

def epochEndCallback(epoch, sym, args, augs):
    # print(args)
    print(sym)
    print('epoch = %d'%(epoch))

    mod.save_checkpoint('./models/mnist_mlp',epoch)
    
batchSize = 50
lr = 0.01
epoch_num = 50

train_data_iter = mx.io.NDArrayIter(train_data, label=train_label, data_name='input_data', label_name='input_label', batch_size=batchSize, shuffle=True)
test_data_iter = mx.io.NDArrayIter(test_data, label=test_label)
# print(test_label)

net = mx.sym.Variable('input_data')
label = mx.sym.Variable('input_label')
net = mx.sym.Flatten(data=net, name='flatten1')
net = mx.sym.FullyConnected(data=net, num_hidden=64, name='fc1')
net = mx.sym.relu(data=net, name='relu1')
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=32)
net = mx.sym.relu(data=net, name='relu2')
net = mx.sym.FullyConnected(data=net, num_hidden=10, name='fc3')
net = mx.sym.SoftmaxOutput(data=net, label=label, name='softmax_output')

# mx.viz.plot_network(net).view()
mod = mx.mod.Module(symbol=net, data_names=['input_data'], label_names=['input_label'])
mod.fit(train_data_iter,eval_metric='acc',optimizer='sgd', optimizer_params={'learning_rate': lr}, num_epoch=epoch_num,epoch_end_callback=epochEndCallback)

#score
metric = mx.metric.Accuracy()
mod.score(test_data_iter, metric)
print(metric.get())

#predict
yy = mod.predict(test_data_iter)
b = np.argmax(yy[0].asnumpy())
print(b)
print(test_label[0])
