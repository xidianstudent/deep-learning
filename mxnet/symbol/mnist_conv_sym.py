import mxnet as mx
from mxboard import SummaryWriter
from mxnet import autograd, nd
import numpy as np
import time

print(mx.__version__)

mnist_data = mx.test_utils.get_mnist()

mnist_train_data = mnist_data['train_data']
mnist_train_label= mnist_data['train_label']
mnist_test_data = mnist_data['test_data']
mnist_test_label = mnist_data['test_label']

batchSize = 50
lr = 0.01
epoch_num = 50

train_data_iter = mx.io.NDArrayIter(mnist_train_data, label=mnist_train_label, batch_size=batchSize, shuffle=True, data_name='input_data', label_name='input_label')
test_data_iter = mx.io.NDArrayIter(mnist_test_data, label=mnist_test_label)

def epochEndCallback(epoch, syms, args, augs):
    print('epoch = %d'%(epoch))

def batchEndCallback(params):
    pass
    # print('batch processing...')

data = mx.sym.Variable('input_data')
label = mx.sym.Variable('input_label')
net = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=64, name='conv1')
net = mx.sym.Activation(data=net, act_type='relu', name='relu1')
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=32, name='conv2')
net = mx.sym.Activation(data=net, act_type='relu', name='relu2')
net = mx.sym.FullyConnected(data=net, num_hidden=10, name='fc1')
net = mx.sym.SoftmaxOutput(data=net, label=label, name='softmax_output')

print('build model')
# mx.viz.plot_network(net).view()
mnist_mod = mx.mod.Module(symbol=net, data_names=['input_data'], label_names=['input_label'], context=mx.cpu())

start_time = time.time()
mnist_mod.fit(train_data_iter, eval_metric='acc', optimizer='sgd', optimizer_params={'learning_rate': lr}, num_epoch= epoch_num, epoch_end_callback=epochEndCallback, batch_end_callback=batchEndCallback)
end_time = time.time()

print('fit time = %.2f'%(end_time - start_time))
#score
metrics = mx.metric.Accuracy()
mnist_mod.score(test_data_iter, metrics)
print(metrics.get())

#predict
start_time = time.time()
yy = mnist_mod.predict(test_data_iter)
end_time = time.time()
print('predict time = %.2f'%(end_time - start_time))

b = np.argmax(yy[0].asnumpy())
print(b)
print(mnist_test_label[0])

#准确率可以达到98%以上