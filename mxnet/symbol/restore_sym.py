import mxnet as mx
from mxboard import SummaryWriter
from mxnet import autograd, nd
import numpy as np

saved_checkpoints_path = './models/mnist_mlp'
epoch = 49

mnist_data = mx.test_utils.get_mnist()
train_data = mnist_data['train_data']
train_label= mnist_data['train_label']
test_data = mnist_data['test_data']
test_label = mnist_data['test_label']

test_data_iter = mx.io.NDArrayIter(test_data, label=test_label, data_name='input_data', label_name='input_label')

sym, args, augs = mx.mod.module.load_checkpoint(saved_checkpoints_path,epoch)

# mx.viz.plot_network(sym).view()
mlp_model = mx.mod.Module(symbol=sym, data_names=['input_data'], label_names=['input_label'])
mlp_model.bind(test_data_iter.provide_data, label_shapes=test_data_iter.provide_label, for_training=False)
mlp_model.set_params(args, augs)

#score
metrics = mx.metric.Accuracy()
mlp_model.score(test_data_iter, metrics)
print(metrics.get())

#predict
yy = mlp_model.predict(test_data_iter)
b = np.argmax(yy[0].asnumpy())
print(b)
print(test_label[0])