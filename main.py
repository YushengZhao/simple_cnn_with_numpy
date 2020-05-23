import numpy as np
from layers import *
from adam import *

SIZE = 100
xs = np.random.standard_normal((SIZE, 3, 32, 32))
labels = np.random.randint(0, 10, SIZE, dtype='int32')
conv1_param = {
    'name': 'conv1',
    'in_shape': (3, 32, 32),
    'cores': np.random.standard_normal((6, 3, 3, 3)) / np.sqrt(27),
    'bias': np.zeros((6, 32, 32))
}
conv2_param = {
    'name': 'conv2',
    'in_shape': (6, 16, 16),
    'cores': np.random.standard_normal((8, 6, 3, 3)) / np.sqrt(54),
    'bias': np.zeros((8, 16, 16))
}
fc1_param = {
    'name' : 'fc1',
    'W': np.random.standard_normal((64, 512)) / np.sqrt(512),
    'b': np.zeros((64,))
}
fc2_param = {
    'name' : 'fc2',
    'W': np.random.standard_normal((10, 64)) / np.sqrt(32),
    'b': np.zeros((10,))
}


def main():
    for epoch in range(10):
        losses = 0
        for i in range(SIZE):
            x = xs[i]
            label = labels[i]

            y, conv1_cache = conv_forward(x, conv1_param)
            y, pool1_cache = pool_forward(y)
            y, conv2_cache = conv_forward(y, conv2_param)
            y, pool2_cache = pool_forward(y)
            y = y.flatten()
            y, fc1_cache = fc_forward(y, fc1_param)
            y, relu_cache = relu_forward(y)
            y, fc2_cache = fc_forward(y, fc2_param)
            grad, loss = loss_eval(y, label)
            dy, d_fc2_param = fc_backward(grad, fc2_param, fc2_cache)
            dy = relu_backward(dy, relu_cache)
            dy, d_fc1_param = fc_backward(dy, fc1_param, fc1_cache)
            dy = dy.reshape((8, 8, 8))
            dy = pool_backward(dy, pool2_cache)
            dy, d_conv2_param = conv_backward(dy, conv2_param, conv2_cache)
            dy = pool_backward(dy, pool1_cache)
            dx, d_conv1_param = conv_backward(dy, conv1_param, conv1_cache)

            adam(fc1_param, d_fc1_param)
            adam(fc2_param, d_fc2_param)
            adam(conv1_param, d_conv1_param)
            adam(conv2_param, d_conv2_param)

            losses += loss
        print('epoch {}: loss={}'.format(epoch, str(losses / SIZE)))

if __name__ == '__main__':
    main()
