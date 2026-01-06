from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class MultiLayerConvNet(object):
    """
    A multi-layer convolutional network with the following architecture:

    {conv - relu} x M - 2x2 max pool - {affine - relu} x (L-1) - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    
    The {conv-relu} block is repeated M times and the {affine-relu} block is repeated L-1 times.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        nums_filters = [16,32],
        filter_size=5,
        hidden_dims = [500,100],
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - nums_filters: A list of integers giving the numbers of filters to use in the convolutional layers. 
                        M = len(nums_filters)
        - filter_size: Width/height of filters to use in the convolutional layer.
        - hidden_dims: A list of integers giving the numbers of units to use in the fully-connected hidden layers.
                        L-1 = len(hidden_dims)
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size

        ############################################################################
        # TODO: Initialize weights and biases for the multi-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the first layer    #
        # in W1 and b1; for the second layer use W2 and b2, etc.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the convolutional layers are chosen so that                #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 1. 准备参数
        C, H, W = input_dim
        self.M = len(nums_filters)      # 卷积层数量
        self.L = len(hidden_dims) + 1   # 全连接层总数 (隐藏层 + 输出层)
        
        # 2. 初始化卷积层 (前 M 层)
        # 输入通道数一开始是 C (比如 RGB=3)
        curr_C = C 
        
        for i in range(self.M):
            # 卷积核形状: (F, C, HH, WW)
            F = nums_filters[i]
            W_shape = (F, curr_C, filter_size, filter_size)
            
            # 使用高斯初始化权重，偏置设为0
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(*W_shape)
            self.params[f'b{i+1}'] = np.zeros(F)
            
            # 更新通道数，下一层的输入通道数 = 当前层的输出通道数
            curr_C = F

        # 3. 计算全连接层的输入维度
        # 结构是: {conv-relu}xM -> pool -> affine
        # 卷积层保持 H, W 不变，只有那个 2x2 pool 会让高宽减半
        pool_H = H // 2
        pool_W = W // 2
        # 全连接层的输入 = 最后一个卷积层的 Filter 数 * 减半后的高 * 减半后的宽
        flatten_dim = curr_C * pool_H * pool_W

        # 4. 初始化全连接层 (后 L 层)
        # 这一串列表包含了所有全连接层的维度：[flatten_dim, hidden_1, hidden_2, ..., num_classes]
        dims = [flatten_dim] + hidden_dims + [num_classes]
        
        for i in range(self.L):
            # 层的索引要接着卷积层后面算，所以是 M + 1 + i
            idx = self.M + 1 + i
            
            w_dim = dims[i]
            next_dim = dims[i+1]
            
            self.params[f'W{idx}'] = weight_scale * np.random.randn(w_dim, next_dim)
            self.params[f'b{idx}'] = np.zeros(next_dim)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the multi-layer convolutional network.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """

        X = X.astype(self.dtype)
        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size

        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the multi-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        caches = {}
        out = X
        
        # 1. 卷积层循环 (Conv - ReLU) x M
        for i in range(self.M):
            w = self.params[f'W{i+1}']
            b = self.params[f'b{i+1}']
            # 使用 layer_utils 里的组合层，自动处理 conv 和 relu
            out, cache = conv_relu_forward(out, w, b, conv_param)
            caches[f'c{i+1}'] = cache
            
        # 2. 池化层 (2x2 Max Pool)
        # 位于所有卷积层之后
        out, cache = max_pool_forward_fast(out, pool_param)
        caches['pool'] = cache
        
        # 3. 全连接隐藏层循环 (Affine - ReLU) x (L-1)
        for i in range(self.L - 1):
            idx = self.M + 1 + i
            w = self.params[f'W{idx}']
            b = self.params[f'b{idx}']
            out, cache = affine_relu_forward(out, w, b)
            caches[f'a{idx}'] = cache
            
        # 4. 最后一层 (Affine) -> 得到 Scores
        idx_last = self.M + self.L
        w = self.params[f'W{idx_last}']
        b = self.params[f'b{idx_last}']
        scores, cache = affine_forward(out, w, b)
        caches[f'a{idx_last}'] = cache
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the multi-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 1. 计算 Softmax Loss
        data_loss, dout = softmax_loss(scores, y)
        
        # 2. 计算正则化 Loss (L2 Regularization)
        # 也就是把所有 W 的平方加起来 * 0.5 * reg
        reg_loss = 0.0
        for k, v in self.params.items():
            if k.startswith('W'):
                reg_loss += 0.5 * self.reg * np.sum(v * v)
        
        loss = data_loss + reg_loss

        # 3. 反向传播 (Chain Rule)
        # 这里的顺序和 Forward 严格相反：
        
        # (A) 最后一层 Affine
        idx_last = self.M + self.L
        dout, dw, db = affine_backward(dout, caches[f'a{idx_last}'])
        grads[f'W{idx_last}'] = dw + self.reg * self.params[f'W{idx_last}']
        grads[f'b{idx_last}'] = db
        
        # (B) 中间的 Affine - ReLU 层
        for i in range(self.L - 2, -1, -1):
            idx = self.M + 1 + i
            dout, dw, db = affine_relu_backward(dout, caches[f'a{idx}'])
            grads[f'W{idx}'] = dw + self.reg * self.params[f'W{idx}']
            grads[f'b{idx}'] = db
            
        # (C) 池化层 Backprop
        dout = max_pool_backward_fast(dout, caches['pool'])
        
        # (D) 卷积层 Backprop
        for i in range(self.M - 1, -1, -1):
            idx = i + 1
            dout, dw, db = conv_relu_backward(dout, caches[f'c{idx}'])
            grads[f'W{idx}'] = dw + self.reg * self.params[f'W{idx}']
            grads[f'b{idx}'] = db
        
    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads