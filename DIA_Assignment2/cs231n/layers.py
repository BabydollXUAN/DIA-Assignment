from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 1. 算出batch size(N)
    N=x.shape[0]

    # 2.压扁输入：把(N, d1, d2, ...)变成(N, D)
    # -1 表示自动计算剩下的维度大小
    x_reshaped=x.reshape(N, -1)

    # 3.线性运算:Y=XW+b
    out=x_reshaped.dot(w)+b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #1. 同样先把x压扁，因为反向传播要用到前向时的形状
    N=x.shape[0]
    x_reshaped=x.reshape(N, -1)

    # 2.计算dw:Input^T *dout
    dw=x_reshaped.T.dot(dout)

    # 3.计算dx: dout *W^T
    # 算出后，必须reshape 回x原来的形状(N, d1, d2...)
    dx=dout.dot(w.T).reshape(x.shape)

    # 4.计算db：在batch 维度上求和
    db=np.sum(dout, axis=0)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    # 简单的取最大值操作
    out=np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 复制一份上游传下来的梯度
    dx = dout.copy()
      
    # 把 forward 阶段 x <= 0 的位置的梯度设为 0
    dx[x <= 0] = 0    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Compute the softmax loss and its gradient.                          #
    # Store the loss in loss and the gradient in dx. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!  
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    
    # 1. 数值稳定性：每个样本减去该样本最大的分数
    # keepdims=True 保持维度以便广播
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    
    # 2. 计算 Softmax 概率
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    probs = np.exp(shifted_logits) / Z
    
    # 3. 计算 Loss: -log(正确类别的概率)
    # y 是正确类别的索引，我们用 fancy indexing 取出来
    correct_class_probs = probs[np.arange(N), y]
    loss = -np.log(correct_class_probs).sum() / N
    
    # 4. 计算梯度 dx: probs - indicator(y)
    dx = probs.copy()
    # 在正确类别的位置减 1
    dx[np.arange(N), y] -= 1
    dx /= N    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 1. 提取参数
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # 2. 计算输出尺寸
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    # 3. 填充 Input (Padding)
    # ((0,0), (0,0), (pad, pad), (pad, pad)) 分别对应 N, C, H, W
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    # 4. 初始化输出
    out = np.zeros((N, F, H_out, W_out))
    
    # 5. 暴力循环 (Naive Loop)
    for n in range(N):             # 遍历每张图片
        for f in range(F):         # 遍历每个卷积核
            for i in range(H_out): # 遍历高度
                for j in range(W_out): # 遍历宽度
                    # 确定当前的窗口位置
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    # 取出这一小块 (Slice)
                    x_slice = x_pad[n, :, h_start:h_end, w_start:w_end]
                    
                    # 卷积运算：点积求和 + 偏置
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # 尺寸
    N, F, H_out, W_out = dout.shape
    
    # Padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    # 初始化梯度
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # db 很简单，直接在 (N, H', W') 维度求和
    db = np.sum(dout, axis=(0, 2, 3))
    
    # 暴力循环计算 dw 和 dx
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    # 找到窗口位置
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    x_slice = x_pad[n, :, h_start:h_end, w_start:w_end]
                    
                    # dw += Input_slice * gradient
                    dw[f] += x_slice * dout[n, f, i, j]
                    
                    # dx += Weight * gradient
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]
    
    # 去掉 padding，得到最终的 dx
    # [pad:-pad] 表示去掉头尾的 padding
    if pad > 0:
        dx = dx_pad[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_pad
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_out = 1 + (H - HH) // stride
    W_out = 1 + (W - WW) // stride
    
    out = np.zeros((N, C, H_out, W_out))
    
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + HH
                w_start = j * stride
                w_end = w_start + WW
                
                # 在窗口内找最大值
                x_slice = x[n, :, h_start:h_end, w_start:w_end]
                out[n, :, i, j] = np.max(x_slice, axis=(1, 2))    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    H_out, W_out = dout.shape[2], dout.shape[3]
    
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):      # 注意 Pooling 是逐通道处理的
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    x_slice = x[n, c, h_start:h_end, w_start:w_end]
                    
                    # 找到最大值的位置 (Mask)
                    max_val = np.max(x_slice)
                    mask = (x_slice == max_val)
                    
                    # 梯度只传给最大值的位置
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

