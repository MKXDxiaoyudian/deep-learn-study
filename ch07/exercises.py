"""第 7 章 卷积神经网络 —— 练习骨架

对应原书
- §7.2 卷积层（印刷 p202–213）：im2col 思想
- §7.3 池化层（p214–215）
- §7.4 卷积层和池化层的实现（p216–223）
- §7.5 CNN 的实现（p224–227）：SimpleConvNet

运行 `python exercises.py` 全过即完成。提示见 ch07/README.md。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def im2col(x, filter_h, filter_w, stride=1, pad=0):
    """把 4D 图像 (N,C,H,W) 展开成 (N*out_h*out_w, C*filter_h*filter_w)。

    out_h = (H + 2*pad - filter_h)//stride + 1；out_w 同理。
    提示：先 pad，再按 (i,j) 双循环切片，最后 reshape（书 §7.4 的写法）。
    """
    raise NotImplementedError("TODO: im2col")


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col 的逆过程：把 (N*out_h*out_w, C*fh*fw) 还原成 (N,C,H,W)。

    注意：im2col 有重叠时 col2im 是"把贡献加回去"（不是严格逆）；本练习只要求
    stride == filter 时能精确还原。
    """
    raise NotImplementedError("TODO: col2im")


def conv2d_naive(x, W, b, stride=1, pad=0):
    """朴素卷积（原书 §7.2 的"按定义算"版本），用于和 im2col 结果对照。

    x: (N,C,H,W)；W: (out_c,C,fh,fw)；b: (out_c,)；返回 (N,out_c,oh,ow)。
    """
    raise NotImplementedError("TODO: conv2d_naive")


class Convolution:
    """卷积层（§7.4）：用 im2col 实现 forward/backward，W 形状 (out_c,C,fh,fw)。"""
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b = W, b
        self.stride, self.pad = stride, pad
        self.x = self.col = self.col_W = None

    def forward(self, x):
        raise NotImplementedError("TODO: Convolution.forward")

    def backward(self, dout):
        raise NotImplementedError("TODO: Convolution.backward")


class Pooling:
    """最大池化层（§7.3）：2x2、stride=2、无 pad；backward 把梯度路由回最大值位置。"""
    def __init__(self, pool_h=2, pool_w=2, stride=2, pad=0):
        self.pool_h, self.pool_w, self.stride, self.pad = pool_h, pool_w, stride, pad
        self.x = self.arg_max = None

    def forward(self, x):
        raise NotImplementedError("TODO: Pooling.forward")

    def backward(self, dout):
        raise NotImplementedError("TODO: Pooling.backward")


def softmax_loss(z, t):
    """softmax + 交叉熵（batch），返回 (loss, dz)；dz=(y-t)/batch（ch05 已推导）。"""
    c = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - c)
    y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    batch = z.shape[0]
    loss = float(-np.sum(t * np.log(y + 1e-7)) / batch)
    return loss, (y - t) / batch


class SimpleConvNet:
    """§7.5 的简化版：Conv(1->4,3x3,pad=1) -> ReLU -> Pool(2x2) -> Affine -> Softmax。

    输入 (N,1,12,12)，输出 3 类。参数放 self.params 字典。
    """
    def __init__(self, num_filters=4, filter_size=3, hidden=16, num_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        C, H, W = 1, 12, 12
        self.params = {
            "W1": rng.normal(0, 0.1, (num_filters, C, filter_size, filter_size)),
            "b1": np.zeros(num_filters),
            "W2": rng.normal(0, 0.1, (num_filters * (H // 2) * (W // 2), hidden)),
            "b2": np.zeros(hidden),
            "W3": rng.normal(0, 0.1, (hidden, num_classes)),
            "b3": np.zeros(num_classes),
        }
        self.layers = {}

    def predict(self, x):
        raise NotImplementedError("TODO: SimpleConvNet.predict")

    def loss(self, x, t):
        raise NotImplementedError("TODO: SimpleConvNet.loss")

    def gradient(self, x, t):
        raise NotImplementedError("TODO: SimpleConvNet.gradient")

    def accuracy(self, x, t):
        raise NotImplementedError("TODO: SimpleConvNet.accuracy")


def main():
    rng = np.random.default_rng(0)

    # §7.4 im2col / col2im（非重叠时严格互逆）
    x = rng.normal(0, 1, (2, 1, 6, 6))
    col = im2col(x, 2, 2, stride=2, pad=0)
    assert col.shape == (2 * 3 * 3, 4), col.shape
    x2 = col2im(col, x.shape, 2, 2, stride=2, pad=0)
    assert np.allclose(x, x2), "stride=filter 时 im2col/col2im 应严格互逆"

    # im2col 卷积 == 朴素卷积
    W = rng.normal(0, 0.3, (3, 1, 3, 3))
    b = rng.normal(0, 0.1, 3)
    x3 = rng.normal(0, 1, (2, 1, 6, 6))
    out_naive = conv2d_naive(x3, W, b, stride=1, pad=1)
    conv = Convolution(W, b, stride=1, pad=1)
    out_fast = conv.forward(x3)
    assert out_fast.shape == out_naive.shape
    assert np.allclose(out_fast, out_naive, atol=1e-10)

    # §7.4 卷积层反向：与数值梯度核对（小尺寸）
    x4 = rng.normal(0, 1, (1, 1, 4, 4))
    W4 = rng.normal(0, 0.3, (2, 1, 3, 3))
    b4 = rng.normal(0, 0.1, 2)
    c4 = Convolution(W4, b4, stride=1, pad=0)
    out4 = c4.forward(x4)
    dout4 = rng.normal(0, 1, out4.shape)
    dX, dW, db = c4.backward(dout4)
    fW = lambda w: np.sum(Convolution(w, b4, 1, 0).forward(x4) * dout4)
    fX = lambda xx: np.sum(Convolution(W4, b4, 1, 0).forward(xx) * dout4)
    assert np.allclose(dW, numerical_gradient(fW, W4), atol=1e-4)
    assert np.allclose(dX, numerical_gradient(fX, x4), atol=1e-4)
    assert np.allclose(db, np.sum(dout4, axis=(0, 2, 3)), atol=1e-10)

    # §7.3 池化层
    p = Pooling(2, 2, stride=2)
    xp = rng.normal(0, 1, (2, 3, 4, 4))
    yp = p.forward(xp)
    assert yp.shape == (2, 3, 2, 2)
    dp = p.backward(np.ones_like(yp))
    assert dp.shape == xp.shape
    # 数值核对（随机数据，最大值唯一）
    fp = lambda xx: np.sum(Pooling(2, 2, 2).forward(xx))
    assert np.allclose(dp, numerical_gradient(fp, xp), atol=1e-6)

    # §7.5 小 CNN：合成图像三分类
    from common.toy_data import synthetic_images, one_hot

    X, y = synthetic_images(n_per_class=40, size=12, seed=0)
    n_train = 90
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    Ttr = one_hot(ytr, 3)
    net = SimpleConvNet(seed=0)
    for epoch in range(150):
        g = net.gradient(Xtr, Ttr)
        for k in net.params:
            net.params[k] -= 0.1 * g[k]
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: loss={net.loss(Xtr, Ttr):.3f}")
    acc = net.accuracy(Xte, yte)
    assert acc >= 0.90, acc
    print(f"[ch07] im2col/卷积/池化/CNN 全部通过；test acc={acc:.3f}")


def numerical_gradient(f, x):
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    h = 1e-4
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        x[idx] = tmp + h
        fx1 = f(x)
        x[idx] = tmp - h
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp
        it.iternext()
    return grad


if __name__ == "__main__":
    main()
