"""第 7 章 卷积神经网络 —— 参考答案。运行：python answers.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def im2col(x, filter_h, filter_w, stride=1, pad=0):
    x = np.asarray(x, dtype=np.float64)
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    xp = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode="constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=np.float64)
    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=np.float64)
    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            xp[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
    return xp[:, :, pad:pad + H, pad:pad + W]


def conv2d_naive(x, W, b, stride=1, pad=0):
    x = np.asarray(x, dtype=np.float64)
    N, C, H, Wd = x.shape
    out_c, _, fh, fw = W.shape
    xp = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode="constant")
    oh = (H + 2 * pad - fh) // stride + 1
    ow = (Wd + 2 * pad - fw) // stride + 1
    out = np.zeros((N, out_c, oh, ow), dtype=np.float64)
    for n in range(N):
        for oc in range(out_c):
            for i in range(oh):
                for j in range(ow):
                    patch = xp[n, :, i * stride:i * stride + fh, j * stride:j * stride + fw]
                    out[n, oc, i, j] = np.sum(patch * W[oc]) + b[oc]
    return out


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b = W, b
        self.stride, self.pad = stride, pad
        self.x = self.col = self.col_W = None

    def forward(self, x):
        self.x = x
        self.W_shape = self.W.shape
        self.col = im2col(x, self.W.shape[2], self.W.shape[3], self.stride, self.pad)
        self.col_W = self.W.reshape(self.W.shape[0], -1).T  # (C*fh*fw, out_c)
        out = self.col @ self.col_W + self.b
        H, W = x.shape[2], x.shape[3]
        out_h = (H + 2 * self.pad - self.W.shape[2]) // self.stride + 1
        out_w = (W + 2 * self.pad - self.W.shape[3]) // self.stride + 1
        self.out_shape = (x.shape[0], self.W.shape[0], out_h, out_w)
        return out.reshape(x.shape[0], out_h, out_w, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, out_c, out_h, out_w = dout.shape
        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, out_c)
        db = np.sum(dout_col, axis=0)
        dW = (self.col.T @ dout_col).T   # (out_c, C*fh*fw)
        dW = dW.reshape(self.W_shape)
        dcol = dout_col @ self.col_W.T
        dx = col2im(dcol, self.x.shape, self.W.shape[2], self.W.shape[3], self.stride, self.pad)
        return dx, dW, db


class Pooling:
    def __init__(self, pool_h=2, pool_w=2, stride=2, pad=0):
        self.pool_h, self.pool_w, self.stride, self.pad = pool_h, pool_w, stride, pad
        self.x = self.arg_max = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        oh = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        ow = (W + 2 * self.pad - self.pool_w) // self.stride + 1
        col = col.reshape(N, oh, ow, C, self.pool_h, self.pool_w).transpose(0, 3, 4, 5, 1, 2)
        col = col.reshape(N, C, self.pool_h * self.pool_w, oh * ow)
        self.arg_max = np.argmax(col, axis=2)
        out = np.max(col, axis=2)
        return out.reshape(N, C, oh, ow)

    def backward(self, dout):
        N, C, oh, ow = dout.shape
        dcol = np.zeros((N, C, self.pool_h * self.pool_w, oh * ow), dtype=np.float64)
        dout_flat = dout.reshape(N, C, 1, oh * ow)
        # 用 take_along_axis 路由到 argmax 位置
        for n in range(N):
            for c in range(C):
                for k in range(oh * ow):
                    m = self.arg_max[n, c, k]
                    dcol[n, c, m, k] = dout.reshape(N, C, -1)[n, c, k]
        # (N,C,ph,pw,oh,ow) -> (N,oh,ow,C,ph,pw) -> im2col 的行布局
        dcol = dcol.reshape(N, C, self.pool_h, self.pool_w, oh, ow).transpose(0, 4, 5, 1, 2, 3)
        dcol = dcol.reshape(N * oh * ow, C * self.pool_h * self.pool_w)
        return col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)


class ReLU:
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        d = dout.copy()
        d[self.mask] = 0
        return d


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T, self.dW, self.db


def softmax_loss(z, t):
    c = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - c)
    y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    batch = z.shape[0]
    loss = float(-np.sum(t * np.log(y + 1e-7)) / batch)
    return loss, (y - t) / batch


class SimpleConvNet:
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

    def _forward(self, x):
        p = self.params
        self.layers["conv"] = Convolution(p["W1"], p["b1"], stride=1, pad=1)
        self.layers["relu1"] = ReLU()
        self.layers["pool"] = Pooling(2, 2, 2)
        x = self.layers["conv"].forward(x)
        x = self.layers["relu1"].forward(x)
        x = self.layers["pool"].forward(x)
        self.pooled_shape = x.shape  # (N, C, oh, ow)，供反向 reshape 用
        x = x.reshape(x.shape[0], -1)
        self.layers["aff1"] = Affine(p["W2"], p["b2"])
        self.layers["relu2"] = ReLU()
        self.layers["aff2"] = Affine(p["W3"], p["b3"])
        h = self.layers["relu2"].forward(self.layers["aff1"].forward(x))
        return self.layers["aff2"].forward(h)

    def predict(self, x):
        return self._forward(x)

    def loss(self, x, t):
        z = self._forward(x)
        loss, dz = softmax_loss(z, t)
        self.layers["dout"] = dz
        return loss

    def gradient(self, x, t):
        self.loss(x, t)
        d = self.layers["dout"]
        d = self.layers["aff2"].backward(d)[0]
        d = self.layers["relu2"].backward(d)
        d = self.layers["aff1"].backward(d)[0]
        d = d.reshape(self.pooled_shape)
        d = self.layers["pool"].backward(d)
        d = self.layers["relu1"].backward(d)
        _, dW1, db1 = self.layers["conv"].backward(d)
        return {
            "W1": dW1, "b1": db1,
            "W2": self.layers["aff1"].dW, "b2": self.layers["aff1"].db,
            "W3": self.layers["aff2"].dW, "b3": self.layers["aff2"].db,
        }

    def accuracy(self, x, t):
        y = self.predict(x).argmax(axis=1)
        return float((y == t).mean())


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


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (2, 1, 6, 6))
    col = im2col(x, 2, 2, stride=2, pad=0)
    assert col.shape == (2 * 3 * 3, 4), col.shape
    x2 = col2im(col, x.shape, 2, 2, stride=2, pad=0)
    assert np.allclose(x, x2)

    W = rng.normal(0, 0.3, (3, 1, 3, 3))
    b = rng.normal(0, 0.1, 3)
    x3 = rng.normal(0, 1, (2, 1, 6, 6))
    assert np.allclose(Convolution(W, b, 1, 1).forward(x3), conv2d_naive(x3, W, b, 1, 1), atol=1e-10)

    x4 = rng.normal(0, 1, (1, 1, 4, 4))
    W4 = rng.normal(0, 0.3, (2, 1, 3, 3))
    b4 = rng.normal(0, 0.1, 2)
    c4 = Convolution(W4, b4, 1, 0)
    out4 = c4.forward(x4)
    dout4 = rng.normal(0, 1, out4.shape)
    dX, dW, db = c4.backward(dout4)
    fW = lambda w: np.sum(Convolution(w, b4, 1, 0).forward(x4) * dout4)
    fX = lambda xx: np.sum(Convolution(W4, b4, 1, 0).forward(xx) * dout4)
    assert np.allclose(dW, numerical_gradient(fW, W4), atol=1e-4)
    assert np.allclose(dX, numerical_gradient(fX, x4), atol=1e-4)
    assert np.allclose(db, np.sum(dout4, axis=(0, 2, 3)), atol=1e-10)

    p = Pooling(2, 2, stride=2)
    xp = rng.normal(0, 1, (2, 3, 4, 4))
    yp = p.forward(xp)
    assert yp.shape == (2, 3, 2, 2)
    dp = p.backward(np.ones_like(yp))
    fp = lambda xx: np.sum(Pooling(2, 2, 2).forward(xx))
    assert np.allclose(dp, numerical_gradient(fp, xp), atol=1e-6)

    from common.toy_data import synthetic_images, one_hot

    X, y = synthetic_images(n_per_class=40, size=12, seed=0)
    Xtr, ytr, Xte, yte = X[:90], y[:90], X[90:], y[90:]
    Ttr = one_hot(ytr, 3)
    net = SimpleConvNet(seed=0)
    for _ in range(150):
        g = net.gradient(Xtr, Ttr)
        for k in net.params:
            net.params[k] -= 0.1 * g[k]
    acc = net.accuracy(Xte, yte)
    assert acc >= 0.90, acc
    print(f"[ch07] im2col/卷积/池化/CNN 全部通过；test acc={acc:.3f}")


if __name__ == "__main__":
    main()
