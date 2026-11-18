"""第 5 章 误差反向传播法 —— 参考答案。运行：python answers.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class MulLayer:
    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, dout):
        return dout * self.y, dout * self.x  # dout*y -> x 的梯度；dout*x -> y 的梯度


class AddLayer:
    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


class Relu:
    def forward(self, x):
        self.mask = np.asarray(x) <= 0
        out = np.asarray(x).copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout = np.asarray(dout).copy()
        dout[self.mask] = 0
        return dout


class Sigmoid:
    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-np.asarray(x)))
        return self.y

    def backward(self, dout):
        return dout * (1.0 - self.y) * self.y


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.x = self.dW = self.db = None

    def forward(self, x):
        self.x = np.asarray(x)
        return self.x @ self.W + self.b

    def backward(self, dout):
        dX = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dX, self.dW, self.db


class SoftmaxWithLoss:
    def forward(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        c = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - c)
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        batch = x.shape[0]
        self.loss = float(-np.sum(self.t * np.log(self.y + 1e-7)) / batch)
        return self.loss

    def backward(self, dout=1):
        return (self.y - self.t) / self.t.shape[0]


class TwoLayerNet:
    def __init__(self, input_size=2, hidden_size=8, output_size=2, weight_init_std=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, weight_init_std, (input_size, hidden_size)),
            "b1": np.zeros(hidden_size),
            "W2": rng.normal(0, weight_init_std, (hidden_size, output_size)),
            "b2": np.zeros(output_size),
        }
        self.layers = {}

    def _forward(self, x):
        p = self.params
        self.layers.update({"a1": Affine(p["W1"], p["b1"]), "r1": Relu(),
                            "a2": Affine(p["W2"], p["b2"])})
        a = self.layers["a1"].forward(x)
        r = self.layers["r1"].forward(a)
        return self.layers["a2"].forward(r)

    def predict(self, x):
        return self._forward(np.asarray(x))

    def loss(self, x, t):
        z = self._forward(x)
        self.layers["last"] = SoftmaxWithLoss()
        return self.layers["last"].forward(z, t)

    def accuracy(self, x, t):
        y = self.predict(x).argmax(axis=1)
        t = np.asarray(t)
        tt = t.argmax(axis=1) if t.ndim > 1 else t
        return float((y == tt).mean())

    def gradient(self, x, t):
        # forward + loss（复用 _forward 里缓存的层）
        self.loss(x, t)
        d = self.layers["last"].backward()
        d = self.layers["a2"].backward(d)[0]   # 只取 dX
        d = self.layers["r1"].backward(d)
        _, dW1, db1 = self.layers["a1"].backward(d)
        return {
            "W1": dW1, "b1": db1,
            "W2": self.layers["a2"].dW, "b2": self.layers["a2"].db,
        }


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
    # §5.3.3 / §5.4 苹果
    mul_apple, mul_tax = MulLayer(), MulLayer()
    apple_price = mul_apple.forward(100, 2)
    price = mul_tax.forward(apple_price, 1.1)
    assert abs(price - 220) < 1e-9
    d_apple_total, d_tax = mul_tax.backward(1)
    d_apple_price, d_apple_num = mul_apple.backward(d_apple_total)
    assert np.allclose([d_apple_price, d_apple_num, d_tax], [2.2, 110.0, 200.0], atol=1e-9)

    # 苹果 + 橘子（加法层）
    mul_a, mul_o, mul_t = MulLayer(), MulLayer(), MulLayer()
    add = AddLayer()
    total = add.forward(mul_a.forward(100, 2), mul_o.forward(150, 3))
    with_tax = mul_t.forward(total, 1.1)
    assert abs(with_tax - 715) < 1e-9
    d_total, d_tax = mul_t.backward(1)   # d_total=1.1（上游梯度），d_tax=650
    assert abs(d_total - 1.1) < 1e-9 and abs(d_tax - 650.0) < 1e-9
    d_a, d_o = add.backward(d_total)
    dp, dn = mul_a.backward(d_a)
    op, on = mul_o.backward(d_o)
    assert np.allclose([dp, dn, op, on], [2.2, 110.0, 3.3, 165.0], atol=1e-9)

    # ReLU / Sigmoid
    x = np.array([[-1.0, 0.0, 2.0]])
    relu = Relu()
    assert np.allclose(relu.forward(x), [[0, 0, 2]])
    assert np.allclose(relu.backward(np.ones_like(x)), [[0, 0, 1]])
    sig = Sigmoid()
    assert np.allclose(sig.forward(np.array([0.0, 1.0])), [0.5, 1 / (1 + np.exp(-1))])
    xs = np.array([[-0.5, 0.3]])
    dout = np.array([[1.0, 2.0]])
    s = Sigmoid()
    _ = s.forward(xs)
    dx = s.backward(dout)
    f = lambda z: np.sum(Sigmoid().forward(z) * dout)
    assert np.allclose(dx, numerical_gradient(f, xs), atol=1e-5)

    # Affine
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.5, (3, 2))
    b = rng.normal(0, 0.1, 2)
    aff = Affine(W, b)
    xb = rng.normal(0, 1, (4, 3))
    assert np.allclose(aff.forward(xb), xb @ W + b)
    dout2 = rng.normal(0, 1, (4, 2))
    dx, dW, db = aff.backward(dout2)
    f_w = lambda wflat: np.sum((xb @ wflat.reshape(3, 2) + b) * dout2)
    f_b = lambda bb: np.sum((xb @ W + bb) * dout2)
    assert np.allclose(dW, numerical_gradient(f_w, W.ravel()).reshape(3, 2), atol=1e-5)
    assert np.allclose(db, numerical_gradient(f_b, b), atol=1e-5)

    # SoftmaxWithLoss
    x3 = np.array([[0.1, 0.2, 0.7]])
    t3 = np.array([[0, 0, 1]])
    sm = SoftmaxWithLoss()
    _ = sm.forward(x3, t3)
    y3 = np.exp(x3) / np.exp(x3).sum(axis=1, keepdims=True)
    assert np.allclose(sm.backward(), (y3 - t3) / 1.0, atol=1e-9)

    # 批量版：backward 必须是 (y - t)/batch，而不是单样本的 (y - t)
    xb = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    tb = np.array([[0, 0, 1], [1, 0, 0]])
    smb = SoftmaxWithLoss()
    _ = smb.forward(xb, tb)
    yb = np.exp(xb) / np.exp(xb).sum(axis=1, keepdims=True)
    assert np.allclose(smb.backward(), (yb - tb) / 2.0, atol=1e-12)

    # 两层网络
    from common.toy_data import two_blobs

    X, y01 = two_blobs(seed=0)
    T = np.eye(2)[y01]
    net = TwoLayerNet()

    def flat(p):
        return np.concatenate([p["W1"].ravel(), p["b1"].ravel(), p["W2"].ravel(), p["b2"].ravel()])

    def unflat(fl):
        return {"W1": fl[:16].reshape(2, 8), "b1": fl[16:24],
                "W2": fl[24:40].reshape(8, 2), "b2": fl[40:42]}

    def loss_flat(fl):
        net.params = unflat(fl)
        return net.loss(X, T)

    net.params = unflat(flat(net.params))
    grads = net.gradient(X, T)
    num = numerical_gradient(loss_flat, flat(net.params))
    ana = flat(grads)
    max_abs = float(np.max(np.abs(ana - num)))
    assert np.allclose(ana, num, atol=1e-4, rtol=1e-4), f"max_abs={max_abs}"
    for _ in range(300):
        g = net.gradient(X, T)
        for k in net.params:
            net.params[k] -= 0.5 * g[k]
    acc = net.accuracy(X, y01)
    assert acc >= 0.95, acc
    print(f"[ch05] 苹果/层/网络 全部通过；gradient-check max_abs={max_abs:.2e}, train acc={acc:.2f}")


if __name__ == "__main__":
    main()
