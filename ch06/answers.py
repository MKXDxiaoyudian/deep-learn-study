"""第 6 章 与学习相关的技巧 —— 参考答案。运行：python answers.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum, self.v = lr, momentum, None

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr, self.h = lr, None

    def update(self, params, grads):
        if self.h is None:
            self.h = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.m = self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


def init_scale_compare(n_in=100, n_units=100, n_layers=5, n_samples=200, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, (n_samples, n_in))
    out = {}
    for name in ("xavier", "he", "small"):
        # 每个方法用独立但同 seed 的随机流（保证可比：只有 std 不同）
        r = np.random.default_rng(seed)
        x = x0.copy()
        stds = []
        for _ in range(n_layers):
            fan_in = x.shape[1]  # 逐层 fan-in（不是全局 n_in）
            if name == "xavier":
                std = 1.0 / np.sqrt(fan_in)
            elif name == "he":
                std = np.sqrt(2.0 / fan_in)
            else:
                std = 0.01
            W = r.normal(0, std, (fan_in, n_units))
            x = np.tanh(x @ W)
            stds.append(float(np.std(x)))
        out[name] = stds
    return out


def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    xhat = (x - mu) / np.sqrt(var + eps)
    y = gamma * xhat + beta
    return y, (x, xhat, gamma, eps, mu, var)


def batchnorm_backward(dout, cache):
    x, xhat, gamma, eps, mu, var = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xhat, axis=0)
    dxhat = dout * gamma
    dvar = np.sum(dxhat * (x - mu) * -0.5 * (var + eps) ** (-1.5), axis=0)
    dmu = np.sum(-dxhat / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2.0 * (x - mu), axis=0) / N
    dx = dxhat / np.sqrt(var + eps) + dvar * 2.0 * (x - mu) / N + dmu / N
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_ratio=0.5, train=True, seed=None):
    if not train:
        return x
    rng = np.random.default_rng(seed)
    mask = (rng.random(x.shape) >= dropout_ratio).astype(np.float64)
    return x * mask / (1.0 - dropout_ratio)


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
    def loss(w):
        return 0.5 * (w[0] ** 2 + 10.0 * w[1] ** 2)

    def grad(w):
        return np.array([w[0], 10.0 * w[1]])

    configs = [
        ("SGD", SGD(lr=0.01), 3000),
        ("Momentum", Momentum(lr=0.01), 800),
        ("AdaGrad", AdaGrad(lr=0.5), 800),
        ("Adam", Adam(lr=0.1), 600),
    ]
    report = {}
    for name, opt, steps in configs:
        w = [np.array([3.0, 4.0])]
        for _ in range(steps):
            opt.update(w, [grad(w[0])])
        report[name] = (float(loss(w[0])), steps)
        assert loss(w[0]) < 1e-2, (name, loss(w[0]))
    assert report["AdaGrad"][0] < report["SGD"][0]

    scales = init_scale_compare()
    for k in ("xavier", "he", "small"):
        assert len(scales[k]) == 5 and all(np.isfinite(v) for v in scales[k])
    assert scales["small"][-1] < scales["he"][-1]
    assert scales["he"][-1] > 0.05
    scales3 = init_scale_compare(n_layers=3)
    scales3b = init_scale_compare(n_layers=3, seed=1)
    assert len(scales3["he"]) == 3
    assert not np.allclose(scales3["he"], scales3b["he"])
    assert all(scales["small"][i + 1] < scales["small"][i] for i in range(len(scales["small"]) - 1)), scales["small"]

    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (6, 4))
    gamma = rng.normal(1.0, 0.2, 4)
    beta = rng.normal(0.0, 0.2, 4)
    y, cache = batchnorm_forward(x, gamma, beta)
    assert np.allclose(y.mean(axis=0), beta, atol=1e-8)
    assert np.allclose(y.var(axis=0), gamma ** 2, atol=1e-4)
    dout = rng.normal(0, 1, y.shape)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    f_g = lambda g: np.sum(batchnorm_forward(x, g, beta)[0] * dout)
    f_b = lambda b: np.sum(batchnorm_forward(x, gamma, b)[0] * dout)
    f_x = lambda xx: np.sum(batchnorm_forward(xx, gamma, beta)[0] * dout)
    assert np.allclose(dgamma, numerical_gradient(f_g, gamma), atol=1e-4)
    assert np.allclose(dbeta, numerical_gradient(f_b, beta), atol=1e-4)
    assert np.allclose(dx, numerical_gradient(f_x, x), atol=1e-4)

    x2 = np.ones((1000, 100))
    out_train = dropout_forward(x2, dropout_ratio=0.3, train=True, seed=0)
    assert abs(float((out_train == 0).mean()) - 0.3) < 0.02
    assert abs(float(out_train.mean()) - 1.0) < 0.05
    assert np.array_equal(dropout_forward(x2, dropout_ratio=0.3, train=False, seed=0), x2)

    print("[ch06] 优化器/初始化/BatchNorm/Dropout 全部通过；"
          f"loss={ {k: round(v[0], 5) for k, v in report.items()} }；"
          f"he_last_std={scales['he'][-1]:.3f}")


if __name__ == "__main__":
    main()
