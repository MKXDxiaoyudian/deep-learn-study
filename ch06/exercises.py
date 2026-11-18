"""第 6 章 与学习相关的技巧 —— 练习骨架

对应原书
- §6.1 参数的更新（印刷 p163–175）：SGD / Momentum / AdaGrad / Adam
- §6.2 权重的初始值（p176–183）：Xavier / He 初始化
- §6.3 Batch Normalization（p184–187）
- §6.4 正则化（p188–194）：Dropout（本书只讲 Dropout 一种）

运行 `python exercises.py` 全过即完成。提示见 ch06/README.md。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class SGD:
    """§6.1.2：w <- w - lr * grad"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        raise NotImplementedError("TODO: SGD.update")


class Momentum:
    """§6.1.4：带动量的随机梯度下降。"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum, self.v = lr, momentum, None

    def update(self, params, grads):
        raise NotImplementedError("TODO: Momentum.update")


class AdaGrad:
    """§6.1.5：AdaGrad（按维度累计梯度平方）。"""
    def __init__(self, lr=0.01):
        self.lr, self.h = lr, None

    def update(self, params, grads):
        raise NotImplementedError("TODO: AdaGrad.update")


class Adam:
    """§6.1.6：Adam（一阶/二阶矩估计 + 偏差修正）。"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.m = self.v = None
        self.t = 0

    def update(self, params, grads):
        raise NotImplementedError("TODO: Adam.update")


def init_scale_compare(n_in=100, n_units=100, n_layers=5, n_samples=200, seed=0):
    """§6.2：比较不同初始化下各层激活值的标准差。

    返回 dict：{"xavier": [...], "he": [...], "small": [...]}（每层激活值的标准差）
    三种尺度见 §6.2；网络用 y = tanh(x @ W)。
    """
    raise NotImplementedError("TODO: 实现三种初始化下的激活值标准差对比")


def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """§6.3：训练模式 BatchNorm 前向。返回 (y, cache)。"""
    raise NotImplementedError("TODO: 实现 BatchNorm 前向")


def batchnorm_backward(dout, cache):
    """§6.3：BatchNorm 反向。返回 (dx, dgamma, dbeta)。"""
    raise NotImplementedError("TODO: 实现 BatchNorm 反向")


def dropout_forward(x, dropout_ratio=0.5, train=True, seed=None):
    """§6.4：inverted dropout。train=True 时按 1/(1-p) 缩放；推理时原样返回。"""
    raise NotImplementedError("TODO: 实现 Dropout 前向")


def main():
    # §6.1 四种优化器：在 f(w)=0.5*(w0^2+10*w1^2) 上比较（梯度=(w0, 10*w1)）
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
    assert report["AdaGrad"][0] < report["SGD"][0], "AdaGrad 应该比纯 SGD 快"

    # §6.2 初始化
    scales = init_scale_compare()
    for k in ("xavier", "he", "small"):
        assert len(scales[k]) == 5
        assert all(np.isfinite(v) for v in scales[k])
    assert scales["small"][-1] < scales["he"][-1], "小初始化应导致激活逐层变小"
    assert scales["he"][-1] > 0.05, "He 初始化不应让激活消失"
    scales3 = init_scale_compare(n_layers=3)
    scales3b = init_scale_compare(n_layers=3, seed=1)
    assert len(scales3["he"]) == 3
    assert not np.allclose(scales3["he"], scales3b["he"])
    assert all(scales["small"][i + 1] < scales["small"][i] for i in range(len(scales["small"]) - 1)), scales["small"]

    # §6.3 BatchNorm 前向 + 反向（数值核对）
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (6, 4))  # 故意均值不为 0
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
    ng = numerical_gradient(f_g, gamma)
    nb = numerical_gradient(f_b, beta)
    nx = numerical_gradient(f_x, x)
    assert np.allclose(dgamma, ng, atol=1e-4), (dgamma, ng)
    assert np.allclose(dbeta, nb, atol=1e-4), (dbeta, nb)
    assert np.allclose(dx, nx, atol=1e-4), (dx, nx)

    # §6.4 Dropout
    x2 = np.ones((1000, 100))
    out_train = dropout_forward(x2, dropout_ratio=0.3, train=True, seed=0)
    assert abs(float((out_train == 0).mean()) - 0.3) < 0.02
    assert abs(float(out_train.mean()) - 1.0) < 0.05  # inverted dropout 保持期望
    out_test = dropout_forward(x2, dropout_ratio=0.3, train=False, seed=0)
    assert np.array_equal(out_test, x2)

    print("[ch06] 优化器/初始化/BatchNorm/Dropout 全部通过；"
          f"loss={ {k: round(v[0], 5) for k, v in report.items()} }；"
          f"he_last_std={scales['he'][-1]:.3f}")


def numerical_gradient(f, x):
    """数值梯度（ch04/ch05 用过；这里直接给出，方便核对）。"""
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
