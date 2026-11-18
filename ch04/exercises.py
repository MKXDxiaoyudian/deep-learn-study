"""第 4 章 神经网络的学习 —— 练习骨架

对应原书
- §4.2 损失函数（印刷 p85–93）：均方误差、交叉熵误差、mini-batch
- §4.3 数值微分（p94–100）
- §4.4 梯度（p100–109）
- §4.5 学习算法的实现（p109–118）：2 层网络 + 数值梯度训练

运行 `python exercises.py` 全过即完成。提示见 ch04/README.md。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def mean_squared_error(y, t):
    """均方误差（§4.2.1）：0.5 * sum((y-t)^2)，y/t 为 one-hot（1D 或批量）。"""
    raise NotImplementedError("TODO: 实现均方误差")


def cross_entropy_error(y, t):
    """交叉熵误差（§4.2.2–§4.2.4）：-sum(t*log(y)) / batch；用 1e-7 防 log(0)。"""
    raise NotImplementedError("TODO: 实现交叉熵误差")


def numerical_diff(f, x):
    """中心差分（§4.3）：(f(x+h)-f(x-h)) / (2h)，h=1e-4。"""
    raise NotImplementedError("TODO: 实现中心差分")


def numerical_gradient(f, x):
    """数值梯度（§4.4）：对 x 的每个元素做中心差分，返回与 x 同形状的梯度。"""
    raise NotImplementedError("TODO: 实现数值梯度")


def softmax_batch(a):
    """批量 softmax：a (N,K)，按行减最大值后归一化（ch03 单样本版的推广）。"""
    raise NotImplementedError("TODO: 实现批量 softmax")


def train_two_layer_numeric(X, y, n_hidden=4, lr=0.5, iters=200):
    """用**数值梯度**训练一个 2 层网络（对应 §4.5 的 mini-batch 数值梯度版）。

    X: (N,2)；y: (N,) 类别 0/1；返回 (params, losses)
    - 网络：2 -> n_hidden (sigmoid) -> 2 (softmax)
    - 损失：mini-batch 交叉熵（这里用全量，N 很小）
    """
    raise NotImplementedError("TODO: 实现数值梯度训练")


def main():
    # §4.2 损失函数
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert abs(mean_squared_error(y, t) - 0.0975) < 1e-9, mean_squared_error(y, t)
    assert abs(cross_entropy_error(y, t) - 0.510825457099338) < 1e-6
    # 批量
    yb = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    tb = np.array([[1, 0, 0], [0, 1, 0]])
    assert abs(cross_entropy_error(yb, tb) - 0.2899092476264711) < 1e-6

    # §4.3 数值微分
    f = lambda x: x ** 2
    assert abs(numerical_diff(f, 3.0) - 6.0) < 1e-6
    g = lambda x: np.sin(x)
    assert abs(numerical_diff(g, np.pi / 4) - np.cos(np.pi / 4)) < 1e-6

    # §4.4 梯度
    f2 = lambda v: v[0] ** 2 + v[1] ** 2
    grad = numerical_gradient(f2, np.array([3.0, 4.0]))
    assert np.allclose(grad, [6.0, 8.0], atol=1e-5), grad

    # softmax_batch：每行和为 1；大 logit 不溢出（减最大值的意义）
    sb = softmax_batch(np.array([[0.0, 1.0, 2.0], [1000.0, 1001.0, 1002.0]]))
    assert np.allclose(sb.sum(axis=1), 1.0)
    assert np.all(np.isfinite(sb))

    # §4.5 数值梯度训练
    from common.toy_data import two_blobs

    X, y01 = two_blobs(seed=0)
    params, losses = train_two_layer_numeric(X, y01)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    probs = softmax_batch(1.0 / (1.0 + np.exp(-(X @ W1 + b1))) @ W2 + b2)
    acc = float((probs.argmax(axis=1) == y01).mean())
    assert acc >= 0.90, acc
    print(f"[ch04] 损失/数值微分/梯度 OK；训练 loss {losses[0]:.3f} -> {losses[-1]:.3f}, acc={acc:.2f}")


if __name__ == "__main__":
    main()
