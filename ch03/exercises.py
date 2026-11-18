"""第 3 章 神经网络 —— 参考答案。运行：python answers.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def step_function(x):
    x = np.asarray(x, dtype=np.float64)
    return (x > 0.0).astype(np.int64)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, x)


def forward_3layer(x, W1, b1, W2, b2, W3, b3):
    a1 = sigmoid(np.asarray(x, dtype=np.float64) @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    a3 = a2 @ W3 + b3  # 输出层：恒等函数（回归/输出 σ 由 3.5.1 规定）
    return a3


def softmax(a):
    a = np.asarray(a, dtype=np.float64)
    c = np.max(a)          # 平移最大值：防 exp 溢出
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


def main():
    xs = np.array([-1.0, 0.0, 1.0])
    assert np.array_equal(step_function(xs), np.array([0, 0, 1]))
    assert abs(float(sigmoid(0.0)) - 0.5) < 1e-12
    assert abs(float(sigmoid(1.0)) - 1/(1+np.exp(-1))) < 1e-12
    assert abs(float(sigmoid(-1.0)) - 1/(1+np.exp(1))) < 1e-12
    assert np.allclose(relu(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 2.0])

    rng = np.random.default_rng(0)
    sizes = [4, 3, 2, 2]
    W1 = rng.normal(0, 0.5, (sizes[0], sizes[1]))
    b1 = rng.normal(0, 0.1, sizes[1])
    W2 = rng.normal(0, 0.5, (sizes[1], sizes[2]))
    b2 = rng.normal(0, 0.1, sizes[2])
    W3 = rng.normal(0, 0.5, (sizes[2], sizes[3]))
    b3 = rng.normal(0, 0.1, sizes[3])
    x = np.array([1.0, 0.5, -1.0, 0.0])
    y = forward_3layer(x, W1, b1, W2, b2, W3, b3)
    assert y.shape == (2,), y.shape
    a1 = sigmoid(x @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    assert np.allclose(y, a2 @ W3 + b3, atol=1e-12)
    X = rng.normal(0, 1, (8, 4))
    assert forward_3layer(X, W1, b1, W2, b2, W3, b3).shape == (8, 2)

    a = np.array([0.3, 2.9, 4.0])
    p = softmax(a)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    big = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(big))
    assert np.allclose(big, softmax(np.array([0.0, 1.0, 2.0])))
    print(f"[ch03] 激活函数/三层前向/softmax 全部通过；softmax(大数)={big.round(4)}")


if __name__ == "__main__":
    main()
