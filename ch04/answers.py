"""第 4 章 神经网络的学习 —— 参考答案。运行：python answers.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def mean_squared_error(y, t):
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if y.ndim == 1:
        return float(0.5 * np.sum((y - t) ** 2))
    return float(0.5 * np.sum((y - t) ** 2) / y.shape[0])


def cross_entropy_error(y, t):
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if y.ndim == 1:
        y, t = y.reshape(1, -1), t.reshape(1, -1)
    batch = y.shape[0]
    return float(-np.sum(t * np.log(y + 1e-7)) / batch)


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


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


def softmax_batch(a):
    a = np.asarray(a, dtype=np.float64)
    c = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _unpack(p):
    return p["W1"], p["b1"], p["W2"], p["b2"]


def _init_params(n_in, n_hidden, n_out, rng):
    return {
        "W1": rng.normal(0, 0.5, (n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, 0.5, (n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def train_two_layer_numeric(X, y, n_hidden=4, lr=0.5, iters=200):
    rng = np.random.default_rng(0)
    params = _init_params(X.shape[1], n_hidden, 2, rng)
    y_onehot = np.eye(2)[y]

    def loss_fn(flat):
        W1, b1, W2, b2 = (
            flat[:2 * n_hidden].reshape(2, n_hidden),
            flat[2 * n_hidden:2 * n_hidden + n_hidden],
            flat[2 * n_hidden + n_hidden:2 * n_hidden + n_hidden + n_hidden * 2].reshape(n_hidden, 2),
            flat[2 * n_hidden + n_hidden + n_hidden * 2:],
        )
        probs = softmax_batch(_sigmoid(X @ W1 + b1) @ W2 + b2)
        return cross_entropy_error(probs, y_onehot)

    def flatten(p):
        return np.concatenate([p["W1"].ravel(), p["b1"].ravel(), p["W2"].ravel(), p["b2"].ravel()])

    flat = flatten(params)
    losses = []
    for _ in range(iters):
        grad = numerical_gradient(loss_fn, flat)
        flat = flat - lr * grad
        losses.append(loss_fn(flat))
    W1, b1 = flat[:2 * n_hidden].reshape(2, n_hidden), flat[2 * n_hidden:2 * n_hidden + n_hidden]
    W2 = flat[2 * n_hidden + n_hidden:2 * n_hidden + n_hidden + n_hidden * 2].reshape(n_hidden, 2)
    b2 = flat[2 * n_hidden + n_hidden + n_hidden * 2:]
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}, losses


def main():
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert abs(mean_squared_error(y, t) - 0.0975) < 1e-9
    assert abs(cross_entropy_error(y, t) - 0.510825457099338) < 1e-6
    yb = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    tb = np.array([[1, 0, 0], [0, 1, 0]])
    assert abs(cross_entropy_error(yb, tb) - 0.2899092476264711) < 1e-6

    f = lambda x: x ** 2
    assert abs(numerical_diff(f, 3.0) - 6.0) < 1e-6
    g = lambda x: np.sin(x)
    assert abs(numerical_diff(g, np.pi / 4) - np.cos(np.pi / 4)) < 1e-6

    f2 = lambda v: v[0] ** 2 + v[1] ** 2
    grad = numerical_gradient(f2, np.array([3.0, 4.0]))
    assert np.allclose(grad, [6.0, 8.0], atol=1e-5)

    # softmax_batch：每行和为 1；大 logit 不溢出（减最大值的意义）
    sb = softmax_batch(np.array([[0.0, 1.0, 2.0], [1000.0, 1001.0, 1002.0]]))
    assert np.allclose(sb.sum(axis=1), 1.0)
    assert np.all(np.isfinite(sb))

    from common.toy_data import two_blobs

    X, y01 = two_blobs(seed=0)
    params, losses = train_two_layer_numeric(X, y01)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    probs = softmax_batch(_sigmoid(X @ W1 + b1) @ W2 + b2)
    acc = float((probs.argmax(axis=1) == y01).mean())
    assert acc >= 0.90, acc
    print(f"[ch04] 损失/数值微分/梯度 OK；训练 loss {losses[0]:.3f} -> {losses[-1]:.3f}, acc={acc:.2f}")


if __name__ == "__main__":
    main()
