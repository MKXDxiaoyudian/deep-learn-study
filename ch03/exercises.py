"""第 3 章 神经网络 —— 练习骨架

对应原书
- §3.2 激活函数（印刷 p42–49）
- §3.3 多维数组的运算（p50–55）
- §3.4 3 层神经网络的实现（p56–62）
- §3.5 输出层的设计 / softmax（p63–68）

运行 `python exercises.py` 全过即完成。提示见 ch03/README.md。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def step_function(x):
    """阶跃函数（书中 §3.2.2）：x > 0 返回 1，否则 0；支持 NumPy 数组，返回 int 数组。"""
    x = np.asarray(x, dtype=np.float64)
    y = x > 0#笔记：要用numpy写成逐元素比较然后输出结果，不然就只是标量会报错
    return y.astype(np.int64)
    
    raise NotImplementedError("TODO: 实现阶跃函数")


def sigmoid(x):
    """Sigmoid（书中 §3.2.4）：1 / (1 + exp(-x))。"""
    x = np.asarray(x, dtype=np.float64)
    return 1.0/(np.exp(-x) + 1.0)#笔记：不要使用float强制转换，这里输入数组的时候就会报错
    raise NotImplementedError("TODO: 实现 sigmoid")


def relu(x):
    """ReLU（书中 §3.2.7）：max(0, x)。"""
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, x)
    raise NotImplementedError("TODO: 实现 ReLU")


def forward_3layer(x, W1, b1, W2, b2, W3, b3):
    """3 层前向（书中 §3.4）：前两层 sigmoid，输出层恒等函数。
    

    参数
    - x: (d0,) 或 (N, d0)
    返回
    - (d3,) 或 (N, d3) 的输出 a3
    """
    x = np.asarray(x, dtype=np.float64)
    x1 = sigmoid(x @ W1 + b1 )
    x2 = sigmoid(x1 @ W2 + b2 )
    return x2 @ W3 +b3 # 注意矩阵乘法具有方向 x2 @ W3是合法的，反过来在数学上不合法会报错
    raise NotImplementedError("TODO: 用 sigmoid 和矩阵乘法实现 3 层前向")


def softmax(a):
    """softmax（书中 §3.5.1–§3.5.2）：注意减去最大值防止 exp 溢出。

    参数 a: (K,)；返回同形状、和为 1 的概率向量。
    """
    a = np.asarray(a, dtype=np.float64)
    exp_a = np.exp(a - np.max(a))#要减去最大值避免数据溢出导致结果混乱
    a1 = exp_a /np.sum(exp_a)
    return a1
    raise NotImplementedError("TODO: 实现数值稳定的 softmax")


def main():
    # §3.2 激活函数
    xs = np.array([-1.0, 0.0, 1.0])
    assert np.array_equal(step_function(xs), np.array([0, 0, 1]))
    assert abs(float(sigmoid(0.0)) - 0.5) < 1e-12
    assert abs(float(sigmoid(1.0)) - 1/(1+np.exp(-1))) < 1e-12
    assert abs(float(sigmoid(-1.0)) - 1/(1+np.exp(1))) < 1e-12
    assert np.allclose(relu(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 2.0])

    # §3.3–§3.4 三层前向（固定参数，便于核对）
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
    # 手算核对：a1=sigmoid(xW1+b1), a2=sigmoid(a1W2+b2), a3=a2W3+b3
    a1 = sigmoid(x @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    assert np.allclose(y, a2 @ W3 + b3, atol=1e-12)
    # 批处理形状
    X = rng.normal(0, 1, (8, 4))
    assert forward_3layer(X, W1, b1, W2, b2, W3, b3).shape == (8, 2)

    # §3.5 softmax
    a = np.array([0.3, 2.9, 4.0])
    p = softmax(a)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    # 数值稳定：大数不能溢出，且与平移后结果一致
    big = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(big))
    assert np.allclose(big, softmax(np.array([0.0, 1.0, 2.0])))
    print(f"[ch03] 激活函数/三层前向/softmax 全部通过；softmax(大数)={big.round(4)}")


if __name__ == "__main__":
    main()
