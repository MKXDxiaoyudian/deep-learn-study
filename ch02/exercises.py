"""第 2 章 感知机 —— 练习骨架（简版）

目标：自己给每个门选一组 w1、w2、b，让
        w1*x1 + w2*x2 + b > 0 时输出 1，否则输出 0
的正负号符合真值表。答案不唯一。

要改的地方（只有两处）：
1. AND / NAND / OR 三个函数里的 w1, w2, b（各 3 个数）；
2. XOR 的函数体（用 AND / NAND / OR 组合）。
   perceptron_fit 是拓展题，可选。

运行时每个输入都会打印一行打分，例如：
    AND(0,1): 0.5*0 + 0.5*1 + (-0.7) = -0.20 -> 0
所以填错时能直接看到哪一行不对。

运行：python exercises.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # 让 common/ 可导入


def AND(x1, x2):
    """与门。先自己定 w1、w2、b（答案不唯一，见 00-先读 第 3 节）。"""
    w1, w2, b = None, None, None   # TODO: 改成三个数，例如 0.5, 0.5, -0.7
    if w1 is None:
        raise NotImplementedError("TODO: 先填 AND 的 w1, w2, b（见 00-先读 §3）")
    score = w1 * x1 + w2 * x2 + b
    print(f"AND({x1},{x2}): {w1}*{x1} + {w2}*{x2} + ({b}) = {score:+.2f} -> {int(score > 0)}")
    return int(score > 0)


def NAND(x1, x2):
    """与非门。先自己定 w1、w2、b（见 00-先读 第 4 节）。"""
    w1, w2, b = None, None, None   # TODO: 改成三个数
    if w1 is None:
        raise NotImplementedError("TODO: 先填 NAND 的 w1, w2, b（见 00-先读 §4）")
    score = w1 * x1 + w2 * x2 + b
    print(f"NAND({x1},{x2}): {w1}*{x1} + {w2}*{x2} + ({b}) = {score:+.2f} -> {int(score > 0)}")
    return int(score > 0)


def OR(x1, x2):
    """或门。先自己定 w1、w2、b（见 00-先读 第 4 节）。"""
    w1, w2, b = None, None, None   # TODO: 改成三个数
    if w1 is None:
        raise NotImplementedError("TODO: 先填 OR 的 w1, w2, b（见 00-先读 §4）")
    score = w1 * x1 + w2 * x2 + b
    print(f"OR({x1},{x2}): {w1}*{x1} + {w2}*{x2} + ({b}) = {score:+.2f} -> {int(score > 0)}")
    return int(score > 0)


def XOR(x1, x2):
    """异或门（两层组合）：只允许用 AND / NAND / OR 组合。

    XOR(0,0)=0, XOR(0,1)=1, XOR(1,0)=1, XOR(1,1)=0
    提示（见 00-先读 第 6 节）：s1 = NAND(x1,x2)；s2 = OR(x1,x2)；y = AND(s1,s2)
    """
    raise NotImplementedError("TODO: 用 NAND / OR / AND 组合实现 XOR")


def perceptron_fit(X, y, lr=1.0, epochs=100):
    """拓展题：感知机学习算法（让机器自己学 w、b；书中第 2 章未讲）。

    参数：X (N,2)；y (N,)，标签必须是 +1 / -1。
    返回：w (2,), b。
    伪代码：w=0,b=0；重复 epochs 轮；对每个样本，若 y_i*(w·x_i+b) <= 0：
            w += lr*y_i*x_i；b += lr*y_i；一轮无更新则 break。
    """
    raise NotImplementedError("TODO: 实现感知机原始形式（见上面伪代码）")


def main():
    # 三个门的真值表（每个输入会先打印打分，再断言）
    assert [AND(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 0, 0, 1]
    assert [NAND(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [1, 1, 1, 0]
    assert [OR(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 1, 1, 1]

    # XOR
    assert [XOR(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 1, 1, 0]
    import inspect
    src = inspect.getsource(XOR)
    assert all(name in src for name in ("NAND", "OR", "AND")), "XOR 必须用 NAND/OR/AND 组合实现"

    # 拓展题：感知机学习
    from common.toy_data import two_blobs

    X, y01 = two_blobs(seed=0)
    y = np.where(y01 == 1, 1, -1)
    w, b = perceptron_fit(X, y)
    acc = float((((X @ w + b) > 0) == (y > 0)).mean())
    assert acc >= 0.98, f"感知机训练准确率过低: {acc}"
    print(f"\n[ch02] 全部通过；感知机 acc={acc:.2f}, w={w.round(2)}, b={b:.2f}")


if __name__ == "__main__":
    main()
