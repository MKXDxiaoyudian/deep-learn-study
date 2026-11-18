"""第 2 章 感知机 —— 参考答案（与 exercises.py 同接口）。

运行：python answers.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def AND(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7   # w·x = 0/0.5/0.5/1；b=-0.7 把阈值卡在 (0.5, 1]
    score = w1 * x1 + w2 * x2 + b
    print(f"AND({x1},{x2}): {w1}*{x1} + {w2}*{x2} + ({b}) = {score:+.2f} -> {int(score > 0)}")
    return int(score > 0)


def NAND(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7  # 把 AND 的 w、b 同时取反
    score = w1 * x1 + w2 * x2 + b
    print(f"NAND({x1},{x2}): {w1}*{x1} + {w2}*{x2} + ({b}) = {score:+.2f} -> {int(score > 0)}")
    return int(score > 0)


def OR(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.2   # 门槛比 AND 低
    score = w1 * x1 + w2 * x2 + b
    print(f"OR({x1},{x2}): {w1}*{x1} + {w2}*{x2} + ({b}) = {score:+.2f} -> {int(score > 0)}")
    return int(score > 0)


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def perceptron_fit(X, y, lr=1.0, epochs=100):
    """感知机学习算法（原始形式）。

    参数
    - X: (N, 2) 的特征，可以是 list / np.ndarray
    - y: (N,) 标签，必须是 +1 / -1
    - lr: 学习率（每次更新走多大一步）
    - epochs: 最多把所有样本扫多少轮
    返回
    - w: (2,) 学到的权重；b: 标量偏置
    """

    # ---- 第 1 步：把输入统一转成 NumPy 数组 ------------------------------
    # np.asarray(..., dtype=np.float64)：
    #   - 如果 X 是 Python list（如 [[1,2],[3,4]]），转成 np.ndarray；
    #   - dtype=np.float64 表示"每个元素用 64 位浮点数存"；
    #     不写 dtype 的话，整数 list 会变成 int 数组，后面和 lr(浮点) 运算时
    #     类型规则更绕，显式转成 float 最省心；
    #   - 如果 X 已经是 float64 的 ndarray，asarray 不会复制，直接返回它本身。
    X = np.asarray(X, dtype=np.float64)   # 形状变成 (N, 2)
    y = np.asarray(y, dtype=np.float64)   # 形状变成 (N,)

    # ---- 第 2 步：初始化 w、b -------------------------------------------
    # X.shape 是一个元组，例如 (120, 2)：
    #   X.shape[0] = 120 = 样本数 N
    #   X.shape[1] = 2   = 每个样本的特征数（这里是 x1、x2 两个）
    # np.zeros(2) 生成 [0., 0.]，即 w1=0、w2=0。
    # 这里用 X.shape[1] 而不是写死 2：以后换成 3 维特征也不用改代码。
    w = np.zeros(X.shape[1], dtype=np.float64)   # w = [0., 0.]
    b = 0.0                                       # 偏置是标量，不是数组

    # ---- 第 3 步：最多扫 epochs 轮 ---------------------------------------
    for _ in range(epochs):          # "_" 表示"这个循环变量用不到"，只数轮数
        updated = False              # 本轮有没有更新过？用来判断是否已收敛

        # zip(X, y)：把 X 的每一行和 y 的每一个元素"配对"取出来。
        #   - X 是 2D，迭代它得到每一行 xi，形状 (2,)；
        #   - y 是 1D，迭代它得到标量 yi；
        #   - 两者按顺序一一对应（所以如果以后打乱数据，必须 X、y 一起打乱）。
        for xi, yi in zip(X, y):

            # w @ xi：@ 是矩阵乘法；两个一维向量时就是点积：
            #         w @ xi = w1*x1 + w2*x2（得到一个标量）
            # + b    ：加上偏置，得到"打分" score
            # yi * score：标签是 ±1 时，这个乘积的正负号能判断对错：
            #   - 分类正确：yi 和 score 同号 → 乘积 > 0
            #   - 分类错误：异号 → 乘积 < 0
            #   - 正好等于 0（落在分界线上）也按"错"处理 → 用 <= 0
            # 注意：如果标签是 0/1（不是 ±1），这个判据就不成立。
            if yi * (w @ xi + b) <= 0.0:      # 误分类

                # 更新 w：xi 是 (2,) 向量，yi 是 ±1，lr 是标量。
                #   - 若 yi=+1 且分错：w ← w + lr*xi，把打分往大的方向推；
                #   - 若 yi=-1 且分错：w ← w - lr*xi，把打分往小的方向推。
                # 为什么方向是 yi*xi（而不是只有 yi）？
                #   打分 s = w·x 对每个 w_j 的依赖是 w_j*x_j：
                #   要抬高这个样本的 s，w_j 就该按 x_j 的大小去改；
                #   若某个特征 x_j=0，它对这个样本没贡献，w_j 就不用动。
                # 更新后这个样本的新打分：
                #   s' = (w + lr*yi*xi)·xi + (b + lr*yi)
                #      = s + lr*yi*(xi·xi) + lr*yi
                #      = s + lr*yi*(||xi||² + 1)
                # 两边乘 yi（且 yi²=1）：
                #   yi*s' = yi*s + lr*(||xi||² + 1) > yi*s
                # 所以"这个样本的正确方向分数"一定被推大了一点；
                # 但这一步可能弄坏别的样本，所以要一轮一轮反复扫。
                w = w + lr * yi * xi

                # 更新 b：偏置没有对应的特征，所以方向就是 yi（步长 lr*yi），
                # 作用是把整条分界线平移一点。
                b = b + lr * yi

                # 换个角度（梯度下降）——和上面手推完全一致：
                #   把感知机损失写成 L = max(0, -yi*(w·xi+b))；
                #   - 分对时 yi*(w·xi+b) > 0，L=0，没有梯度，不更新；
                #   - 分错时 L = -yi*(w·xi+b)，于是
                #       ∂L/∂w = -yi*xi
                #       ∂L/∂b = -yi
                #   梯度下降是“往负梯度方向走”：
                #       w ← w - lr*(∂L/∂w) = w + lr*yi*xi
                #       b ← b - lr*(∂L/∂b) = b + lr*yi
                #   所以感知机更新 = 对感知机损失做随机梯度下降（SGD），
                #   每次只看一个分错的样本，用它的梯度走一步。
                updated = True                # 本轮发生过更新

        # 一整轮都没有任何样本被分错 → 已收敛，提前结束
        if not updated:
            break

    return w, b


def main():
    assert [AND(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 0, 0, 1]
    assert [NAND(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [1, 1, 1, 0]
    assert [OR(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 1, 1, 1]
    assert [XOR(*p) for p in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 1, 1, 0]
    import inspect
    src = inspect.getsource(XOR)
    assert all(name in src for name in ("NAND", "OR", "AND")), "XOR 必须用 NAND/OR/AND 组合实现"

    from common.toy_data import two_blobs

    X, y01 = two_blobs(seed=0)
    y = np.where(y01 == 1, 1, -1)
    w, b = perceptron_fit(X, y)
    acc = float((((X @ w + b) > 0) == (y > 0)).mean())
    assert acc >= 0.98, f"感知机训练准确率过低: {acc}"
    print(f"\n[ch02] 全部通过；感知机 acc={acc:.2f}, w={w.round(2)}, b={b:.2f}")


if __name__ == "__main__":
    main()
