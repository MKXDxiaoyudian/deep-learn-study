"""练习用的离线数据集生成器（不依赖网络、不依赖 sklearn）。

对应原书：
- spiral(): 原书 dataset/spiral.py 风格的三分类螺旋数据（备用，当前章节未直接使用）
- two_blobs(): 两分类二维高斯团（更简单的线性可分/近可分数据，ch02/ch04 用）
- synthetic_images(): 12x12 三分类合成图像（ch07 CNN 用；原书用 MNIST，这里离线可跑）

约定：函数都是纯函数，`seed` 固定可复现。
"""
import numpy as np


def one_hot(y, n_class):
    """把 [0,1,2,...] 标签转成 one-hot。"""
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros((y.size, n_class), dtype=np.float64)
    out[np.arange(y.size), y] = 1.0
    return out


def two_blobs(n_per_class=60, seed=0):
    """两分类：两个二维高斯"团"（blob）。

    生成方式（每个类别 60 个点，共 120 个）：
      类别 0：中心 (2, 2)，每个坐标 N(2, 0.7²) —— 右上角一团；
      类别 1：中心 (-2, -2)，每个坐标 N(-2, 0.7²) —— 左下角一团；
    然后把两类拼起来、打乱顺序返回。

    为什么这组数据"可以用一条直线分开"：
      两个中心连线的中垂线是 x1 + x2 = 0；
      中心 (2,2) 到这条线的距离 = |2+2|/√2 ≈ 2.83，而每个坐标的噪声 σ=0.7；
      也就是说"中心离分界线约 4σ"，样本绝大多数不会越过这条线。
      实测（seed=0）：类别 0 的 x1+x2 ∈ [2.08, 6.39]，类别 1 的 x1+x2 ∈ [-6.97, -2.04]，
      中间有空隙 → 直线 x1+x2=0 能 100% 分开（投影到法线方向的最小间距约 1.44）。

    注意：高斯分布是无界的，所以"一定可分"只对**这次抽到的样本**成立（seed 固定）；
    如果中心更近或 σ 更大，两团会重叠，感知机就可能不收敛。
    """
    # 固定随机种子：同一个 seed 每次生成同一批点，便于复现/对答案。
    # （如果每次 seed 不同，感知机学到的 w、b 也会不同——数据变了。）
    rng = np.random.default_rng(seed)

    # 类别 0：均值向量 (2, 2)、每个坐标标准差 0.7 的二维正态。
    # size=(n_per_class, 2) = n 行、2 列：每一行是一个点 (x1, x2)。
    X0 = rng.normal([2.0, 2.0], 0.7, size=(n_per_class, 2))

    # 类别 1：均值向量 (-2, -2)、同样的噪声强度 0.7。
    # 两个均值在原点两侧，且连线的中垂线是 x1 + x2 = 0。
    X1 = rng.normal([-2.0, -2.0], 0.7, size=(n_per_class, 2))

    # vstack 按行拼接：先放 60 个类别 0，再放 60 个类别 1，得到形状 (120, 2)。
    X = np.vstack([X0, X1])

    # 标签：前 60 个是 0，后 60 个是 1（形状 (120,)）。
    y = np.hstack([np.zeros(n_per_class, dtype=np.int64), np.ones(n_per_class, dtype=np.int64)])

    # 打乱行顺序，避免"先全是 0、后全是 1"影响训练。
    # 注意 X 和 y 用同一个 idx，配对关系不会乱。
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def spiral(n_per_class=100, n_class=3, noise=0.2, seed=0):
    """原书 dataset/spiral.py 同款三分类螺旋；返回 X (N,2), y (N,) ∈ {0,1,2}。"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in range(n_class):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(cls * 4.0, (cls + 1) * 4.0, n_per_class) + rng.normal(0.0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, cls, dtype=np.int64))
    X = np.vstack(X)
    y = np.hstack(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def synthetic_images(n_per_class=40, size=12, noise=0.05, seed=0):
    """12x12 三分类合成图像（方框/圆/十字），返回 X (N,1,size,size), y (N,) ∈ {0,1,2}。

    用于 ch07 的 CNN 练习：原书 ch07 用 MNIST（28x28），这里用离线小图代替，
    网络结构与训练流程保持同构（im2col -> conv -> pool -> affine -> softmax）。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    c = (size - 1) / 2.0
    for cls in range(3):
        for _ in range(n_per_class):
            img = np.zeros((size, size), dtype=np.float64)
            if cls == 0:  # 方框边框
                img[2:size - 2, 2] = 1.0
                img[2:size - 2, size - 3] = 1.0
                img[2, 2:size - 2] = 1.0
                img[size - 3, 2:size - 2] = 1.0
            elif cls == 1:  # 圆环
                yy, xx = np.mgrid[0:size, 0:size]
                r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
                img[np.abs(r - (size / 2.0 - 2.5)) < 0.8] = 1.0
            else:  # 十字
                img[size // 2 - 1:size // 2 + 1, 2:size - 2] = 1.0
                img[2:size - 2, size // 2 - 1:size // 2 + 1] = 1.0
            # 随机平移 0~1 像素（近似原书对 MNIST 的预处理思路）
            dx, dy = rng.integers(-1, 2, size=2)
            img = np.roll(np.roll(img, int(dy), axis=0), int(dx), axis=1)
            img = img + rng.normal(0.0, noise, img.shape)
            X.append(img[None, :, :])
            y.append(cls)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
