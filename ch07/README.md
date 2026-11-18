# 第 7 章 卷积神经网络 —— 代码练习

> **对应原书**：第 7 章（印刷 p201–233）｜**知识点**：§7.2 卷积运算、§7.3 池化、§7.4 im2col 实现、§7.5 SimpleConvNet
> **建议用时**：6–8 h（寒假 W6）｜**和端侧部署的关系**：这一章直接对应 TinyML / MCU 上的卷积实现与内存预算。

## 任务

| # | 内容 | 对应原文 | 要求 | 验收 |
|---|---|---|---|---|
| 1 | `im2col` / `col2im` | §7.4.1–§7.4.2（p216–218） | 4D 图像展开/还原 | stride=filter 时严格互逆；shape 符合 `(N*oh*ow, C*fh*fw)` |
| 2 | `conv2d_naive` vs `Convolution` | §7.2–§7.4（p202–223） | 朴素卷积 + im2col 卷积层 | 两者输出 allclose；卷积反向与数值梯度一致 |
| 3 | `Pooling` | §7.3（p214–215） | 2×2 max pooling 前向 + argmax 回传 | 形状正确；反向与数值梯度一致 |
| 4 | `SimpleConvNet` | §7.5（p224–227） | Conv-ReLU-Pool-Affine-Softmax | 合成三分类测试集 acc ≥ 0.90 |

## 怎么跑

```bash
cd ch07
python exercises.py
python answers.py
```

> 原书用 MNIST、输出 98.96%；本练习用 `common/toy_data.py` 的 **12×12 合成图像**（方框/圆/十字）离线可跑，网络结构与训练流程同构。

## 提示

- `im2col` 的列布局是 `(C, filter_h, filter_w)`（C 在最外层）；`col2im` 的还原要按同一布局 `reshape/transpose` 回去。
- 反向：`dW = (col.T @ dout_col).T` 再 reshape 回 `(out_c,C,fh,fw)`；`dx` 用 `col2im(dout_col @ W_col.T)`。
- 池化反向不是"求导"，而是**把梯度路由到最大值的位置**（forward 时存 argmax）。
- CNN 训练：这批合成图用 lr=0.1、150 个 epoch 可到 100%；若 loss 不降，先检查 im2col/col2im 的顺序与 `dW` 形状。

## 跑通原文（可选）

见 [run_book.md](run_book.md)：`ch07/train_convnet.py` → MNIST 测试精度 **98.96%**（§7.5 起于印刷 p224；**98.96% 在印刷 p228／PDF p253**）。
