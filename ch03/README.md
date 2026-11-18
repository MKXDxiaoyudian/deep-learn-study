# 第 3 章 神经网络 —— 代码练习

> **对应原书**：第 3 章（印刷 p37–79）｜**知识点**：§3.2 激活函数、§3.3 多维数组运算、§3.4 三层前向、§3.5 softmax
> **建议用时**：3–4 h（W2 寒假）

## 任务

| # | 函数 | 对应原文 | 要求 | 验收 |
|---|---|---|---|---|
| 1 | `step_function` / `sigmoid` / `relu` | §3.2（p42–49） | 用 NumPy 实现三种激活函数 | step([-1,0,1])=[0,0,1]；sigmoid(0)=0.5；relu 负数归零 |
| 2 | `forward_3layer` | §3.3–§3.4（p50–62） | 矩阵乘法 + 前两层 sigmoid、输出层恒等 | 单样本 (2,) / 批量 (N,2)；与手算式一致 |
| 3 | `softmax` | §3.5.1–§3.5.2（p64–66） | **减最大值**防溢出，归一化 | 和为 1；`softmax([1000,1001,1002])` 有限且等于 `softmax([0,1,2])` |

## 怎么跑

```bash
cd ch03
python exercises.py
python answers.py
```

## 提示

- `sigmoid` 直接 `1/(1+np.exp(-x))` 即可；`step_function` 返回 **int 数组**（不是 bool）。
- `forward_3layer` 的维度链：`x(d0) → a1(d1) → a2(d2) → y(d3)`，每步 `a = f(a@W + b)`；输出层不加激活。
- softmax 的单样本版就够；批量版（每行归一化）是 §3.5.1 的延伸，可留到 ch04 的 mini-batch 再做。
- 为什么要减最大值？见 §3.5.2：`exp(1000)` 会变 `inf`，减去最大值不改变结果但避免溢出。

## 跑通原文（可选，需 clone 官方仓库 + MNIST 数据）

见 [run_book.md](run_book.md)：运行 `ch03/neuralnet_mnist.py`，复现书中 **93.52%** 的推理精度。
