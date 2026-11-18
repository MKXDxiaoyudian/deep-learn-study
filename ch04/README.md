# 第 4 章 神经网络的学习 —— 代码练习

> **对应原书**：第 4 章（印刷 p81–119）｜**知识点**：§4.2 损失函数、§4.3 数值微分、§4.4 梯度、§4.5 学习算法实现
> **建议用时**：4–5 h（W3 寒假）｜**原书警示**：`train_neuralnet.py` 用数值梯度跑 10000 次非常慢，第一次请改小 `iters_num`（原书 §4.5.2 也建议先用 gradient() 版）。

## 任务

| # | 函数 | 对应原文 | 要求 | 验收 |
|---|---|---|---|---|
| 1 | `mean_squared_error` / `cross_entropy_error` | §4.2（p85–93） | 支持单样本与批量；log 里加 1e-7 | MSE=0.0975；CE=0.5108；批量 CE=0.2899 |
| 2 | `numerical_diff` / `numerical_gradient` | §4.3–§4.4（p94–109） | 中心差分 h=1e-4；对向量逐元素求 | x² 在 3 处导数为 6；x²+y² 在 (3,4) 梯度 (6,8) |
| 3 | `softmax_batch` | §3.5.1（p64–66）的批量推广 | 按行减最大值再归一化 | 每行和为 1 |
| 4 | `train_two_layer_numeric` | §4.4–§4.5（p102–118） | 用数值梯度训练 2→4→2 网络 | loss 至少降到初始的 50% 以下，准确率 ≥0.90 |

## 怎么跑

```bash
cd ch04
python exercises.py
python answers.py
```

> 答案用 `two_blobs` 两分类数据（`common/toy_data.py`）；原书用 MNIST + 螺旋数据，思路完全一致。

## 提示

- 交叉熵批量公式：`-sum(t*log(y+1e-7))/batch`；`t` 是 one-hot。
- 数值梯度很慢：参数展平后对每一维做两次前向；本练习参数只有 22 维，所以能秒级跑完——这也是为什么原书第 5 章要发明反向传播。
- 训练循环：`grad = numerical_gradient(loss_fn, flat); flat -= lr * grad`；每 50 步打印一次 loss。
- 如果 loss 不降：把 lr 调小（0.5 → 0.1）、或检查 one-hot 与 softmax 维度。

## 跑通原文（可选）

见 [run_book.md](run_book.md)：`ch04/train_neuralnet.py`（记得改小 `iters_num`）。
