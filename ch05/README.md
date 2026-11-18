# 第 5 章 误差反向传播法 —— 代码练习

> **对应原书**：第 5 章（印刷 p121–161）｜**知识点**：§5.3–§5.4 计算图/层、§5.5 激活函数层、§5.6 Affine/Softmax 层、§5.7 两层网络
> **建议用时**：6–8 h（寒假 W5，本章是全书的分水岭）

## 任务

| # | 内容 | 对应原文 | 要求 | 验收 |
|---|---|---|---|---|
| 1 | `MulLayer` / `AddLayer` | §5.3.3–§5.4（p133–138） | 实现乘/加层并复现苹果与"苹果+橘子"例子 | 220、715；梯度 [2.2,110,200] 与 [2.2,110,3.3,165] |
| 2 | `Relu` / `Sigmoid` | §5.5（p139–143） | forward 缓存、backward 用 y 或 mask | 与数值微分一致（≤1e-5） |
| 3 | `Affine` | §5.6（p144–153） | 2D 版：`x@W+b`；反向 dX/dW/db | 形状正确；dW、db 与数值梯度一致 |
| 4 | `SoftmaxWithLoss` | §5.6（p150–153） | forward 标量 loss；backward 返回 `(y-t)/batch` | 与手算 `(y-t)` 一致 |
| 5 | `TwoLayerNet` | §5.7（p154–160） | 用上面 4 种层组装；`gradient()` 反向传播 | 与数值梯度 `allclose`；two_blobs 训练 acc ≥0.95 |

## 怎么跑

```bash
cd ch05
python exercises.py
python answers.py
```

## 提示

- 乘法层反向：`forward(x,y)=x*y` → `backward(dout) = (dout*y, dout*x)`（顺序别反）。
- ReLU 反向只保留 `x>0` 的位置；Sigmoid 反向是 `dout * (1-y) * y`。
- Affine 的 `db = sum(dout, axis=0)`（对 batch 求和）；`dW = x.T @ dout`。
- SoftmaxWithLoss 的 backward 是 **`(y-t)/batch`**——这个式子第 5.6.3 节推导过，务必能手推。
- 梯度核对用 `np.allclose(ana, num, atol=1e-4)`；接近 0 的维度用"相对误差"会误报。
- 本章不实现 4 维卷积版 Affine（那是 ch07 的事）；本章只做 2D。

## 跑通原文（可选）

见 [run_book.md](run_book.md)：`ch05/gradient_check.py` 的数值微分 vs 反向传播误差约 **1e-12**。
