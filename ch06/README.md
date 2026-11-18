# 第 6 章 与学习相关的技巧 —— 代码练习

> **对应原书**：第 6 章（印刷 p163–200）｜**知识点**：§6.1 参数更新、§6.2 权重初始化、§6.3 BatchNorm、§6.4 Dropout
> **建议用时**：5–6 h（寒假 W5–W6 之间）

## 任务

| # | 内容 | 对应原文 | 要求 | 验收 |
|---|---|---|---|---|
| 1 | `SGD` / `Momentum` / `AdaGrad` / `Adam` | §6.1（p163–175） | 四个 optimizer 的 `update(params, grads)` | 在 f=0.5(w0²+10w1²) 上均收敛到 loss<1e-2 |
| 2 | `init_scale_compare` | §6.2（p176–183） | 比较 Xavier / He / 0.01 的逐层激活标准差 | He 末层 std>0.05；小初始化逐层衰减 |
| 3 | `batchnorm_forward` / `backward` | §6.3（p184–187） | 训练模式前向 + 反向 | 每特征均值≈β、方差≈γ²；反向与数值梯度一致 |
| 4 | `dropout_forward` | §6.4（p188–194） | inverted dropout（训练缩放、推理原样） | 训练时置零比例≈p、期望≈原值；推理时恒等 |

## 怎么跑

```bash
cd ch06
python exercises.py
python answers.py
```

## 提示

- Momentum 要保存 `v`（形状同参数）；AdaGrad 保存累计平方梯度 `h`；Adam 要算偏差修正后的步长。
- 初始化对比：每层 `x = tanh(x @ W)`，W 的 std 分别取 `1/sqrt(n)`（Xavier）、`sqrt(2/n)`（He）、`0.01`；观察第 5 层激活 std。
- BN 反向公式较繁：先算 `dbeta/dgamma`，再算 `dx`（对 `xhat`、`var`、`mu` 三段反传）。
- 本练习的 Dropout 用 `seed` 固定随机数，便于断言；真实训练不要固定 seed。
