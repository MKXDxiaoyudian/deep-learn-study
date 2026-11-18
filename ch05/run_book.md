# 第 5 章「跑通原文代码」任务

> 目标：跑通官方仓库 `ch05/gradient_check.py`，复现 **§5.7.3（印刷 p158／PDF p183）** 的梯度确认结果：**误差约 1e-12**（书内 b1:9.7e-13、W2:8.4e-13）。
> ⚠️ 本页是操作说明；官方代码版权归原作者，请遵守其 LICENSE。

## 步骤

```bash
# 若还没 clone
git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git
cd deep-learning-from-scratch/ch05
python gradient_check.py
```

## 预期输出

```
--------------- numerical:  ...
--------------- analytical: ...
b1: 9.7e-13
W2: 8.4e-13
```

（数值可能因 NumPy 版本略有不同；量级应为 **1e-12 ~ 1e-13**，不应出现 1e-6 以上。）

## 思考

- 为什么这里能到 1e-12，而 ch04 的数值梯度训练慢？—— 反向传播是"解析求导"，数值微分是"近似求导"。
- 如果误差突然变大到 1e-6，最可能错在哪？（提示：某一层的 backward 公式或 dout 形状）

## 记录（自己填）

- 运行日期：
- 两个误差值：
- 用时：
- 卡点：
