# 第 7 章「跑通原文代码」任务

> 目标：跑通官方仓库 `ch07/train_convnet.py`，复现书中 **§7.5（印刷 p224–227；98.96% 在印刷 p228／PDF p253）** 的 MNIST 测试精度 **98.96%**。
> ⚠️ 本页是操作说明；官方代码版权归原作者，请遵守其 LICENSE。

## 步骤

```bash
git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git   # 若已 clone 可跳过
cd deep-learning-from-scratch/ch07
# 可选：先改小 iters_num（10000 -> 1000）快速看 loss 是否下降；完整跑需要几分钟
python train_convnet.py
```

## 预期输出

- 每个 epoch 打印 `train acc, test acc`（可能以 `=== epoch ...` 形式）；
- 最终 test acc 约 **0.98–0.99**（书中为 **98.96%**）；
- 运行时间：CPU 数分钟（原书 p237 提到深层网络训练要半天以上；本脚本是小型 CNN，几到十几分钟）。

## 思考

- 书中说 `im2col` 比朴素实现**更费内存**（p218）——在本练习的 `SimpleConvNet` 里，哪一步最占内存？
- 若要把这个 CNN 放到 STM32F103（64 KB Flash / 20 KB RAM），你会先砍哪一层？

## 记录（自己填）

- 运行日期：
- 最终 test acc：
- 用时：
- 卡点：
