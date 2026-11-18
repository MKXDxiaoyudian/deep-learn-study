# 第 4 章「跑通原文代码」任务

> 目标：跑通官方仓库 `ch04/train_neuralnet.py`，观察 **§4.5.2（p114–116）** 的两层网络 mini-batch 训练。
> ⚠️ 本页是操作说明；官方代码版权归原作者，请遵守其 LICENSE。

## 步骤（重要：先改小迭代数）

```bash
git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git   # 若已 clone 可跳过
cd deep-learning-from-scratch/ch04
# 打开 train_neuralnet.py，把 iters_num = 10000 改成 200～500（原书 p114 脚注也建议先用 gradient()）
python train_neuralnet.py
```

## 预期输出

- 每 100 次左右打印 `train acc, test acc | ...`，loss 持续下降；
- 迭代 200–500 次后 test acc 大致在 **0.85–0.93** 区间（随迭代数变化；原书完整 10000 次的结果受随机初始化影响）；
- 运行时间：数值梯度版本 200 次约 10–30 s（取决于机器）；10000 次会非常久。

## 思考

- 为什么这一章用**数值梯度**？它和 ch05 的反向传播在「结果正确性」与「速度」上差多少？
- 如果把 `iters_num` 改成 10000，预计要跑多久？（用 200 次的时间 ×50 估算）

## 记录（自己填）

- 运行日期：
- iters_num：
- 最终 test acc：
- 用时：
- 卡点：
