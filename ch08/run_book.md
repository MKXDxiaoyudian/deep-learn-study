# 第 8 章「跑通原文代码」任务

> 目标：跑通官方仓库 `ch08/half_float_network.py`，复现 **§8.3.4（印刷 p252／PDF p277）** 的"半精度推理精度几乎不变"。
> ⚠️ 本页是操作说明；官方代码版权归原作者，请遵守其 LICENSE。

## 步骤

```bash
git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git   # 若已 clone 可跳过
cd deep-learning-from-scratch/ch08
# 需要预训练参数文件 deep_convnet_params.pkl（仓库/书中提供；若缺失按脚本提示下载）
python half_float_network.py
```

## 预期输出

- 会分别打印 **float32 推理精度** 与 **半精度（fp16）推理精度**；
- 两者应非常接近（书中结论：半精度推理的识别精度与单精度基本相同；具体数字以实际运行为准，书中深层 CNN 约 99.38%）。

## 思考

- 为什么"训练"很少用 fp16、但"推理"可以？（提示：梯度累积 vs 前向存储）
- 如果只能存 int8，你会先量化权重还是先量化激活？为什么（见 §8.3.4）？

## 记录（自己填）

- 运行日期：
- float32 精度 / half 精度：
- 用时：
- 卡点：
