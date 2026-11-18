# 第 3 章「跑通原文代码」任务

> 目标：跑通官方配套仓库的 `ch03/neuralnet_mnist.py`，复现书中 **§3.6.2（p73–75）** 的推理精度 **93.52%**。
> ⚠️ 本页是**操作说明**，不是答案代码；官方仓库代码版权归原作者，请遵守其 LICENSE。

## 步骤

```bash
# 1) 在练习目录外单独 clone 官方仓库（不要提交进本练习仓库）
git clone https://github.com/oreilly-japan/deep-learning-from-scratch.git
cd deep-learning-from-scratch

# 2) 确认数据集脚本存在（仓库自带 dataset/mnist.py，首次运行会自动下载 MNIST）
ls dataset/mnist.py

# 3) 运行第 3 章的推理脚本（在 ch03 目录内运行，脚本自己 sys.path.append(os.pardir)）
cd ch03
python neuralnet_mnist.py
```

## 预期输出

- 最后一行形如：`Accuracy:0.9352`（不同 NumPy 版本可能有微小差异；书内为 93.52%）。
- 中间会打印每张图片的预测类别（0–9）。

## 你可能踩的坑

1. **数据集下载失败**：`dataset/mnist.py` 首次运行会联网下载；国内网络可先手动下载 4 个 idx 文件放到 `dataset/`（见脚本注释）。
2. **从错误目录运行**：必须在 `ch03/` 内运行（脚本用 `sys.path.append(os.pardir)` 找 `dataset/` 与 `common/`）。
3. **NumPy 版本**：书中脚本在 NumPy 2.x 可能报弃用警告，不影响结果；若报错按提示改。
4. **别把官方仓库提交进本练习仓库**：本练习仓库只保留"跑通记录"（命令 + 你的输出截图/文本）。

## 记录（自己填）

- 运行日期：
- 实际输出最后一行：
- 用时：
- 卡点：
