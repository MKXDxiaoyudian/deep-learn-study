# 第 8 章 深度学习 —— 代码练习（含 TinyML 接口）

> **对应原书**：第 8 章（印刷 p235–265）｜**知识点**：§8.1 加深网络、§8.2 深度学习小历史（VGG/GoogLeNet/ResNet）、§8.3 高速化（含 **8.3.4 位精度缩减**）、§8.4 应用案例
> **建议用时**：3–4 h（寒假 W7）｜**和端侧主线直接相关**：参数/内存/算力预算 + 8-bit 量化直觉。

## 任务

| # | 内容 | 对应原文 | 要求 | 验收 |
|---|---|---|---|---|
| 1 | `layer_budget` | §8.1（p235–241） | 给一串 conv/pool/affine，算输出尺寸/参数/乘加数 | conv 8×3×3 输出 (8,12,12)、params=80、macs=10368；affine 输出 3、params=867 |
| 2 | `precision_compare` | §8.3.4（p252 原文直接谈嵌入式） | 模拟 fp16 与 int8 量化后的 MLP 输出误差 | fp16 相对误差 <1e-2；int8 <20% |
| 3 | 跑通原文 | §8.3.4（p252） | [run_book.md](run_book.md)：`ch08/half_float_network.py` | 半精度推理精度与单精度基本持平 |
| 4 | 阅读整理 | §8.2.2–§8.2.4（p244–246） | 对比 VGG/GoogLeNet/ResNet；算 3×3 堆叠参数量 | 见 [reading.md](reading.md) 的答案要点 |

## 怎么跑

```bash
cd ch08
python exercises.py
python answers.py
```

## 提示

- `layer_budget` 的 MACs 只算乘加（不含 bias）：conv 层 `out_c*oh*ow*(C*k*k)`；affine 层 `in*out`；pool 记 0。
- 模拟 fp16：`x.astype(np.float16).astype(np.float32)`（模拟"存成半精度再算"）。
- 模拟 int8：对称量化 `scale=max|a|/127; q=round(a/scale); a≈q*scale`；分别量化 x 与权重。
- 原书 §8.3.4 的结论：**推理时位数缩减对精度影响小**，这正是 TinyML 量化部署的起点。

