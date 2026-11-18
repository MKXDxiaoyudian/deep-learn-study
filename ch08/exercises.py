"""第 8 章 深度学习 —— 练习骨架（含 TinyML 接口）

对应原书
- §8.1 加深网络（印刷 p235–241）：参数/计算量增长
- §8.3 深度学习的高速化（印刷 p248–252／PDF p273–277）：位精度缩减（8.3.4，p252 原文直接谈到嵌入式）
- §8.4 深度学习的应用案例（p253–257）

运行 `python exercises.py` 全过即完成。提示见 ch08/README.md。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def layer_budget(layers, input_shape):
    """计算一串层的输出尺寸 / 参数量 / 乘加次数（单样本）。

    参数
    - input_shape: (C, H, W)
    - layers: list[dict]，支持三种：
        {"type":"conv","out_c":8,"k":3,"stride":1,"pad":1}
        {"type":"pool","k":2,"stride":2}
        {"type":"affine","out":3}
    返回
    - list[dict]：每个 dict 含 name / out_shape / params / macs
      （pool 的 macs 记 0，参数量也记 0；conv 的 macs 不含 bias）
    """
    raise NotImplementedError("TODO: 实现 layer_budget")


def _sym_quant(a, bits=8):
    """对称量化（单张量）：返回反量化后的 float32（保留形状）。"""
    raise NotImplementedError("TODO: 实现对称量化")


def precision_compare(x, W1, b1, W2, b2):
    """对比 fp32 / 模拟 fp16 / 模拟 int8 的 MLP 输出（§8.3.4 的位精度缩减）。

    返回 dict：{"fp32": logits32, "fp16": logits16, "int8": logits8}
    网络：relu(x@W1+b1)@W2+b2
    - fp16：把 x/W/b 都 cast 成 np.float16 再转回 float32 计算（模拟半精度存储）
    - int8：对 x、W1、W2 做对称量化（b 保持 fp32）
    """
    raise NotImplementedError("TODO: 实现 precision_compare")


def main():
    # §8.1 参数/计算量预算（对应原书图 8-1 的"层越深参数越多"）
    layers = [
        {"type": "conv", "out_c": 8, "k": 3, "stride": 1, "pad": 1},
        {"type": "pool", "k": 2, "stride": 2},
        {"type": "affine", "out": 3},
    ]
    budget = layer_budget(layers, (1, 12, 12))
    assert [b["out_shape"] for b in budget] == [(8, 12, 12), (8, 6, 6), (3,)], budget
    assert budget[0]["params"] == 80, budget[0]
    assert budget[0]["macs"] == 10368, budget[0]
    assert budget[2]["params"] == 288 * 3 + 3, budget[2]
    assert budget[2]["macs"] == 288 * 3, budget[2]

    # §8.3.4 位精度缩减：fp32 vs 模拟 fp16 / int8
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (64, 16))
    W1 = rng.normal(0, 0.3, (16, 32))
    b1 = rng.normal(0, 0.1, 32)
    W2 = rng.normal(0, 0.3, (32, 8))
    b2 = rng.normal(0, 0.1, 8)
    out = precision_compare(x, W1, b1, W2, b2)
    ref = out["fp32"]
    err16 = float(np.max(np.abs(out["fp16"] - ref)) / (np.max(np.abs(ref)) + 1e-12))
    err8 = float(np.max(np.abs(out["int8"] - ref)) / (np.max(np.abs(ref)) + 1e-12))
    assert err16 < 1e-2, err16
    assert err8 < 0.2, err8
    assert float(np.max(np.abs(out["fp16"] - ref))) > 0.0
    assert float(np.max(np.abs(out["int8"] - ref))) > 0.0
    print(f"[ch08] 预算 OK；fp16 相对误差={err16:.2e}, int8 相对误差={err8:.2%}")


if __name__ == "__main__":
    main()
