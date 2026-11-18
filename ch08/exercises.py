"""第 8 章 深度学习 —— 参考答案（含 TinyML 接口）。运行：python answers.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def layer_budget(layers, input_shape):
    C, H, W = input_shape
    out = []
    for layer in layers:
        t = layer["type"]
        if t == "conv":
            oc, k, s, p = layer["out_c"], layer["k"], layer["stride"], layer["pad"]
            oh = (H + 2 * p - k) // s + 1
            ow = (W + 2 * p - k) // s + 1
            params = (C * k * k + 1) * oc            # 权重 + 偏置
            macs = oc * oh * ow * (C * k * k)        # 不含 bias
            out.append({"name": f"conv{oc}x{k}", "out_shape": (oc, oh, ow),
                        "params": params, "macs": macs})
            C, H, W = oc, oh, ow
        elif t == "pool":
            k, s = layer["k"], layer["stride"]
            oh = (H - k) // s + 1
            ow = (W - k) // s + 1
            out.append({"name": f"pool{k}", "out_shape": (C, oh, ow), "params": 0, "macs": 0})
            H, W = oh, ow
        elif t == "affine":
            units = layer["out"]
            params = C * H * W * units + units
            macs = C * H * W * units
            out.append({"name": f"affine{units}", "out_shape": (units,),
                        "params": params, "macs": macs})
            C, H, W = units, 1, 1
        else:
            raise ValueError(f"unknown layer: {t}")
    return out


def _sym_quant(a, bits=8):
    a = np.asarray(a, dtype=np.float32)
    qmax = 2 ** (bits - 1) - 1
    scale = float(np.max(np.abs(a))) / qmax
    if scale == 0:
        return a.copy()
    q = np.clip(np.round(a / scale), -qmax - 1, qmax)
    return (q * scale).astype(np.float32)


def _relu(x):
    return np.maximum(0.0, x)


def precision_compare(x, W1, b1, W2, b2):
    x = np.asarray(x, dtype=np.float32)
    W1 = np.asarray(W1, dtype=np.float32)
    W2 = np.asarray(W2, dtype=np.float32)

    def mlp(xx, w1, w2, bb1, bb2):
        return _relu(xx @ w1 + bb1) @ w2 + bb2

    fp32 = mlp(x, W1, W2, b1, b2)
    # 模拟半精度存储：cast 到 fp16 再转回 fp32
    fp16 = mlp(x.astype(np.float16).astype(np.float32),
               W1.astype(np.float16).astype(np.float32),
               W2.astype(np.float16).astype(np.float32),
               b1.astype(np.float16).astype(np.float32),
               b2.astype(np.float16).astype(np.float32))
    # 模拟 int8：仅量化 x / W1 / W2（bias 保持 fp32）
    int8 = mlp(_sym_quant(x), _sym_quant(W1), _sym_quant(W2),
               np.asarray(b1, dtype=np.float32), np.asarray(b2, dtype=np.float32))
    return {"fp32": fp32, "fp16": fp16, "int8": int8}


def main():
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
