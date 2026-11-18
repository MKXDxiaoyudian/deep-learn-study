#!/usr/bin/env python3
"""一键自检：
① 逐章跑参考答案；
② 确认初始骨架按预期失败（exit 1 + NotImplementedError）；
③ 把答案的公开实现注入骨架后，跑**骨架自己的 main()**（防"骨架补全后仍失败"）。

用法：
    python verify_all.py
"""
import contextlib
import importlib.util
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CHAPTERS = ["ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(ch, filename):
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, ch, filename)],
        cwd=os.path.join(ROOT, ch), capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def check_answers(ch):
    rc, out = _run(ch, "answers.py")
    return rc == 0, out


def check_skeleton(ch):
    """返回 ("fail_expected", out) / ("done", out) / ("bad", out)。

    - 还没做：exit != 0 且含 NotImplementedError -> 符合预期；
    - 已经补全：exit == 0 -> 跳过“应失败”检查（这是正常的，说明题做完了）；
    - 其它：真的坏了。
    """
    rc, out = _run(ch, "exercises.py")
    if rc == 0:
        return "done", out
    if "NotImplementedError" in out:
        return "fail_expected", out
    return "bad", out


def check_injection(ch):
    """注入答案的公开实现（跳过 main），再执行骨架自己的 main。"""
    ex = _load(os.path.join(ROOT, ch, "exercises.py"), f"ex_{ch}")
    ans = _load(os.path.join(ROOT, ch, "answers.py"), f"ans_{ch}")
    import types
    injected = 0
    for name in dir(ans):
        if name.startswith("_") or name == "main":
            continue
        obj = getattr(ans, name)
        if isinstance(obj, types.ModuleType) or name in ("os", "sys", "np", "HERE"):
            continue
        if hasattr(ex, name):
            setattr(ex, name, obj)
            injected += 1
    assert injected >= 2, f"{ch}: 只注入了 {injected} 个名字，检查接口是否对齐"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ex.main()  # 骨架自己的 main，不是 answers.main
    return True, (buf.getvalue().strip(), injected)


def main():
    failed = []
    for ch in CHAPTERS:
        print(f"\n=== {ch} ===")
        ok, out = check_answers(ch)
        if out:
            print(out)
        if not ok:
            failed.append(f"{ch} (answers)")
            print(f"❌ {ch} answers FAILED")
            continue
        state, out2 = check_skeleton(ch)
        if state == "done":
            print(f"ℹ️ {ch} 骨架已补全（exit 0，跳过“应失败”检查）")
        elif state == "fail_expected":
            pass
        else:
            failed.append(f"{ch} (skeleton)")
            print(f"❌ {ch} 骨架报错但不是 NotImplementedError（见上面输出）")
            continue
        try:
            ok3, (out3, injected) = check_injection(ch)
        except Exception as exc:  # noqa: BLE001
            ok3, out3, injected = False, f"{type(exc).__name__}: {exc}", 0
        if out3:
            print(out3)
        if not ok3:
            failed.append(f"{ch} (injection)")
            print(f"❌ {ch} injection FAILED")
        else:
            print(f"✅ {ch} PASSED（answers + 骨架失败 + 注入 {injected} 个实现）")
    print("\n" + "=" * 40)
    if failed:
        print(f"❌ 失败：{', '.join(failed)}")
        return 1
    print(f"✅ 全部 {len(CHAPTERS)} 章通过（answers + 骨架失败 + 注入自检）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
