# 鱼书①《深度学习入门》代码练习（第 2–8 章）

> 配合《深度学习入门：基于Python的理论与实现》（斋藤康毅）的**代码练习**：每章一组任务，附参考答案与自检。
> 题目**贴着原书章节**出（每条都标了 §节号 + 印刷页码），任务类型包括：实现书中算法、跑通书中某段代码、以及少量阅读整理题。
> **来源说明**：题目与说明为原创；部分层的示例实现**参考/改编**自官方配套仓库 `oreilly-japan/deep-learning-from-scratch`（MIT License，见 `NOTICE`）。原书文字、插图与官方代码版权归原作者/出版社。

## 这仓库里有什么

- **7 个章节目录**（`ch02`–`ch08`），共 **31 行任务**：25 个代码题 + 5 个"跑通原文代码"题（可选）+ 1 个阅读整理题；只算必做代码题：不含 ch06 为 21、含 ch06（选做）为 25；ch02 另有 2 道思考题不计入；
- **代码题**结构：`exercises.py`（**骨架 + 自检**，你补全）→ `answers.py`（**参考答案**）→ `README.md`（任务说明/提示/验收）；**跑通题**见各章 `run_book.md`，**阅读题**见 `ch08/reading.md`；
- `common/toy_data.py`：离线数据生成器（两高斯团、12×12 合成图像；`spiral()` 备用），**不依赖网络/MNIST**；
- `verify_all.py`：一键自检 = ① 跑全部参考答案；② 确认初始骨架按预期失败；③ 把答案注入骨架后跑骨架 `main()`（防止"骨架补全后仍失败"）。

## 快速开始

```bash
# 1) 环境（Python 3.10+）
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) 从第 2 章开始：先读 ch02/README.md，再补全 ch02/exercises.py
cd ch02
python exercises.py        # 你的版本；全过 = 完成
python answers.py          # 参考答案（先自己做！）

# 3) 全部答案自检（在仓库根目录）
python verify_all.py
```

## 目录结构

```
.
├── README.md
├── requirements.txt · .gitignore · LICENSE · NOTICE
├── verify_all.py
├── common/  toy_data.py（离线数据）· __init__.py
├── ch02/  README.md  00-先读-w和b怎么来的.md  exercises.py  answers.py
├── ch03/  README.md  exercises.py  answers.py  run_book.md
├── ch04/  README.md  exercises.py  answers.py  run_book.md
├── ch05/  README.md  exercises.py  answers.py  run_book.md
├── ch06/  README.md  exercises.py  answers.py
├── ch07/  README.md  exercises.py  answers.py  run_book.md
└── ch08/  README.md  exercises.py  answers.py  run_book.md  reading.md
```

## 与书 / 寒假周次的对应

| 章 | 原书页 | 主题 | 练习 | 建议周次（2027 寒假） |
|---|---|---|---|---|
| ch02 | p21–36 | 感知机 | 3 代码题 | W1 |
| ch03 | p37–79 | 神经网络 | 3 代码题 + 1 跑通题（MNIST 93.52%） | W2 |
| ch04 | p81–119 | 神经网络的学习 | 4 代码题 + 1 跑通题 | W3 |
| ch05 | p121–161 | 误差反向传播 | 5 代码题 + 1 跑通题（1e-12） | W5 |
| ch06 | p163–200 | 学习技巧（优化器/初始化/BN/Dropout） | 4 代码题 | W6（选做） |
| ch07 | p201–233 | CNN | 4 代码题 + 1 跑通题（98.96%） | W6 |
| ch08 | p235–265 | 深度学习 + 位精度缩减 | 2 代码题 + 1 跑通题 + 1 阅读题 | W7 |

> 说明：ch06 属选读（理解放春季，不压寒假）；这里的 4 道题可以跟着 ch05 之后挑做。
> **ch02 特别提示**：做之前先读 `ch02/00-先读-w和b怎么来的.md`——它把「w、b 怎么选」用 AND 完整演算了一遍（含 NAND/OR、XOR 两层、学习算法）。

## 每题怎么用（建议 4 步）

1. **读原文**：按 `README.md` 标的 §节号翻书（页码也给了）；
2. **写骨架**：打开 `exercises.py`，把 `TODO` 补全（自检块已经写好，可以先看它要什么）；
3. **跑通**：`python exercises.py`，全部 assert 通过即完成；
4. **对照答案**：`answers.py` 看实现差异，把"为什么这样写"写进你自己的笔记（不是抄一遍）。

> "跑通原文代码"题：需要自行 `git clone` 官方仓库并准备数据（如 MNIST），按 `run_book.md` 的命令执行、记录输出。**把官方仓库 clone 到练习目录之外**（或放进已被 `.gitignore` 忽略的 `deep-learning-from-scratch/`），不要把官方仓库/数据集提交进本仓库。

## 上传到 GitHub（用 git 导出“骨架版”，不要直接复制工作目录）

> ⚠️ 工作目录里的 `ch02/exercises.py` 可能已经被你填完，`ch02/` 下还可能有 `我的尝试-*.py` 草稿。
> 直接 `cp -R` 会把**答案/草稿/.venv/.vscode** 都带上。正确做法是从 study 仓库导出**已提交的骨架版**：

```bash
# 1) 从 study 仓库导出已提交的版本（ch02/exercises.py 是未填写的骨架）
DEST="$HOME/Documents/deep-learning-from-scratch-exercises"
rm -rf "$DEST" && mkdir -p "$DEST"
git -C "$HOME/study" archive HEAD   "notes/02-科研/深度学习/01-鱼书1-深度学习入门/代码练习"   | tar -x -C "$DEST" --strip-components=5

# 2) 去掉本地尝试快照（它们也提交过的话）
rm -f "$DEST"/ch02/我的尝试-*.py

# 3) 在独立目录初始化并提交
cd "$DEST"
git init
git add .
git commit -m "鱼书①代码练习：ch02–ch08（含答案）"

# 4) 到 GitHub 建空仓库后：
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
git branch -M main && git push -u origin main
# 或者用 gh：gh repo create <仓库名> --public --source=. --push
```

> 说明：导出的是 **HEAD（已提交）** 的版本，所以不会把你在本地填好的 `ch02/exercises.py` 带上去。
> 上传前建议先跑一遍 `python verify_all.py`，并确认 `git status --short` 为空、没有 `data/`、`*.pkl`、`我的尝试-*`、`.venv/`。
## 验收与自检

```bash
python verify_all.py    # 答案 + 骨架失败 + 注入自检；有失败会打印 ❌ 并返回非 0
```

## 常见问题

- **`ModuleNotFoundError: No module named 'numpy'`**：你选错解释器了——选的是系统/Homebrew 的 Python，不是练习目录里的 venv。按下面二选一：
  - **编辑器里选解释器**：选 `代码练习/.venv/bin/python`（VS Code：`Cmd+Shift+P` → `Python: Select Interpreter` → `Enter interpreter path...` → 粘贴这个路径；PyCharm：Settings → Project → Python Interpreter → Add → Existing → 选它）；
  - **终端里激活**：在 `代码练习/` 目录执行 `source .venv/bin/activate`，再 `cd ch02 && python exercises.py`。
  - 验证：`python -c "import sys, numpy; print(sys.executable, numpy.__version__)"` 应显示 `.../代码练习/.venv/bin/python 2.5.5`（或你的 numpy 版本）。
- **需要 MNIST 吗？** 不需要：代码题用 `common/toy_data.py` 的离线数据（或固定随机数）；只有"跑通原文"题需要你自己 clone 官方仓库并准备数据。
- **能只用标准库吗？** 不能，需要 NumPy（`pip install -r requirements.txt`）。
- **书中代码在哪？** 官方仓库 `github.com/oreilly-japan/deep-learning-from-scratch`；使用时请遵守其 LICENSE。
- **和寒假节奏的关系**：建议 W1 ch02、W2 ch03、W3 ch04、W5 ch05、W6 ch06（选做）+ ch07、W7 ch08；每章完成记一行即可。
