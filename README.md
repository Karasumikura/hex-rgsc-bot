# Hex RGSC Bot

一个 11×11 海克斯棋（Hex）AI 引擎：神经网络引导的 PUCT-MCTS，叠加针对 Hex 的战术检测和比赛用时管理。权重通过复现 [RGSC（Regret-Guided Search Control，ICLR 2026）](https://arxiv.org/abs/2602.20809) 自对弈训练得到。

本引擎用于**中国大学生计算机博弈大赛 · 海克斯棋项目**，代表吉林大学参赛，获**国家级三等奖**。

> 仓库只包含引擎代码，不包含训练好的权重文件。

## 特性

- **神经网络 + PUCT-MCTS**：AlphaZero 式残差网络给出落子先验和局面价值，蒙特卡洛树搜索在时间预算内展开。
- **Hex 战术层**：必胜点直接落子、唯一必堵点直接封堵；多个必堵点时把根节点限定在这些点上；对“落子后对手出现双重必胜”的着法大幅降权。
- **危险局面识别**：用 Dijkstra 计算双方最短连接距离，对手距连通 ≤ 4 步时进入危险模式，提升能拉长对手距离的着法先验。
- **比赛用时管理**：按 30 分钟包干计时，根据剩余着数动态分配每步时间，分档封顶并保留安全余量。
- **两种运行协议**：比赛现场的官方记谱命令行（默认），以及 [Botzone](https://www.botzone.org.cn/) 的 JSON 长驻协议。
- **失败即停**：比赛模式下权重未载入直接拒绝启动，不会悄悄退化成规则引擎。

## 快速开始

需要 Python 3.8+ 和 PyTorch（CPU 版即可）。

```bash
pip install -r requirements.txt
```

把权重文件 `weight_iter_78000.pkl` 放在以下任一位置（相对当前目录或脚本所在目录）：

```text
data/weight_iter_78000.pkl
data/models/weight_iter_78000.pkl
models/weight_iter_78000.pkl
weight_iter_78000.pkl
```

权重文件可以是 `state_dict`，也可以是带 `network` 键的 checkpoint 字典。

## 运行模式

### 比赛命令行（默认）

```bash
python hex_bot.py      # 启动后输入 start R 或 start B
python hex_bot.py --side R    # 直接以先手（红方）开局
python hex_bot.py --side B --time-log logs/game1.csv   # 后手，并记录每步思考耗时
```

输入对手着法，引擎输出自己的着法。坐标为列 `A`–`K`、行 `1`–`11`，支持 `E7`、`E,7`、`(E,7)`、`R(E,7)` 四种写法。

```text
READY HEX-CLI model=state_dict weight=weight_iter_78000.pkl side=B
SIDE B
WAIT R
F6   ← 输入对手着法
B(E,8)      ← 引擎着法
```

| 命令 | 作用 |
| --- | --- |
| `start R` / `start B` | 开新局，R 为先手，B 为后手 |
| `E7` 等 | 输入对手着法 |
| `help` | 显示帮助 |
| `quit` | 退出 |

标准输出只打印着法，方便直接抄写；搜索次数、价值估计、用时等调试信息写到标准错误。`--time-log` 会把每步的思考时间、累计用时、剩余时间和预算写成 CSV，与官方棋谱分开保存。

### Botzone

```bash
python hex_bot.py --botzone
```

使用 Botzone 的 JSON 输入输出和长驻模式。本地单次调试可设置 `BOTZONE_KEEP_RUNNING=0`，每回合处理一次后退出。此模式下若权重缺失，会退回启发式走法，保证能正常提交。

## 引擎设计

### 网络

| 部分 | 结构 |
| --- | --- |
| 输入 | 4 个 11×11 平面：己方棋子、对方棋子、两个轮次平面 |
| 主干 | 3×3 卷积 256 通道 + 3 个残差块 |
| policy 头 | 121 维落子分布 |
| value 头 | tanh 输出 [-1, 1] 的局面价值 |
| regret / ranking 头 | RGSC 训练阶段用于挑选高 regret 局面作为自对弈起点；对局推理时不使用 |

共约 366 万参数。推理固定单线程 CPU，并缓存最近 4096 个局面的网络输出。

### 搜索

每步决策按以下顺序：

1. 有一步连通的必胜点，直接下（多个时按 policy 和靠近中心挑选）；
2. 对手只有一个必胜点，直接堵；
3. 其余情况运行 PUCT-MCTS（`c_puct = 1.25`），根节点最多展开 32 个候选、子节点 16 个；
4. 对手有多个必胜点时，根节点只在这些点之间搜索；
5. 最终选择访问次数最多的子节点。

搜索内部在根节点和候选较少的节点做同样的必胜 / 必堵扫描，并直接给出叶子价值，减少无效模拟。

### 用时管理

比赛模式按每方 30 分钟计时：

- 预留 180 秒安全时间，其余时间按剩余着数平均分配；
- 开局一步最多 25 秒，常规最多 45 秒，危险局面最多 90 秒，且危险局面分配量乘 1.75；
- 剩余不足 180 秒时进入快棋模式，每步最多 5 秒；
- 每 8 次模拟检查一次时钟，并保留硬截止余量。

Botzone 模式使用平台的单步限时（首步 7.7 秒，之后 3.8 秒）。

### 坐标约定

引擎内部先手连接左右两边、后手连接上下两边。比赛记谱里 R（先手）连接上下、B（后手）连接左右，命令行在输入输出时做一次转置：记谱的列对应内部的行，记谱的行对应内部的列。Botzone 坐标另有一套映射，见 `bot_to_internal` / `internal_to_bot`。

## 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `BOTZONE_KEEP_RUNNING` | `1` | Botzone 长驻模式；设为 `0` 每回合退出 |
| `RGSC_ALLOW_TORCHSCRIPT` | 未设置 | 设为 `1` 时，找不到 `.pkl` 会尝试加载 TorchScript 格式的 `weight_iter_78000.pt` |

## 参考

- Tsai et al. *Regret-Guided Search Control for Efficient Learning in AlphaZero*. ICLR 2026. [arXiv:2602.20809](https://arxiv.org/abs/2602.20809)
- Silver et al. *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Science, 2018.

## License

[MIT](LICENSE)
