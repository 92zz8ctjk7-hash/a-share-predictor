# A 股涨跌幅预测系统

基于 Python 的 A 股预测与量化交易研究框架。对接 **baostock / akshare** 数据源，
覆盖「数据拉取 → 特征工程 → 级联模型训练 → 策略回测 → 生产决策推送」全链路：

- **双层级联模型**：日线基座（LSTM，沪深 300 联合训练）→ 分钟模型（IntradayLSTM，滚动 30 分钟样本，面板三股联合）
- **生产策略**：深跌加倍 + logistic 门控 + 延迟卖出（walk-forward 验证 4 年 **+83%**）
- **严格评估**：多 regime walk-forward、逐段重训、资金跨段连续；一切策略改动必须过回测
- **自动化服务**：launchd 定时增量训练、双账户（影子模拟/真实跟踪）盘中决策推送

## 特性

- **数据接口**：baostock 日线/分钟线/宏观 + akshare 盘中实时分钟源（新浪）；本地 Parquet 分片缓存，断点续传、增量更新
- **特征工程**：14 个技术因子 + 宏观利率（125 维）+ 市场环境（指数环境/日历 12 维）；训练前 z-score 标准化（训练段统计量，防泄漏）
- **防泄漏体系**：标签用未来数据计算、数据集按 code 优先排序 + 全局日期分位数切分、seq 滑窗跨股掩码、回测逐段重训严格样本外
- **双任务标签**：`label_reg`（未来 5 日涨跌幅）+ `label_cls`（涨跌方向）；分钟级标签为「下一个 30 分钟涨跌」
- **模型**：基线 sklearn 集成 / MLP / LSTM（回归+分类双头）/ IntradayLSTM（序列+静态双输入）/ 轨迹解码器
- **回测**：网格策略（T+1 撮合、整手交易、佣金万2.5/印花税/滑点成本模型）+ 多种门控与仓位策略对比
- **评估**：MSE/MAE/R² + AUC/F1 + 因子 IC/ICIR/分层收益 + 绩效（年化/回撤/夏普/胜率）

## 目录结构

```
a-share-predictor/
├── config.py                 # 全局配置（数据/模型/回测/调度/推送/辅助训练股）
├── main.py                   # CLI 入口（约 20 个子命令）
├── data/
│   ├── fetcher.py            # baostock 日线/分钟线拉取（断点续传/增量）
│   ├── fetcher_external.py   # 基本面事件 + 宏观利率拉取
│   ├── fetcher_realtime.py   # akshare 盘中实时分钟线（新浪源）
│   ├── store.py              # 本地存储层：分片 Parquet 读写
│   └── schema.py             # 数据结构定义
├── features/
│   ├── builder.py            # 日线特征与标签（流式构建 samples 分片）
│   ├── min_builder.py        # 分钟样本：10:00 截面 + 滚动 30 分钟窗口
│   ├── dataset.py            # PyTorch Dataset、时间切分、标准化 scaler 持久化
│   ├── min_dataset.py        # 分钟样本 Dataset
│   ├── external.py           # 宏观/基本面特征化与对齐
│   └── market.py             # 市场环境特征（沪深300指数环境 + 日历效应）
├── models/
│   ├── nn.py                 # MLP / LSTM / IntradayLSTM
│   ├── baselines.py          # sklearn 基线
│   ├── train.py              # 训练/推理/checkpoint
│   └── incremental.py        # 每日增量更新流水线（数据→样本→基座→分钟→gate）
├── backtest/
│   ├── engine.py             # 回测引擎（T+1 撮合/成本/绩效/交易明细/动态仓位）
│   ├── strategy.py           # 网格 + 耐心低吸 + 深跌加倍 + 延迟卖出策略族
│   ├── run.py                # 多窗口编排、级联回测、pretrained 模式
│   ├── factor_eval.py        # 因子 IC / ICIR / 分层收益
│   ├── fusion.py             # 信号融合门控实验（分位数阈值/回撤预算）
│   ├── capital_plan.py       # 资金计划实验（分层触发/回撤预算）
│   └── plot.py               # 资金曲线绘图
├── rl_gate/
│   ├── gate.py               # logistic 门控（生产采用）
│   ├── opportunity.py        # 买入机会样本构造
│   ├── backtest.py           # walk-forward 多模式对比框架（8 模式）
│   ├── adaptive_gate.py      # 自适应分位数门控（探索-利用）
│   ├── dqn.py / env.py       # DQN 门控研究（差分奖赏）
│   ├── intensity_rl.py       # 强度控制与 regime 切换研究
│   └── trajectory.py         # 轨迹解码器（日内路径拟合）
├── serving/
│   ├── predict.py            # 盘中预测核心（分钟模型 30 分钟预判）
│   ├── intraday_signal.py    # 盘中参考消息（方向/轨迹/波动/策略建议）
│   ├── strategy_advice.py    # 生产决策引擎（深跌加倍+门控+延迟卖出+执行流）
│   ├── shadow_account.py     # 双账户记账（影子模拟/真实跟踪，FIFO/T+1/成本）
│   ├── push.py               # pushplus / 企业微信推送
│   ├── signal_store.py       # 信号持久化
│   └── scheduler.py          # launchd 定时任务安装
└── utils/metrics.py          # 指标工具
```

## 安装

```bash
cd a-share-predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

依赖：baostock、akshare、pandas、numpy、scikit-learn、torch、pyarrow、colorlog、matplotlib。

## 使用流程

### 1. 数据拉取

```bash
# 日线（沪深 300 成分股，近 5 年，断点续传）
python main.py fetch --all

# 分钟线（5 分钟，baostock 约 5 年历史）
python main.py fetch-min --codes sz.000100 --frequency 5

# 宏观利率 / 基本面事件
python main.py fetch-external --macro
```

本地存储为 Parquet 分片（`cache/raw/`、`cache/raw_min/5/`、`cache/samples/` 等），
增量更新（`--refresh`）按日期去重合并。

### 2. 特征与样本

```bash
python main.py features                    # 日线样本（技术+宏观+市场环境，自动注入）
python main.py min-features --rolling      # 滚动分钟样本（当前30分钟→预测下一个30分钟）
python main.py factor-ic --codes sz.000100 # 因子有效性分析（IC/ICIR/分层收益）
```

分钟滚动样本：每天 7 条（每 30 分钟窗口一条），序列 24 维（6 根 bar × 涨跌/相对开盘/量比/振幅）
+ 静态 33 维（T-1 技术特征 + 压力位 + base_pred 级联 + 市场环境）。

### 3. 模型训练与每日增量

```bash
python main.py train --model lstm --epochs 10          # 基座 LSTM
python main.py base-predict                            # 基座全量推理 → base_preds.parquet
python main.py min-train --frequency 5 --epochs 30     # 分钟模型
```

**每日增量更新（生产流水线核心）**：

```bash
python main.py incremental-update --codes sz.000100
```

一次执行：数据刷新（沪深 300 日线 + 面板三股分钟线）→ 日线样本重建 →
基座滚动窗口重训（近 2 年，标准化 scaler 持久化）→ 基座推理（含辅助股）→
滚动分钟样本重建 → 分钟模型重训（近 12 个月）→ logistic 门控重训。

**模型体系**：

| 模型 | 训练数据 | 输出 | 用途 |
|---|---|---|---|
| 基座 LSTM | 沪深 300 日线，151 维×10 日序列 | 未来 5 日涨跌幅 | 门控/延迟卖出/级联特征 |
| 分钟 IntradayLSTM | 面板三股滚动样本（sz.000100 + 京东方A + 彩虹股份，后两者仅训练不 serving） | 未来 30 分钟涨跌 | 盘中概率预判 |
| logistic gate | 沪深 300 买入机会样本（22 维状态特征） | 放行/拦截买入 | 生产买入门控 |

### 4. 回测系统

回测引擎按 T+1 撮合（T 日收盘决策 → T+1 开盘成交 / 分钟信号当日收盘成交），
成本含佣金（双边万 2.5，最低 5 元）、印花税（卖出 0.05%）、滑点（0.1%）。

```bash
# 多窗口网格回测（窗口前 80% 训练、后 20% 样本外）
python main.py backtest --codes sz.000100 --windows 5y,3y,2y,1y --model lstm

# 级联分钟信号回测 / walk-forward 多 regime 分段
python main.py backtest --codes sz.000100 --signal-source cascade --walk --segments 4

# 加载已有 checkpoint 直接回测（不重训）
python main.py backtest --codes sz.000100 --model lstm --pretrained

# 门控策略 walk-forward 多模式对比（none/fixed/rolling/gate/adaptive/patient/pgate/agg）
python main.py rl-backtest --codes sz.000100 --walk --segments 4

# 绘制资金曲线
python main.py plot-equity --code sz.000100 --window 3y
```

产物：`cache/meta/backtest_report.csv`、`rl_walk_report.csv`、`equity_*.csv/png`、
`trades_{code}_{window}.csv`（逐笔交易明细）等。

## 策略体系（生产配置）

当前生产策略为 **深跌加倍 + logistic 门控 + 延迟卖出**，
walk-forward 4 年（4 段牛熊震荡、逐段重训、资金跨段连续）验证 **+83.0%**
（对比：纯网格 +62.9%、logistic 门控 +68.6%）：

| 组件 | 规则 | 验证结论 |
|---|---|---|
| 深跌加倍（买入侧） | 模型看跌时浅跌区（距下界 2 格内）攒弹药不买；深跌区每格加倍 2 份筹码 | 弹药集中在深跌底部，反弹利润放大 |
| logistic 门控（买入侧） | 22 维市场状态特征学「该接的跌 vs 该躲的跌」 | 基座修复后跑赢纯网格 +5.7pp |
| 延迟卖出（卖出侧） | 强看涨（未来5日>+2%）且有持仓时少卖 1 格，让利润奔跑 | +6.1pp（卖出侧唯一有效改造） |
| 网格骨架 | 10 格 ±20%，上界清仓/下界冻结，卖出机械止盈 | 反弹自动止盈不可「优化」得更激进 |

**已验证无效的方向**（代码保留为实验参数）：固定 0.5 阈值门控（零交易）、
自适应分位数门控、动态资金缩放、分层/预算回撤控制、加速止盈——
共同教训：网格的买入发生在下跌日、卖出发生在反弹日，任何与这个节奏冲突的
仓位/卖出干预都会侵蚀收益；回撤控制与网格低吸结构性冲突。

## 自动化服务与推送（serving）

每个交易日的自动流程（launchd）：

- **08:30** `incremental-update`：完整增量流水线（数据→样本→基座→分钟→gate）
- **10:01** 两条推送：
  - `intraday-predict`：分钟模型未来 30 分钟预判（预测价/涨跌幅/上涨把握/买卖倾向）
  - `intraday-signal`：盘中参考（基座方向/轨迹拟合/波动档位/策略建议/账户状态/波段计划）

**双账户体系**：

| 账户 | 用途 | 状态 |
|---|---|---|
| 影子账户 | 10 万模拟起步，按策略自动记账，验证 +83% 能否复现 | 推送中展示 |
| 真实账户 | 按实际资金筹码跟踪，记录分日收益与波段盈亏 | 仅本地（`cache/signals/account_real.json`），推送只含策略建议不含资金数额 |

撮合语义与回测一致（T+1 开盘成交/FIFO/整手/佣金印花税滑点）；
首次建仓锁定网格锚定价，清仓止盈后重锚。

**推送消息结构**（盘中参考）：现价与开盘对比 → 大盘模型方向 → 剩余时间轨迹 →
波动档位 → 策略建议+决策依据（网格位置/未来5日预测/门控状态）→ 影子账户 →
真账建议（含每格股数）→ 波段计划（反弹至 X 卖 N 股 / 回落至 Y 买 N 股）。

```bash
# 手动执行服务任务（--dry-run / --no-push 不推送）
python main.py serve --codes sz.000100
python main.py serve-intraday --codes sz.000100

# 安装定时任务（--load 立即启用）
python main.py install-scheduler --load
python main.py install-intraday-scheduler --load
```

推送凭证在 `cache/.env`（不入版本库）：

```bash
PUSHPLUS_TOKEN=xxxx        # pushplus token（需实名认证）
PUSHPLUS_TOPIC=xxxx        # 群组频道代码（可选，一对多群发）
WECOM_WEBHOOK=https://...  # 企业微信机器人（可选备用渠道）
```

可靠性设计：实时源新鲜度守卫（开盘初期数据源未同步到今天则跳过，不推过期信号）、
全链路降级（基座/gate/账户任一失败降级为纯网格建议，推送不中断）、
分钟模型 serving 对漏训天然鲁棒（当日样本不可训练，漏训无累积损害）。

## 重要提示

- **数据与回测仅用于研究，不构成投资建议**
- 标签 `label_reg` 用未来第 `horizon` 个交易日收盘价计算，末尾缺失行自动剔除
- 数据集切分在时间维度顺序进行（code 优先排序 + 全局日期分位数切分），不可随机打乱
- 特征训练前 z-score 标准化（仅用训练段统计量，持久化 scaler 供推理对齐）；
  未标准化会导致大量级慢变特征主导梯度、模型退化为常数预测器
- 一切策略/门控/仓位改动必须过多 regime walk-forward 验证，单窗口结论不可靠
- baostock 多进程并发会互踢会话，拉取任务必须串行
