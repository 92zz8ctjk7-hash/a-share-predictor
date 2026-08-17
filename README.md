# A 股涨跌幅预测系统

基于 Python 的 A 股预测与量化交易研究框架。对接 **baostock / akshare** 数据源，
覆盖「数据拉取 → 特征工程 → 级联模型训练 → 网格回测 → 盘中信号推送」全链路：

- **双层级联模型**：日线基座（LSTM，全市场训练）+ 盘中分钟模型（IntradayLSTM）
- **回测系统**：网格交易策略 + 模型信号门控，多窗口 / walk-forward 严格样本外评估
- **RL 门控研究**：logistic 门控、Bandit 自适应、DQN、混合 regime 切换
- **自动化服务**：launchd 定时增量训练、盘中信号 pushplus/企业微信推送

## 特性

- **数据接口**：baostock 日线/分钟线/基本面/宏观 + akshare 盘中实时分钟源；本地 Parquet 分片缓存，断点续传、增量更新
- **数据结构**：`Bar` / `Stock` / `FeatureSample` dataclass，兼容 pandas / numpy / torch
- **特征工程**：15 个技术特征 + 宏观利率 + 基本面事件 + 图结构 + 市场环境（指数环境/日历）特征，自动检测注入
- **双任务标签**：`label_reg`（未来 N 日涨跌幅）+ `label_cls`（涨跌方向）
- **防泄漏**：标签用未来数据计算，数据集严格按时间顺序切分；回测逐窗口训练、walk-forward 分段严格样本外
- **模型**：
  - 基线：`GradientBoostingRegressor` + `GradientBoostingClassifier`
  - 神经网络：`MLP` 与 `LSTM`（回归+分类双头多任务损失）；盘中 `IntradayLSTM`、轨迹解码器
- **回测**：网格策略（T+1 撮合、佣金/印花税/滑点成本模型）+ 模型门控，绩效含年化/回撤/夏普/胜率
- **评估**：MSE / RMSE / MAE / R² + Accuracy / F1 / AUC + 因子 IC/ICIR/分层收益

## 目录结构

```
a-share-predictor/
├── config.py                 # 全局配置（数据/模型/回测/调度/推送）
├── main.py                   # CLI 入口（约 20 个子命令）
├── data/
│   ├── fetcher.py            # baostock 日线拉取（全市场/断点续传/增量）
│   ├── fetcher_external.py   # baostock 基本面事件 + 宏观利率拉取
│   ├── fetcher_realtime.py   # akshare 盘中实时分钟线
│   ├── store.py              # 本地存储层：分片 Parquet 读写
│   ├── external_store.py     # 基本面/宏观序列存储
│   ├── graph_store.py        # 图结构存储与一阶邻居聚合
│   └── schema.py             # 数据结构定义
├── features/
│   ├── builder.py            # 日线特征与标签（流式构建 samples 分片）
│   ├── min_builder.py        # 分钟级样本构造（10:00 截面）
│   ├── dataset.py / min_dataset.py  # PyTorch Dataset 与时间切分
│   ├── external.py           # 宏观/基本面/图特征化与对齐
│   └── market.py             # 市场环境特征（指数环境 + 日历）
├── models/
│   ├── baselines.py          # sklearn 基线
│   ├── nn.py                 # MLP / LSTM / IntradayLSTM
│   ├── train.py              # 训练与评估
│   └── incremental.py        # 滚动窗口增量训练
├── backtest/
│   ├── engine.py             # 回测引擎（T+1 撮合 / 成本 / 绩效）
│   ├── strategy.py           # 网格策略 + 模型门控
│   ├── factor_eval.py        # 因子 IC / ICIR / 分层收益
│   ├── run.py                # 多窗口编排、级联回测、walk-forward
│   └── plot.py               # 资金曲线绘图
├── rl_gate/
│   ├── gate.py               # logistic / Bandit 门控
│   ├── dqn.py                # DQN 门控（差分奖赏）
│   ├── intensity_rl.py       # 强度控制
│   ├── trajectory.py         # 轨迹解码器（剩余路径拟合）
│   ├── opportunity.py        # 机会识别
│   ├── env.py / backtest.py  # RL 环境与 walk-forward 对比回测
├── serving/
│   ├── predict.py            # 收盘后预测服务
│   ├── intraday_signal.py    # 盘中实盘信号（轨迹+强度+网格建议）
│   ├── push.py               # pushplus / 企业微信推送
│   ├── signal_store.py       # 信号持久化（cache/signals/）
│   └── scheduler.py          # launchd 定时任务安装
└── utils/
    └── metrics.py            # 指标与简易回测工具
```

## 安装

```bash
cd a-share-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖：baostock、pandas、numpy、scikit-learn、torch、pyarrow、colorlog、akshare、matplotlib。

## 使用流程

### 1. 拉取历史数据

**全量日线（推荐，全市场分片存储）**：

```bash
# 首次全量拉取（近 5 年日线，约 5000+ 只，含退市股）
python main.py fetch --all --start 2021-08-01

# 中断后重跑同一命令即可断点续传（已有分片自动跳过）

# 之后每日增量更新（拉取缓存最后日期之后的数据）
python main.py fetch --all --refresh
```

**小批量模式**：

```bash
python main.py fetch --codes sh.600000,sh.600519,sz.000001 --adjust 2
```

**分钟线（仅部分股票，按频率分片存储）**：

```bash
# 拉取部分股票近 3 个月的 5 分钟线
python main.py fetch-min --codes sz.000100 --frequency 5 --start 2026-05-14

# 每日增量更新到最新交易日
python main.py fetch-min --codes sz.000100 --frequency 5 --refresh
```

- `--frequency`：5 / 15 / 30 / 60 分钟（默认 5）
- baostock 分钟线**仅保留最近约 1 年历史**，且**指数无分钟线**；
  分钟线字段与日线不同（含 time，无 preclose/turn/pctChg），默认不复权
- `--codes`：逗号分隔，带市场前缀（`sh.` 沪市 / `sz.` 深市）
- `--adjust`：1=后复权 2=前复权（默认）3=不复权
- `--no-delisted`：全量拉取时排除退市股（默认包含，避免幸存者偏差）

**外部数据（基本面事件 + 宏观利率）**：

```bash
# 拉取默认股票池的基本面事件 + 宏观利率序列
python main.py fetch-external --macro

# 拉取指定股票（分红/财报/业绩预告/业绩快报）
python main.py fetch-external --codes sh.600000,sz.000001

# 全市场基本面事件（断点续传，重跑同一命令即可继续）
python main.py fetch-external --all --macro
```

### 1.1 本地存储格式

全量数据以 **Parquet 分片** 落盘在 `cache/` 下：

```
cache/
├── .env                      # 推送凭证（WECOM_WEBHOOK / PUSHPLUS_TOKEN）
├── meta/
│   ├── stock_list.parquet    # 全市场股票清单
│   ├── base_preds.parquet    # 基座全量推理结果
│   ├── backtest_report.csv   # 回测报告 / factor_ic.csv / walk 报告等
│   └── equity_*.csv / *.png  # 资金曲线与图表
├── raw/{code}.parquet        # 原始日线，每只股票一个文件
├── raw_min/{freq}/{code}.parquet    # 原始分钟线，按「频率/股票」两层分片
├── samples/{code}.parquet    # 日线特征样本，每只股票一个文件
├── min_samples/{freq}/{code}.parquet  # 分钟级样本
├── models/                   # checkpoint（best.pt / epoch_N.pt / intraday_lstm_5.pt 等）
├── external/                 # 基本面 / 宏观 / 图结构（见下文）
└── signals/                  # 盘中信号持久化
```

| 层 | 内容 | 字段 | dtype |
|---|---|---|---|
| raw | 原始日线（11 列） | date code open high low close preclose volume amount turn pct_chg | date=datetime64, code=str, 其余 float64 |
| raw_min | 原始分钟线（9 列） | date time code open high low close volume amount | date/time=datetime64, code=str, 其余 float64 |
| samples | 特征+双标签 | date code close label_reg label_cls + 15 个特征（外部特征自动追加） | 特征与标签 float32, label_cls int8 |

设计要点（详见 `data/store.py`）：

- **raw 按股票分片**：断点续传（文件存在即跳过）、单只增量更新（按 date 去重合并）、流式特征构建（内存恒定）
- **samples 按股票分片**：训练时 `pd.read_parquet("cache/samples")` 按目录读取，等价于合并长表
- 特征列存 float32，存储与内存均减半

> 注意：前复权数据在发生新的除权除息后历史价格会整体平移，
> 增量更新（--refresh）只追加新交易日；如需严格一致可定期全量重拉。

### 2. 构建特征与标签

```bash
python main.py features --window 20 --horizon 5
```

有 raw 分片时自动流式逐只构建 samples 分片（中断重跑自动跳过已构建股票；
`--overwrite` 强制重建）；宏观/基本面/图特征若存在会自动注入为新列。

### 3. 训练模型

```bash
python main.py train --model lstm --epochs 30   # 神经网络
python main.py train --model baseline           # sklearn 基线
python main.py base-predict                     # 基座全量推理 → cache/meta/base_preds.parquet
```

### 4. 分钟级收盘价预测（级联架构）

用开盘后 30 分钟的 5 分钟线 + 昨日技术指标（含压力位）+ 基座模型预测，
预测当日收盘价。两级模型：

1. **基座模型**：全市场日线训练（horizon=1），预测次日涨跌
2. **分钟模型**（IntradayLSTM）：部分股票的分钟线训练，基座预测作为输入特征级联

```bash
python main.py min-features                      # 构造分钟样本 → cache/min_samples/5/
python main.py min-train --epochs 30             # 训练 → cache/models/intraday_lstm_5.pt
python main.py min-predict --code sz.000100      # 盘中预测（需 10:00 后运行）
```

分钟样本每条包含：

- 序列特征 24 维：10:00 前 6 根 5 分钟线 × 4 维（bar 间涨跌 / 相对开盘价 / 量比 / 振幅）
- 静态特征 21 维：昨日 14 个技术特征 + 6 个压力位特征 + 基座预测
- 标签：`label_rest`（10:00 至收盘涨跌 %）、`label_cls_rest`（涨跌方向）

无泄漏约定：输入只用 ≤10:00 的信息，标签只用 15:00 收盘价；
样本切分严格按全局日期顺序（训练→验证→测试）。

## 回测系统

回测引擎按 T+1 规则撮合（T 日开盘执行前一日 `next_open` 信号 /
T 日收盘执行当日 `close` 信号），成本模型含佣金（双边万 2.5，最低 5 元）、
印花税（卖出 0.05%）、滑点（0.1%）。主策略为**网格交易**：
上下界 = 回测首日收盘价 × (1±range_pct)，价格越低持仓格数越多，
突破上界清仓 / 下界冻结买入；模型门控在预测下跌（prob_up < 阈值）时只卖不买。

```bash
# 多窗口网格回测（每窗口前 80% 训练、后 20% 样本外）
python main.py backtest --codes sz.000100 --windows 5y,3y,2y,1y --model lstm

# 纯网格（关闭模型门控）
python main.py backtest --codes sz.000100 --no-gate

# 级联信号（分钟模型收盘撮合 + 基座回退）
python main.py backtest --codes sz.000100 --signal-source cascade --min-frequency 5

# 级联 walk-forward：多 regime 分段严格样本外，逐段重训基座，资金跨段连续
python main.py backtest --codes sz.000100 --signal-source cascade --walk --segments 4

# 直接加载已有 checkpoint 回测（不重新训练）
python main.py backtest --codes sz.000100 --model lstm --pretrained

# 绘制回测资金曲线
python main.py plot-equity --code sz.000100 --window 3y
```

**因子有效性分析**（≥5 只用逐日截面 Spearman IC，单只用滚动 60 日时序 IC；
输出 IC 均值 / ICIR / 正占比 / 5 分位分层收益）：

```bash
python main.py factor-ic --codes sh.600000,sz.000001 --horizon 5
```

产物：`cache/meta/backtest_report.csv`、`equity_{code}_{window}.csv/png`、
`factor_ic.csv`、`cascade_walk_report.csv` 等。

## RL 门控体系（rl_gate）

在网格主策略之上研究多种门控信号切换方式，并用 walk-forward 严格样本外对比：

| 门控模式 | 说明 |
|---|---|
| none | 纯网格（基线，全周期最稳健） |
| logistic | 逻辑回归门控（模型概率阈值） |
| rolling | Bandit / 滚动自适应门控（探索-利用平衡） |
| gate（RL） | DQN 差分奖赏门控 |
| hybrid | regime 软切换：震荡用 DQN、趋势用 logistic |

另有轨迹解码器（拟合当日剩余价格路径）、已实现波动率强度信号与机会识别。

```bash
# Bandit 门控四模式对比回测（严格样本外）
python main.py rl-backtest --codes sz.000100 --windows 5y,3y,2y,1y

# 全周期滚动 walk-forward（切多段覆盖牛熊震荡，资金跨段连续）
python main.py rl-backtest --codes sz.000100 --walk --segments 4

# 对比实验：禁用市场环境因子 / 启用 DQN / 启用混合门控
python main.py rl-backtest --codes sz.000100 --walk --no-market
python main.py rl-backtest --codes sz.000100 --walk --with-dqn --with-hybrid
```

产物：`cache/meta/rl_walk_report.csv`、`rl_gate_report.csv`、`equity_rl_*.csv/png`。

> 当前结论：主策略为纯网格（全周期回测最稳健），DQN / 混合门控保留作为后续优化方向。

## 自动化服务与推送（serving）

每个交易日的自动流程：

- **08:30** `incremental-update`：刷新行情数据、滚动窗口重训基座与分钟模型
- **10:01** 两条独立推送任务：
  - `intraday-predict`：分钟模型输出未来 30 分钟预测价、涨跌幅、上涨概率与偏多/偏空信号
  - `intraday-signal`：基座方向 + 轨迹解码器拟合路径 + 波动强度 + 网格买卖建议
- 非交易日自动跳过；推送渠道 pushplus（微信/群组）或企业微信机器人，
  信号同时持久化到 `cache/signals/`

```bash
# 一键增量更新（刷新数据 + 滚动窗口重训）
python main.py incremental-update --codes sz.000100 --base-window 2y --min-window 12m

# 手动执行一次服务任务（--dry-run 只打印不推送）
python main.py serve --codes sz.000100
python main.py serve-intraday --codes sz.000100 --no-push

# 安装 launchd 定时任务（--load 立即启用）
python main.py install-scheduler --load
python main.py install-intraday-scheduler --load
```

推送凭证配置在 `cache/.env`（不入版本库）：

```bash
PUSHPLUS_TOKEN=xxxx        # pushplus 个人 token
PUSHPLUS_TOPIC=xxxx        # pushplus 群组频道代码（可选，一对多群发）
WECOM_WEBHOOK=https://...  # 企业微信机器人 webhook（可选）
```

盘中实时数据通过 akshare 分钟源拉取，并临时构造当日日线行供日线特征使用。

## 外部数据与图结构

以图结构组织基本面 / 宏观关联关系并作为模型输入，数据落盘在 `cache/external/`：

```
cache/external/
├── fundamental/{code}.parquet   # 个股基本面事件长表（每行一个事件）
├── macro/{name}.parquet         # 宏观时间序列（基准利率/准备金率/M0-M2 等）
└── graph/
    ├── nodes.parquet            # 图节点：stock / industry / macro
    └── edges.parquet            # 图边：belongs_to / correlates / affects
```

**基本面事件 schema**：`date`（公告日）, `code`, `event_type`
（annual_report / semi_report / quarterly_report / performance_forecast / dividend）,
`revenue`, `revenue_yoy`, `net_profit`, `net_profit_yoy`, `eps`, `roe`,
`dividend_per_share`, `ex_date`, `pay_date`

### 外部数据如何注入模型（自动检测）

| 数据 | 影响基座模型（日线） | 影响分钟模型（盘中） |
|---|---|---|
| 宏观序列 | 按样本当日 join（收盘后视角） | 按 T-1 日历日 merge_asof（10:00 盘前视角） |
| 图结构（行业/关联/宏观影响） | 一阶邻居聚合为图特征静态注入 | 暂不注入（可扩展） |
| 基本面事件（财报/分红） | 暂不注入（可扩展） | 每日状态特征静态注入（T-1 视角） |

外部特征以新列进入 samples / min_samples，模型输入维度自动扩展；
无外部数据时行为与之前完全一致。写完数据后重新运行
`python main.py features` / `min-features` 即自动注入。

## 自定义模型

在 `models/nn.py` 中新增网络并在 `MODEL_REGISTRY` 中注册；或继承
`BaselineEnsemble` 的风格实现 `fit() / predict()` 接口即可，
数据管道与评估逻辑无需改动。

## 重要提示

- **数据仅用于研究，不构成投资建议**
- 目标标签 `label_reg` 使用未来第 `horizon` 个交易日收盘价计算，
  样本表末尾标签缺失的行已自动剔除
- 数据集切分在 **时间维度** 顺序进行，不可随机打乱，否则引入未来信息泄漏
- baostock 获取的数据为复权后价格（默认前复权），适合技术指标计算
- 门控 / 策略评估须用多 regime 滚动 walk-forward，单窗口结论不可靠
