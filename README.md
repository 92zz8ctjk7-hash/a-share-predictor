# A 股涨跌幅预测系统

基于 Python 的 A 股历史数据 + 机器学习/深度学习涨跌幅预测框架。对接 **baostock** 数据源，统一数据结构，支持 sklearn 基线与 PyTorch 神经网络（MLP / LSTM），同时预测「未来 N 日涨跌方向」与「涨跌幅度」。

## 特性

- **数据接口**：封装 baostock 登录 / 查询，本地 Parquet 缓存，避免重复请求
- **数据结构**：`Bar` / `Stock` / `FeatureSample` dataclass，兼容 pandas / numpy / torch
- **特征工程**：MA、RSI、MACD、波动率、量比、价格位置等 15 个技术特征
- **双任务标签**：`label_reg`（未来 N 日涨跌幅）+ `label_cls`（涨跌方向）
- **防泄漏**：标签用未来数据计算，数据集严格按时间顺序切分（训练→验证→测试）
- **模型**：
  - 基线：`GradientBoostingRegressor` + `GradientBoostingClassifier`（或线性模型）
  - 神经网络：`MLP` 与 `LSTM`（PyTorch，回归+分类双头多任务损失）
- **评估**：MSE / RMSE / MAE / R² + Accuracy / F1 / AUC + 简单信号回测

## 目录结构

```
a-share-predictor/
├── config.py                 # 全局配置
├── main.py                   # CLI 入口
├── data/
│   ├── fetcher.py            # baostock 接口封装（全市场拉取/断点续传/增量更新）
│   ├── store.py              # 本地存储层：分片 Parquet 格式定义与读写
│   └── schema.py             # 数据结构定义
├── features/
│   ├── builder.py            # 特征工程与标签（支持流式构建 samples 分片）
│   └── dataset.py            # PyTorch Dataset 与时间切分
├── models/
│   ├── baselines.py          # sklearn 基线
│   ├── nn.py                 # MLP / LSTM
│   └── train.py              # 训练与评估
└── utils/
    └── metrics.py            # 指标与回测
```

## 安装

```bash
cd a-share-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 注意：`torch` 体积较大。如不需要神经网络，可先只跑基线模型，
> 将 `requirements.txt` 中的 `torch>=2.0.0` 注释掉。

## 使用流程

### 1. 拉取历史数据

**全量模式（推荐，全市场分片存储）**：

```bash
# 首次全量拉取（近 5 年日线，约 5000+ 只，含退市股）
python main.py fetch --all --start 2021-08-01 --end 2026-08-14

# 中断后重跑同一命令即可断点续传（已有分片自动跳过）

# 之后每日增量更新（拉取缓存最后日期之后的数据）
python main.py fetch --all --refresh
```

**小批量模式**：

```bash
python main.py fetch \
  --codes sh.600000,sh.600519,sz.000001 \
  --start 2021-08-01 --end 2026-08-14 \
  --adjust 2
```

**分钟线模式（仅部分股票，按频率分片存储）**：

```bash
# 拉取部分股票近 3 个月的 5 分钟线
python main.py fetch-min --codes sh.600000,sz.000001 --frequency 5 \
  --start 2026-05-14 --end 2026-08-14

# 每日增量更新到最新交易日
python main.py fetch-min --codes sh.600000,sz.000001 --frequency 5 --refresh
```

- `--frequency`：5 / 15 / 30 / 60 分钟（默认 5）
- baostock 分钟线**仅保留最近约 1 年历史**，且**指数无分钟线**；
  分钟线字段与日线不同（含 time，无 preclose/turn/pctChg），默认不复权

- `--codes`：逗号分隔，带市场前缀（`sh.` 沪市 / `sz.` 深市）
- `--adjust`：1=后复权 2=前复权（默认）3=不复权
- `--no-delisted`：全量拉取时排除退市股（默认包含，避免幸存者偏差）

**外部数据（基本面事件 + 宏观利率）**：

```bash
# 拉取默认股票池（cfg.default_codes）的基本面事件 + 宏观利率序列
python main.py fetch-external --macro

# 拉取指定股票（分红/财报/业绩预告/业绩快报），分片存储到 cache/external/fundamental/
python main.py fetch-external --codes sh.600000,sz.000001

# 全市场基本面事件（断点续传，重跑同一命令即可继续）
python main.py fetch-external --all --macro
```

- 基本面事件按公告日对齐：分红（每股股利/除权日/派息日）、季频财报（营收/净利/EPS/ROE/净利同比）、
  业绩预告（净利增减幅度）、业绩快报；特征化见 `features/external.py`
- 宏观序列（存款/贷款基准利率、存款准备金率、货币供应量 M0/M1/M2）统一
  reindex 到交易日历并 ffill，训练时按 T-1 日历日 backward 合并，无未来泄漏
- baostock 无汇率 / Shibor / 新闻接口，需外部数据源（如 akshare）

### 1.1 本地存储格式

全量数据以 **Parquet 分片** 落盘在 `cache/` 下：

```
cache/
├── meta/
│   ├── stock_list.parquet    # 全市场股票清单（code/名称/上市/退市日期/状态）
│   └── failed_codes.txt      # 拉取失败的股票（重跑 fetch --all 即可续传）
├── raw/                      # 原始日线，每只股票一个文件
│   ├── sh.600000.parquet
│   └── sz.000001.parquet
├── raw_min/                  # 原始分钟线，按「频率/股票」两层分片
│   ├── 5/                    # 5 分钟线（另有 15/30/60 分钟目录）
│   │   └── sh.600000.parquet
│   └── 15/
└── samples/                  # 特征样本，每只股票一个文件
    ├── sh.600000.parquet
    └── sz.000001.parquet
```

| 层 | 内容 | 字段 | dtype |
|---|---|---|---|
| raw | 原始日线（11 列） | date code open high low close preclose volume amount turn pct_chg | date=datetime64, code=str, 其余 float64 |
| raw_min | 原始分钟线（9 列） | date time code open high low close volume amount | date/time=datetime64, code=str, 其余 float64 |
| samples | 特征+双标签（22 列） | date code close label_reg label_cls + 15 个特征 | 特征与标签 float32, label_cls int8 |

设计要点（详见 `data/store.py`）：
- **raw 按股票分片**：断点续传（文件存在即跳过）、单只增量更新（按 date 去重合并）、流式特征构建（内存恒定，约 5000 只全市场数据约 400~600 MB）
- **raw_min 按频率/股票分片**：分钟线字段与日线不同（无 preclose/turn/pctChg，含 time），单独目录隔离，只对部分股票拉取；baostock 仅保留最近约 1 年历史
- **samples 按股票分片**：训练时 `pd.read_parquet("cache/samples")` 直接按目录读取，等价于合并长表，无需维护大文件
- 特征列存 float32（技术指标为近似值），存储与内存均减半

> 注意：前复权数据在发生新的除权除息后，历史价格会整体平移，
> 增量更新（--refresh）只追加新交易日、不改写历史，
> 对训练的绝对价格特征影响可忽略；如需严格一致可定期全量重拉。

### 2. 构建特征与标签

```bash
python main.py features --window 20 --horizon 5
```

有 raw 分片时自动以流式方式逐只构建 samples 分片（内存恒定，
中断后重跑自动跳过已构建股票；`--overwrite` 可强制重建）；
无分片时兼容旧的 `raw_all.parquet` 单文件流程。

### 3. 训练模型

```bash
# 神经网络（LSTM / MLP）
python main.py train --model lstm --epochs 30
python main.py train --model mlp  --epochs 30

# sklearn 基线
python main.py train --model baseline
```

训练结束后会在终端打印测试集评估指标，并保存模型到 `cache/models/`。

### 4. 预测（推理示例）

`predict` 子命令为占位演示。完整推理可直接复用训练逻辑：

```python
from config import cfg
from features.builder import build_dataset, FEATURE_COLUMNS
from features.dataset import build_bundle
from models.train import train_model

# 拉取最新数据、构建特征后，对最新一条样本推理
# 省略：详见 train.py 中的 NNTrainer.predict 用法
```

## 分钟级收盘价预测（级联架构）

用开盘后 30 分钟的 5 分钟线 + 昨日技术指标（含压力位）+ 基座模型预测，
预测当日收盘价。两级模型：

1. **基座模型**：全市场日线训练（horizon=1），预测次日涨跌
2. **分钟模型**（IntradayLSTM）：部分股票的分钟线训练，基座预测作为输入特征级联

```bash
# 0. 数据准备（日线 + 部分股票分钟线，分钟线默认前复权与日线同基准）
python main.py fetch --all
python main.py fetch-min --codes sh.600000,sz.000001 --frequency 5 --start 2025-08-14

# 1. 基座模型（horizon=1）
python main.py features --horizon 1
python main.py train --model lstm --epochs 30
python main.py base-predict                      # 基座全量推理 → cache/meta/base_preds.parquet

# 2. 分钟模型
python main.py min-features                      # 构造分钟样本 → cache/min_samples/5/
python main.py min-train --epochs 30             # 训练 → cache/models/intraday_lstm_5.pt
python main.py min-predict --code sh.600000      # 盘中预测（需 10:00 后运行）
```

min-predict 输出示例（10:00 截面 → 当日收盘）：

```json
{
  "code": "sh.600000",
  "date": "2026-08-13",
  "open": 9.16,
  "price_at_1000": 9.13,
  "predicted_change_pct_from_1000": -0.307,
  "predicted_close": 9.1,
  "prob_close_up_from_1000": 0.439,
  "base_pred_pct": -0.06
}
```

分钟样本（`cache/min_samples/{freq}/{code}.parquet`）每条包含：
- 序列特征 24 维：10:00 前 6 根 5 分钟线 × 4 维（bar 间涨跌 / 相对开盘价 / 量比 / 振幅）
- 静态特征 21 维：昨日 14 个技术特征 + 6 个压力位特征（20/60 日高低点距离、布林带）+ 基座预测
- 标签：`label_rest`（10:00 至收盘涨跌 %）、`label_cls_rest`（涨跌方向）

无泄漏约定：输入只用 ≤10:00 的信息，标签只用 15:00 收盘价；
样本切分严格按全局日期顺序（训练→验证→测试）。

## 外部数据与图结构（存储框架）

为年报/半年报/业绩/分红/汇率等信息预留存储位置与特征化抽象，
以图结构组织关联关系并作为模型输入。数据落盘在 `cache/external/`：

```
cache/external/
├── fundamental/{code}.parquet   # 个股基本面事件长表（每行一个事件）
├── macro/{name}.parquet         # 宏观时间序列（如 fx_usdcny 美元兑人民币）
└── graph/
    ├── nodes.parquet            # 图节点：stock / industry / macro
    └── edges.parquet            # 图边：belongs_to / correlates / affects
```

**基本面事件 schema**（可空字段按事件类型取舍）：
`date`（公告日）, `code`, `event_type`（annual_report / semi_report /
quarterly_report / performance_forecast / dividend）, `revenue`,
`revenue_yoy`, `net_profit`, `net_profit_yoy`, `eps`, `roe`,
`dividend_per_share`, `ex_date`（除权日）, `pay_date`

**宏观序列 schema**：`date` + 数值列（如 `usdcny`），一个指标一个文件。

**图 schema**：节点（node_id/node_type/code/name）+ 有向边（src/dst/edge_type）。

### 外部数据如何影响两类模型（自动检测注入）

| 数据 | 影响基座模型（日线） | 影响分钟模型（盘中） |
|---|---|---|
| 宏观序列（汇率等） | 按样本当日 join（收盘后视角） | 按 T-1 日历日 merge_asof（10:00 盘前视角） |
| 图结构（行业/关联/宏观影响） | 一阶邻居聚合为图特征静态注入 | 暂不注入（可扩展） |
| 基本面事件（财报/分红） | 暂不注入（可扩展） | 每日状态特征静态注入（T-1 视角） |

外部特征以新列进入 samples / min_samples，模型输入维度自动扩展；
无外部数据时行为与之前完全一致。特征化与对齐逻辑见 `features/external.py`。

### 数据准备示例（存储 API，暂不包含拉取）

```python
from data import external_store

# 基本面事件（真实数据可对接 baostock query_profit_data / query_dividend_data）
external_store.save_fundamental_events(events_df, "sh.600000")

# 宏观序列（汇率等，需外部数据源）
external_store.save_macro_series(fx_df, "fx_usdcny")  # date + usdcny 列

# 图结构
external_store.save_graph(nodes_df, edges_df)

# 图特征聚合（一阶邻居）
from data.graph_store import GraphStore
from features.external import build_node_features, load_graph_features

graph = GraphStore.from_disk()
graph.aggregate("sh.600000", build_node_features())  # → 图特征 Series
```

写完数据后重新运行 `python main.py features` / `min-features` 即自动注入。

## 自定义模型

在 `models/nn.py` 中新增网络，并在 `MODEL_REGISTRY` 中注册；或继承
`BaselineEnsemble` 的风格，实现 `fit() / predict()` 接口即可。数据管道与
评估逻辑无需改动。

## 重要提示

- **数据仅用于研究，不构成投资建议**
- 目标标签 `label_reg` 使用未来第 `horizon` 个交易日收盘价计算，
  因此样本表末尾会有若干行标签缺失（已自动剔除）
- 数据集切分在 **时间维度** 顺序进行，不可随机打乱，否则会引入未来信息泄漏
- baostock 获取的数据为复权后价格（默认前复权），适合技术指标计算