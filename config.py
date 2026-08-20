"""全局配置：数据周期、复权方式、预测周期、特征与模型参数等。"""

from dataclasses import dataclass, field
from pathlib import Path


# 项目根目录
ROOT = Path(__file__).resolve().parent

# 数据缓存目录
DATA_DIR = ROOT / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # ---- 数据 ----
    start_date: str = "2021-08-01"          # 拉取开始日期（近 5 年）
    end_date: str = ""                      # 拉取结束日期（空 = 今天）
    adjust: str = "2"                       # 复权方式：1=后复权 2=前复权 3=不复权
    frequency: str = "d"                    # 周期：d=日线 w=周线 m=月线
    fetch_sleep: float = 0.3                # 批量拉取时每只股票之间的间隔秒数
    include_delisted: bool = True           # 全量拉取时是否包含退市股（避免幸存者偏差）
    min_frequency: str = "5"                # 分钟线频率：5/15/30/60
    # 分钟模型辅助训练股（面板板块，与 sz.000100 走势联动）：
    # 仅参与联合训练，不做 serving
    min_aux_codes: list = field(
        default_factory=lambda: ["sz.000725", "sh.600707"]
    )

    # ---- 标签 / 预测 ----
    horizon: int = 5                        # 预测未来 N 个交易日

    # ---- 特征 ----
    window: int = 20                        # 回看窗口，用于技术指标

    # ---- 数据集切分（严格按时间顺序，防止泄漏）----
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    # 其余为测试集

    # ---- 模型 ----
    model: str = "lstm"                     # lstm | mlp | baseline
    hidden_dim: int = 64                    # NN 隐藏层维度
    num_layers: int = 2                     # LSTM 层数
    seq_len: int = 10                       # LSTM 输入的序列长度（回溯步数）
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42

    # ---- 运行 ----
    device: str = "auto"                    # auto | cpu | cuda | mps

    # ---- 回测（网格交易 + 多窗口）----
    bt_capital: float = 100000.0            # 初始资金（元）
    bt_windows: list = field(
        default_factory=lambda: ["5y", "3y", "2y", "1y"]
    )
    bt_train_ratio: float = 0.8             # 窗口内训练段占比（5y 窗口 ≈ 4年训练 + 1年回测）
    bt_grid_n: int = 10                     # 网格数量（每格资金 = capital / grid_n）
    bt_range_pct: float = 0.20              # 网格上下界 = 回测首日收盘价 × (1±range_pct)
    bt_gate_on: bool = True                 # 模型信号门控：预测下跌时只卖不买
    bt_gate_threshold: float = 0.5          # 上涨概率门控阈值
    bt_commission: float = 2.5e-4           # 佣金（双边，万 2.5）
    bt_min_commission: float = 5.0          # 单笔最低佣金（元）
    bt_stamp_tax: float = 5e-4              # 印花税（仅卖出，0.05%）
    bt_slippage: float = 1e-3               # 滑点（成交价按比例偏移）
    bt_save_every: int = 10                 # 训练 checkpoint 间隔（epoch）

    # ---- L5 预测输出 / 定时服务 ----
    schedule_hour: int = 9                  # launchd 调度小时（09:50 截面后）
    schedule_minute: int = 51               # launchd 调度分钟（09:50 bar 收盘留 1 分钟同步余量）
    train_hour: int = 8                     # 增量训练调度小时（开盘前）
    train_minute: int = 30                  # 增量训练调度分钟
    push_enabled: bool = True               # 是否推送信号
    push_channel: str = "auto"              # 推送渠道：auto/wecom/pushplus
    wecom_webhook: str = ""                 # 建议用 cache/.env 的 WECOM_WEBHOOK 配置
    pushplus_token: str = ""                # 建议用 cache/.env 的 PUSHPLUS_TOKEN 配置
    pushplus_topic: str = ""                # pushplus 一对多频道代码（群发，可选）

    # 示例股票池（可扩展为全市场）
    default_codes: list = field(
        default_factory=lambda: [
            "sh.600000", "sh.600519", "sh.601318",
            "sz.000001", "sz.000333", "sz.000858",
        ]
    )


cfg = Config()