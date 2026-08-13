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

    # 示例股票池（可扩展为全市场）
    default_codes: list = field(
        default_factory=lambda: [
            "sh.600000", "sh.600519", "sh.601318",
            "sz.000001", "sz.000333", "sz.000858",
        ]
    )


cfg = Config()