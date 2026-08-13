"""baostock 外部数据拉取：基本面事件 + 宏观利率序列。

基本面事件（个股，分片缓存到 cache/external/fundamental/）：
- 分红      query_dividend_data             → event_type="dividend"
- 季频财报  query_profit_data / growth_data → annual_report / semi_report / quarterly_report
- 业绩预告  query_forecast_report            → performance_forecast
- 业绩快报  query_performance_express_report → 按统计期归入财报事件类型

宏观序列（全市场共用，缓存到 cache/external/macro/）：
- deposit_rate           存款基准利率（活期/定期/零存整取）
- loan_rate              贷款基准利率 + 房贷利率
- required_reserve_ratio 存款准备金率（大/中型金融机构，按生效日）
- money_supply           货币供应量 M0/M1/M2（月度，按次月 15 日对齐避免前视）

对齐与无泄漏约定：
- 基本面事件 date = 公告日（信息截止基准）；分红事件另带 ex_date/pay_date，
  特征化时按除权日对齐（见 features/external.py）
- 宏观序列统一 reindex 到交易日历并 ffill，保证交易日频特征语义；
  注入时 merge_asof_previous_day 按 T-1 日历日 backward 合并，自动规避未来泄漏
- 单位约定：revenue=百万元、net_profit=万元、eps=元、dividend_per_share=元、
  roe / *_yoy = %（growth/快报的比例字段已 ×100）
- baostock 当前版本无 Shibor / 汇率接口，需其他数据源（如 akshare）
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import baostock as bs

from data import external_store
from data import store as data_store
from data.external_store import FUNDAMENTAL_COLUMNS
from data.fetcher import BaoStockClient, get_all_codes

logger = logging.getLogger(__name__)

# ---- 宏观序列列名重命名（baostock 字段 → 简洁英文列名）----
_DEPOSIT_RENAME = {
    "pubDate": "date",
    "demandDepositRate": "deposit_demand",
    "fixedDepositRate3Month": "deposit_fixed_3m",
    "fixedDepositRate6Month": "deposit_fixed_6m",
    "fixedDepositRate1Year": "deposit_fixed_1y",
    "fixedDepositRate2Year": "deposit_fixed_2y",
    "fixedDepositRate3Year": "deposit_fixed_3y",
    "fixedDepositRate5Year": "deposit_fixed_5y",
    "installmentFixedDepositRate1Year": "deposit_install_1y",
    "installmentFixedDepositRate3Year": "deposit_install_3y",
    "installmentFixedDepositRate5Year": "deposit_install_5y",
}

_LOAN_RENAME = {
    "pubDate": "date",
    "loanRate6Month": "loan_6m",
    "loanRate6MonthTo1Year": "loan_6m_1y",
    "loanRate1YearTo3Year": "loan_1y_3y",
    "loanRate3YearTo5Year": "loan_3y_5y",
    "loanRateAbove5Year": "loan_above_5y",
    "mortgateRateBelow5Year": "mortgage_below_5y",
    "mortgateRateAbove5Year": "mortgage_above_5y",
}


# ---- 基本面事件 ----

def _event_type_from_stat_date(stat_date: str) -> str:
    """按统计截止日归类财报事件：年报 / 中报 / 季报。"""
    md = str(stat_date)[5:10]
    if md == "12-31":
        return "annual_report"
    if md == "06-30":
        return "semi_report"
    return "quarterly_report"


def fetch_dividend_events(
    client: BaoStockClient, code: str, start_year: int, end_year: int
) -> pd.DataFrame:
    """拉取分红送配事件（按预案公告日对齐，除权日/派息日另存）。"""
    parts = []
    for year in range(start_year - 1, end_year + 1):
        raw = client.query(
            bs.query_dividend_data, code=code, year=str(year), yearType="report"
        )
        if not raw.empty:
            parts.append(raw)
    if not parts:
        return pd.DataFrame()

    raw = pd.concat(parts, ignore_index=True)

    def to_dt(col: str) -> pd.Series:
        return pd.to_datetime(raw[col].replace("", pd.NaT), errors="coerce")

    # 信息截止基准：公告日缺失时回退股东大会日 / 除权除息日
    date = to_dt("dividPlanAnnounceDate")
    date = date.fillna(to_dt("dividAgmPumDate")).fillna(to_dt("dividOperateDate"))
    return pd.DataFrame({
        "date": date,
        "event_type": "dividend",
        "dividend_per_share": pd.to_numeric(
            raw["dividCashPsBeforeTax"], errors="coerce"
        ),
        "ex_date": to_dt("dividOperateDate"),
        "pay_date": to_dt("dividPayDate"),
    })


def fetch_profit_events(
    client: BaoStockClient, code: str, start_year: int, end_year: int
) -> pd.DataFrame:
    """拉取季频财报（盈利能力 + 成长能力同比），按公告日对齐。"""
    profits, growths = [], []
    for year in range(start_year - 1, end_year + 1):
        for quarter in (1, 2, 3, 4):
            p = client.query(
                bs.query_profit_data, code=code, year=year, quarter=quarter
            )
            if not p.empty:
                profits.append(p)
            g = client.query(
                bs.query_growth_data, code=code, year=year, quarter=quarter
            )
            if not g.empty:
                growths.append(g)
    if not profits:
        return pd.DataFrame()

    profit = pd.concat(profits, ignore_index=True)
    out = pd.DataFrame({
        "date": pd.to_datetime(profit["pubDate"], errors="coerce"),
        "stat_date": profit["statDate"],
        "event_type": profit["statDate"].map(_event_type_from_stat_date),
        "revenue": pd.to_numeric(profit["MBRevenue"], errors="coerce"),
        "net_profit": pd.to_numeric(profit["netProfit"], errors="coerce"),
        "eps": pd.to_numeric(profit["epsTTM"], errors="coerce"),
        "roe": pd.to_numeric(profit["roeAvg"], errors="coerce") * 100.0,
    })
    if growths:
        growth = pd.concat(growths, ignore_index=True)
        growth = growth.rename(columns={"statDate": "stat_date"})
        growth["net_profit_yoy"] = (
            pd.to_numeric(growth["YOYNI"], errors="coerce") * 100.0
        )
        out = out.merge(growth[["stat_date", "net_profit_yoy"]],
                         on="stat_date", how="left")
    else:
        out["net_profit_yoy"] = np.nan
    return out.drop(columns=["stat_date"])


def fetch_forecast_events(
    client: BaoStockClient, code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """拉取业绩预告事件（净利润增减幅度上限作为 net_profit_yoy，单位 %）。"""
    raw = client.query(
        bs.query_forecast_report,
        code=code, start_date=start_date, end_date=end_date,
    )
    if raw.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "date": pd.to_datetime(raw["profitForcastExpPubDate"], errors="coerce"),
        "event_type": "performance_forecast",
        "net_profit_yoy": pd.to_numeric(
            raw["profitForcastChgPctUp"], errors="coerce"
        ),
    })


def fetch_express_events(
    client: BaoStockClient, code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """拉取业绩快报事件（按统计期归入财报事件类型，比例字段 ×100 转 %）。"""
    raw = client.query(
        bs.query_performance_express_report,
        code=code, start_date=start_date, end_date=end_date,
    )
    if raw.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "date": pd.to_datetime(raw["performanceExpPubDate"], errors="coerce"),
        "event_type": raw["performanceExpStatDate"].map(
            _event_type_from_stat_date
        ),
        "eps": pd.to_numeric(raw["performanceExpressEPSDiluted"], errors="coerce"),
        "roe": pd.to_numeric(raw["performanceExpressROEWa"], errors="coerce"),
        "revenue_yoy": pd.to_numeric(
            raw["performanceExpressGRYOY"], errors="coerce"
        ) * 100.0,
        "net_profit_yoy": pd.to_numeric(
            raw["performanceExpressOPYOY"], errors="coerce"
        ) * 100.0,
    })


def fetch_fundamental_events(
    client: BaoStockClient, code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """拉取单只股票全部基本面事件，输出 external_store 规定的 schema。"""
    start_year, end_year = int(start_date[:4]), int(end_date[:4])
    frames = [
        fetch_dividend_events(client, code, start_year, end_year),
        fetch_profit_events(client, code, start_year, end_year),
        fetch_forecast_events(client, code, start_date, end_date),
        fetch_express_events(client, code, start_date, end_date),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    for col in FUNDAMENTAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out.dropna(subset=["date"]).drop_duplicates()
    return out[FUNDAMENTAL_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_fundamental(
    code: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    refresh: bool = False,
    client: Optional[BaoStockClient] = None,
) -> pd.DataFrame:
    """拉取单只股票基本面事件，优先从 fundamental 分片缓存读取。

    缓存行为与 fetch_stock 一致：已有分片且 refresh=False 直接返回；
    无数据时保存空表，避免重复空拉。
    """
    cache_file = external_store.fundamental_path(code)
    if use_cache and cache_file.exists() and not refresh:
        try:
            df = pd.read_parquet(cache_file)
            logger.info("命中基本面缓存 %s", cache_file.name)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("缓存读取失败 %s: %s", cache_file, exc)

    owns_client = client is None
    client = client or BaoStockClient()
    try:
        df = fetch_fundamental_events(client, code, start_date, end_date)
    finally:
        if owns_client:
            client.logout()

    external_store.save_fundamental_events(df, code)
    logger.info("已保存 %s 基本面事件 %d 条", code, len(df))
    return df


def fetch_fundamental_many(
    codes: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    refresh: bool = False,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """批量拉取多只股票基本面事件，按 code 拼接为一张长表。"""
    frames: List[pd.DataFrame] = []
    with BaoStockClient() as client:
        for i, code in enumerate(codes):
            logger.info("拉取基本面 %s (%d/%d)", code, i + 1, len(codes))
            try:
                df = fetch_fundamental(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                    refresh=refresh,
                    client=client,
                )
                if not df.empty:
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error("拉取 %s 基本面失败: %s", code, exc)
            if i < len(codes) - 1 and sleep > 0:
                time.sleep(sleep)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def fetch_all_fundamental(
    start_date: str,
    end_date: str,
    sleep: float = 0.3,
    include_delisted: bool = True,
    refresh: bool = False,
) -> Dict[str, int]:
    """拉取全市场股票基本面事件，写入 fundamental 分片缓存。

    断点续传：已有分片的股票自动跳过（refresh=False）；
    失败代码记录到 cache/meta/failed_fundamental_codes.txt。
    """
    codes_df = get_all_codes(include_delisted=include_delisted)
    if codes_df.empty:
        logger.error("未获取到股票清单")
        return {"total": 0, "ok": 0, "skipped": 0, "failed": 0}

    # 过滤区间内不存在交易的股票（尚未上市 / 早已退市）
    if "ipoDate" in codes_df.columns:
        codes_df = codes_df[
            (codes_df["ipoDate"].fillna("") == "")
            | (codes_df["ipoDate"] <= end_date)
        ]
    if "outDate" in codes_df.columns:
        codes_df = codes_df[
            (codes_df["outDate"].fillna("") == "")
            | (codes_df["outDate"] >= start_date)
        ]

    codes = codes_df["code"].tolist()
    stats = {"total": len(codes), "ok": 0, "skipped": 0, "failed": 0}
    failed_codes: List[str] = []
    t0 = time.time()

    with BaoStockClient() as client:
        for i, code in enumerate(codes):
            # 断点续传：分片已存在且非 refresh 模式，直接跳过
            if external_store.fundamental_path(code).exists() and not refresh:
                stats["skipped"] += 1
            else:
                try:
                    fetch_fundamental(
                        code=code,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=True,
                        refresh=refresh,
                        client=client,
                    )
                    stats["ok"] += 1
                except Exception as exc:  # noqa: BLE001
                    stats["failed"] += 1
                    failed_codes.append(code)
                    logger.error("拉取 %s 基本面失败: %s", code, exc)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / max(i + 1, 1) * (len(codes) - i - 1)
                logger.info(
                    "基本面进度 %d/%d | ok=%d skipped=%d failed=%d | "
                    "用时 %.1fmin 预计剩余 %.1fmin",
                    i + 1, len(codes), stats["ok"], stats["skipped"],
                    stats["failed"], elapsed / 60, eta / 60,
                )
            if i < len(codes) - 1 and sleep > 0:
                time.sleep(sleep)

    logger.info(
        "全市场基本面拉取完成：total=%d ok=%d skipped=%d failed=%d，耗时 %.1fmin",
        stats["total"], stats["ok"], stats["skipped"], stats["failed"],
        (time.time() - t0) / 60,
    )
    if failed_codes:
        failed_file = data_store.META_DIR / "failed_fundamental_codes.txt"
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        failed_file.write_text("\n".join(failed_codes))
        logger.warning("失败代码已记录到 %s（重跑 fetch-external --all 即可续传）", failed_file)
    return stats


# ---- 宏观利率序列 ----

def _trade_dates(
    client: BaoStockClient, start_date: str, end_date: str
) -> pd.DatetimeIndex:
    """拉取区间内交易日历（用于宏观序列对齐到交易日频）。"""
    raw = client.query(
        bs.query_trade_dates, start_date=start_date, end_date=end_date
    )
    if raw.empty:
        raise RuntimeError("获取交易日历失败")
    dates = pd.to_datetime(
        raw.loc[raw["is_trading_day"] == "1", "calendar_date"], errors="coerce"
    )
    return pd.DatetimeIndex(dates.dropna().sort_values())


def _to_daily(df: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """宏观事件表 → 交易日频序列（ffill 最近已知值）。"""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    value_cols = [c for c in out.columns if c != "date"]
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.set_index("date").sort_index()
    out = out.reindex(out.index.union(trade_dates)).ffill().reindex(trade_dates)
    out = out.reset_index()
    # index 的 name 可能被 trade_dates 继承（如 calendar_date），统一重命名
    out = out.rename(columns={out.columns[0]: "date"})
    return out


def fetch_macro_series(
    client: BaoStockClient, start_date: str, end_date: str
) -> Dict[str, pd.DataFrame]:
    """拉取全部宏观利率序列，reindex 到交易日历并 ffill。"""
    trade_dates = _trade_dates(client, start_date, end_date)
    series: Dict[str, pd.DataFrame] = {}

    # 存/贷款基准利率 2015 年后已不再调整，从更早年份查询，
    # 让历史利率 ffill 延续到训练区间（否则近 5 年区间内为空）
    raw = client.query(
        bs.query_deposit_rate_data, start_date="1990-01-01", end_date=end_date
    )
    if not raw.empty:
        series["deposit_rate"] = _to_daily(
            raw.rename(columns=_DEPOSIT_RENAME), trade_dates
        )

    raw = client.query(
        bs.query_loan_rate_data, start_date="1990-01-01", end_date=end_date
    )
    if not raw.empty:
        series["loan_rate"] = _to_daily(
            raw.rename(columns=_LOAN_RENAME), trade_dates
        )

    raw = client.query(bs.query_required_reserve_ratio_data)
    if not raw.empty:
        rrr = pd.DataFrame({
            "date": pd.to_datetime(raw["effectiveDate"], errors="coerce"),
            "rrr_big": pd.to_numeric(
                raw["bigInstitutionsRatioAfter"], errors="coerce"
            ),
            "rrr_medium": pd.to_numeric(
                raw["mediumInstitutionsRatioAfter"], errors="coerce"
            ),
        })
        series["required_reserve_ratio"] = _to_daily(rrr, trade_dates)

    # 往前多查一年，保证训练区间起点处已有月度值可 ffill
    raw = client.query(
        bs.query_money_supply_data_month,
        start_date=f"{int(start_date[:4]) - 1}-{start_date[5:7]}",
        end_date=f"{end_date[:4]}-{end_date[5:7]}",
    )
    if not raw.empty:
        per = pd.to_datetime(
            raw["statYear"] + "-" + raw["statMonth"], format="%Y-%m"
        )
        # 月度数据实际在下旬公布，按次月 15 日对齐，避免前视
        ms = pd.DataFrame({
            "date": per + pd.offsets.MonthEnd(0) + pd.offsets.Day(15),
            "m0": pd.to_numeric(raw["m0Month"], errors="coerce"),
            "m0_yoy": pd.to_numeric(raw["m0YOY"], errors="coerce"),
            "m1": pd.to_numeric(raw["m1Month"], errors="coerce"),
            "m1_yoy": pd.to_numeric(raw["m1YOY"], errors="coerce"),
            "m2": pd.to_numeric(raw["m2Month"], errors="coerce"),
            "m2_yoy": pd.to_numeric(raw["m2YOY"], errors="coerce"),
        })
        series["money_supply"] = _to_daily(ms, trade_dates)

    return series


def fetch_macro_all(
    start_date: str, end_date: str, client: Optional[BaoStockClient] = None
) -> Dict[str, int]:
    """拉取并保存全部宏观利率序列，返回 {序列名: 行数}。"""
    owns_client = client is None
    client = client or BaoStockClient()
    try:
        series = fetch_macro_series(client, start_date, end_date)
    finally:
        if owns_client:
            client.logout()

    stats: Dict[str, int] = {}
    for name, df in series.items():
        external_store.save_macro_series(df, name)
        stats[name] = len(df)
        logger.info("宏观序列 %s 已保存 %d 行", name, len(df))
    return stats
