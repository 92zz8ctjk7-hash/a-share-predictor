"""baostock 数据接口封装。

特性：
- 封装登录 / 登出，避免重复登录
- 拉取 A 股历史 K 线（日/周/月）与全市场股票清单
- 本地 Parquet 分片缓存（见 data/store.py），支持断点续传与增量更新
- 输出标准化的 pandas DataFrame（字段已在 schema 中统一）
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from data import store
from data.schema import bars_from_frame, min_bars_from_frame

logger = logging.getLogger(__name__)


class BaoStockClient:
    """baostock 客户端封装，自动管理登录状态。"""

    def __init__(self) -> None:
        self._bs = None
        self._logged_in = False

    def _ensure_client(self):
        if self._bs is None:
            import baostock as bs
            self._bs = bs
        return self._bs

    def login(self) -> None:
        if self._logged_in:
            return
        bs = self._ensure_client()
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
        self._logged_in = True
        logger.info("baostock 登录成功")

    def logout(self) -> None:
        if not self._logged_in:
            return
        bs = self._ensure_client()
        bs.logout()
        self._logged_in = False
        logger.info("baostock 已登出")

    def query_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust: str = "2",
    ) -> pd.DataFrame:
        """查询单只股票的历史 K 线。

        参数：
            code      : 股票代码，带市场前缀，如 sh.600000
            start_date: 开始日期 YYYY-MM-DD
            end_date  : 结束日期 YYYY-MM-DD
            frequency : d/w/m 日周月线，或 5/15/30/60 分钟线
            adjust    : 1=后复权 2=前复权 3=不复权

        返回统一字段名的 DataFrame：
        - 日线含 date/code/open/...；分钟线含 date/time/code/open/...
          （分钟线无 preclose/turn/pctChg，baostock 不返回）
        """
        self.login()
        bs = self._ensure_client()

        is_min = frequency in ("5", "15", "30", "60")

        if is_min:
            # 分钟线字段与日线不同（含 time，无 preclose/turn/pctChg）
            fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
        elif adjust == "3":
            # 注意：baostock 的换手率字段名为 turn，且仅在不复权(3)时可取；
            # 复权模式下请求该字段会报错，因此按 adjust 动态裁剪字段。
            fields = (
                "date,code,open,high,low,close,preclose,"
                "volume,amount,turn,pctChg"
            )
        else:
            fields = (
                "date,code,open,high,low,close,preclose,"
                "volume,amount,pctChg"
            )

        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjust,
        )
        if rs.error_code != "0":
            raise RuntimeError(f"查询 {code} 失败: {rs.error_msg}")

        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        raw = pd.DataFrame(rows, columns=rs.fields)
        if is_min:
            return min_bars_from_frame(raw)
        return bars_from_frame(raw)

    def query_stock_basic(self) -> pd.DataFrame:
        """获取全部证券基本资料（股票/指数/债券等），返回原始字段。

        字段：code, code_name, ipoDate, outDate, type, status
        其中 type: 1=股票 2=指数 3=其它；status: 1=上市 0=退市
        """
        self.login()
        bs = self._ensure_client()

        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            raise RuntimeError(f"获取证券基本资料失败: {rs.error_msg}")

        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=rs.fields)

    def query(self, query_fn, **kwargs) -> pd.DataFrame:
        """通用查询：调用任意 baostock 接口函数，返回 DataFrame。

        参数：
            query_fn : baostock 模块的查询函数，如 bs.query_dividend_data
            kwargs   : 传给查询函数的参数

        空结果返回空 DataFrame；接口报错抛 RuntimeError。
        """
        self.login()
        rs = query_fn(**kwargs)
        if rs.error_code != "0":
            raise RuntimeError(f"查询失败: {rs.error_msg}")

        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame(columns=rs.fields)
        return pd.DataFrame(rows, columns=rs.fields)

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()


def fetch_stock(
    code: str,
    start_date: str,
    end_date: str,
    frequency: str = "d",
    adjust: str = "2",
    use_cache: bool = True,
    refresh: bool = False,
    client: Optional[BaoStockClient] = None,
) -> pd.DataFrame:
    """拉取单只股票历史数据，优先从 raw 分片缓存读取。

    缓存行为：
    - 已有分片且 refresh=False：直接返回缓存（断点续传）
    - refresh=True：从缓存最后日期次日增量拉取，与旧数据按 date 去重合并
    - 无分片：按 [start_date, end_date] 全量拉取
    """
    cache_file = store.raw_path(code)
    do_merge = False
    old_df: Optional[pd.DataFrame] = None

    if use_cache and cache_file.exists():
        if not refresh:
            try:
                df = pd.read_parquet(cache_file)
                logger.info("命中缓存 %s", cache_file.name)
                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("缓存读取失败 %s: %s", cache_file, exc)
        else:
            try:
                old_df = pd.read_parquet(cache_file)
                last = old_df["date"].max()
                new_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if new_start > end_date:
                    logger.info("已是最新数据，跳过 %s", code)
                    return old_df
                start_date = new_start
                do_merge = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "缓存读取失败 %s: %s，将全量重拉", cache_file, exc
                )

    owns_client = client is None
    client = client or BaoStockClient()
    try:
        df = client.query_k_data(
            code=code,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjust=adjust,
        )
    finally:
        if owns_client:
            client.logout()

    if df.empty:
        # 增量模式下无新数据：返回已有完整数据
        if do_merge and old_df is not None:
            return old_df
        return df

    if use_cache:
        store.save_bars(df, code, merge=do_merge)
        logger.info("已缓存 %s（%d 行）", cache_file.name, len(df))

    return df


def fetch_many(
    codes: List[str],
    start_date: str,
    end_date: str,
    frequency: str = "d",
    adjust: str = "2",
    use_cache: bool = True,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """批量拉取多只股票历史数据，按 code 拼接为一张长表。

    返回字段与单只一致，额外包含 code 列用于区分。
    """
    frames: List[pd.DataFrame] = []
    with BaoStockClient() as client:
        for i, code in enumerate(codes):
            logger.info("拉取 %s (%d/%d)", code, i + 1, len(codes))
            try:
                df = fetch_stock(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjust=adjust,
                    use_cache=use_cache,
                    client=client,
                )
                if not df.empty:
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error("拉取 %s 失败: %s", code, exc)
            if i < len(codes) - 1 and sleep > 0:
                time.sleep(sleep)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def fetch_minutes(
    code: str,
    start_date: str,
    end_date: str,
    frequency: str = "5",
    adjust: str = "2",
    use_cache: bool = True,
    refresh: bool = False,
    client: Optional[BaoStockClient] = None,
) -> pd.DataFrame:
    """拉取单只股票分钟线数据，优先从 raw_min 分片缓存读取。

    缓存行为与 fetch_stock 一致：
    - 已有分片且 refresh=False：直接返回缓存（断点续传）
    - refresh=True：从缓存最后日期次日增量拉取，按 (date, time) 去重合并
    - 无分片：按 [start_date, end_date] 全量拉取

    注意：
    - baostock 分钟线仅保留最近约 1 年历史
    - 默认前复权（adjust="2"），与日线 raw 保持同一价格基准；
      如需不复权的真实盘中价格，显式传 adjust="3"
    """
    cache_file = store.min_path(code, frequency)
    do_merge = False
    old_df: Optional[pd.DataFrame] = None

    if use_cache and cache_file.exists():
        if not refresh:
            try:
                df = pd.read_parquet(cache_file)
                logger.info("命中分钟线缓存 %s", cache_file.name)
                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("缓存读取失败 %s: %s", cache_file, exc)
        else:
            try:
                old_df = pd.read_parquet(cache_file)
                last_date = old_df["date"].max()
                new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if new_start > end_date:
                    logger.info("分钟线已是最新，跳过 %s", code)
                    return old_df
                start_date = new_start
                do_merge = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "缓存读取失败 %s: %s，将全量重拉", cache_file, exc
                )

    owns_client = client is None
    client = client or BaoStockClient()
    try:
        df = client.query_k_data(
            code=code,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjust=adjust,
        )
    finally:
        if owns_client:
            client.logout()

    if df.empty:
        # 增量模式下无新数据：返回已有完整数据
        if do_merge and old_df is not None:
            return old_df
        return df

    if use_cache:
        store.save_min_bars(df, code, frequency, merge=do_merge)
        logger.info("已缓存分钟线 %s（%d 行）", cache_file.name, len(df))

    return df


def fetch_min_many(
    codes: List[str],
    start_date: str,
    end_date: str,
    frequency: str = "5",
    adjust: str = "2",
    use_cache: bool = True,
    refresh: bool = False,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """批量拉取多只股票的分钟线数据，按 code 拼接为一张长表。

    适合只对部分股票存储分钟级数据的场景；每只股票一个分片文件
    （cache/raw_min/{frequency}/{code}.parquet），失败自动跳过。
    """
    frames: List[pd.DataFrame] = []
    failed: List[str] = []
    with BaoStockClient() as client:
        for i, code in enumerate(codes):
            logger.info("拉取分钟线 %s %smin (%d/%d)", code, frequency, i + 1, len(codes))
            try:
                df = fetch_minutes(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjust=adjust,
                    use_cache=use_cache,
                    refresh=refresh,
                    client=client,
                )
                if not df.empty:
                    frames.append(df)
                else:
                    logger.warning("拉取 %s 分钟线无数据", code)
            except Exception as exc:  # noqa: BLE001
                failed.append(code)
                logger.error("拉取 %s 分钟线失败: %s", code, exc)
            if i < len(codes) - 1 and sleep > 0:
                time.sleep(sleep)

    if failed:
        logger.warning("分钟线拉取失败 %d 只: %s", len(failed), failed)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["code", "date", "time"]).reset_index(drop=True)


def get_all_codes(
    include_delisted: bool = True,
    client: Optional[BaoStockClient] = None,
) -> pd.DataFrame:
    """获取全市场 A 股股票清单（query_stock_basic）。

    返回字段：code, code_name, ipoDate, outDate, type, status
    已过滤 type == "1"（仅股票，排除指数/基金/债券）。
    include_delisted=True 时保留退市股（status == "0"），
    用于训练可避免幸存者偏差。
    """
    owns_client = client is None
    client = client or BaoStockClient()
    try:
        df = client.query_stock_basic()
    finally:
        if owns_client:
            client.logout()

    if df.empty:
        return df

    df = df[df["type"] == "1"].copy()
    if not include_delisted:
        df = df[df["status"] == "1"]
    df = df.sort_values("code").reset_index(drop=True)
    n_delisted = int((df["status"] == "0").sum())
    logger.info(
        "全市场股票 %d 只（其中退市 %d 只）", len(df), n_delisted
    )
    return df


def _log_progress(done: int, total: int, stats: Dict[str, int], t0: float) -> None:
    elapsed = time.time() - t0
    eta = elapsed / max(done, 1) * (total - done)
    logger.info(
        "进度 %d/%d | ok=%d skipped=%d failed=%d | 用时 %.1fmin 预计剩余 %.1fmin",
        done, total, stats["ok"], stats["skipped"], stats["failed"],
        elapsed / 60, eta / 60,
    )


def fetch_all(
    start_date: str,
    end_date: str,
    frequency: str = "d",
    adjust: str = "2",
    sleep: float = 0.3,
    include_delisted: bool = True,
    refresh: bool = False,
) -> Dict[str, int]:
    """拉取全市场股票历史数据，写入 raw 分片缓存（见 data/store.py）。

    - 股票清单取自 query_stock_basic，保存到 cache/meta/stock_list.parquet
    - 断点续传：已有分片的股票自动跳过（refresh=False 时），
      中断后重跑同一命令即可继续
    - refresh=True 时对已有分片做增量更新（拉取缓存最后日期之后的数据）
    - 区间内从未上市 / 早已退市的股票自动过滤，减少无效请求
    - 失败代码记录到 cache/meta/failed_codes.txt

    返回统计字典：{total, ok, skipped, failed}
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
    store.save_stock_list(codes_df)

    codes = codes_df["code"].tolist()
    stats = {"total": len(codes), "ok": 0, "skipped": 0, "failed": 0}
    failed_codes: List[str] = []
    t0 = time.time()

    with BaoStockClient() as client:
        for i, code in enumerate(codes):
            # 断点续传：分片已存在且非 refresh 模式，直接跳过
            if store.raw_path(code).exists() and not refresh:
                stats["skipped"] += 1
                if (i + 1) % 100 == 0:
                    _log_progress(i + 1, len(codes), stats, t0)
                continue

            try:
                df = fetch_stock(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjust=adjust,
                    use_cache=True,
                    refresh=refresh,
                    client=client,
                )
                if df.empty:
                    logger.warning("拉取 %s 无数据（可能长期停牌）", code)
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                stats["failed"] += 1
                failed_codes.append(code)
                logger.error("拉取 %s 失败: %s", code, exc)

            if (i + 1) % 100 == 0:
                _log_progress(i + 1, len(codes), stats, t0)
            if i < len(codes) - 1 and sleep > 0:
                time.sleep(sleep)

    logger.info(
        "全量拉取完成：total=%d ok=%d skipped=%d failed=%d，耗时 %.1fmin",
        stats["total"], stats["ok"], stats["skipped"], stats["failed"],
        (time.time() - t0) / 60,
    )
    if failed_codes:
        failed_file = store.META_DIR / "failed_codes.txt"
        failed_file.write_text("\n".join(failed_codes))
        logger.warning(
            "失败代码已记录到 %s（重跑 fetch --all 即可续传）", failed_file
        )
    return stats