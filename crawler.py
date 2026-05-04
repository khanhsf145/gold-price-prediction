"""
crawler.py — Thu thập dữ liệu tự động cho pipeline dự đoán giá vàng SJC.
Nguồn dữ liệu:
  - yfinance         : XAUUSD, DXY, VIX, SP500
  - 24h.com.vn       : Giá mua/bán vàng SJC (fallback: vietstock.vn)
  - exchangerate-api : Tỷ giá USD/VND
  - DNSE Entrade API : VN-Index
"""

import os
import time
import logging
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from pathlib import Path

# ──────────────────────────────────────────────
# Cấu hình logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────
DATA_PATH = Path("data/gold_dataset_v2.csv")
MAX_RETRIES = 3
RETRY_DELAY = 5  # giây

YFINANCE_SYMBOLS: dict[str, str] = {
    "xauusd_close": "GC=F",
    "dxy_close":    "DX-Y.NYB",
    "vix_close":    "^VIX",
    "sp500_close":  "^GSPC",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ──────────────────────────────────────────────
# Tiện ích
# ──────────────────────────────────────────────
def retry(func):
    """Decorator thử lại tối đa MAX_RETRIES lần khi gặp lỗi mạng."""
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.warning(f"[{func.__name__}] Lần {attempt}/{MAX_RETRIES} thất bại: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        logger.error(f"[{func.__name__}] Đã thất bại sau {MAX_RETRIES} lần thử.")
        return None
    return wrapper


# ──────────────────────────────────────────────
# Nguồn 1: yfinance
# ──────────────────────────────────────────────
@retry
def fetch_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Tải XAUUSD, DXY, VIX, SP500 từ yfinance.

    Args:
        start_date: Ngày bắt đầu (YYYY-MM-DD)
        end_date:   Ngày kết thúc  (YYYY-MM-DD)

    Returns:
        DataFrame với các cột: date, xauusd_close, dxy_close, vix_close, sp500_close
    """
    logger.info(f"[yfinance] Tải dữ liệu từ {start_date} đến {end_date}...")
    frames: list[pd.DataFrame] = []
    for col, symbol in YFINANCE_SYMBOLS.items():
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            logger.warning(f"[yfinance] Không có dữ liệu cho {symbol}")
            continue
        s = df["Close"].rename(col)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        frames.append(s)

    if not frames:
        raise ValueError("yfinance không trả về dữ liệu nào.")

    result = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "date", "Date": "date"})
    result["date"] = pd.to_datetime(result["date"]).dt.date
    logger.info(f"[yfinance] Đã tải {len(result)} dòng.")
    return result


# ──────────────────────────────────────────────
# Nguồn 2a: SJC từ 24h.com.vn
# ──────────────────────────────────────────────
@retry
def fetch_sjc_24h() -> dict[str, float] | None:
    """
    Lấy giá vàng SJC (mua/bán) từ 24h.com.vn.

    Returns:
        Dict {'buy_price': float, 'sell_price': float} hoặc None nếu thất bại.
    """
    url = "https://www.24h.com.vn/gia-vang-hom-nay-c1000.html"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Tìm bảng giá vàng SJC
    for row in soup.select("table tr"):
        cells = row.select("td")
        if len(cells) >= 3 and "SJC" in cells[0].get_text():
            try:
                buy  = float(cells[1].get_text(strip=True).replace(",", "").replace(".", "")) / 1_000
                sell = float(cells[2].get_text(strip=True).replace(",", "").replace(".", "")) / 1_000
                logger.info(f"[24h.com.vn] SJC buy={buy}, sell={sell}")
                return {"buy_price": buy, "sell_price": sell}
            except ValueError:
                pass
    raise ValueError("Không tìm thấy dữ liệu SJC trên 24h.com.vn")


# ──────────────────────────────────────────────
# Nguồn 2b: SJC từ vietstock.vn (fallback)
# ──────────────────────────────────────────────
@retry
def fetch_sjc_vietstock() -> dict[str, float] | None:
    """
    Fallback: Lấy giá vàng SJC từ vietstock.vn.

    Returns:
        Dict {'buy_price': float, 'sell_price': float} hoặc None.
    """
    url = "https://finance.vietstock.vn/gia-vang.htm"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for row in soup.select("table.table-gold tr"):
        cells = row.find_all("td")
        if len(cells) >= 3 and "SJC" in cells[0].get_text():
            try:
                buy  = float(cells[1].get_text(strip=True).replace(",", "")) / 1_000
                sell = float(cells[2].get_text(strip=True).replace(",", "")) / 1_000
                logger.info(f"[vietstock] SJC buy={buy}, sell={sell}")
                return {"buy_price": buy, "sell_price": sell}
            except ValueError:
                pass
    raise ValueError("Không tìm thấy dữ liệu SJC trên vietstock.vn")


def fetch_sjc_price() -> dict[str, float] | None:
    """Thử 24h.com.vn, fallback sang vietstock.vn."""
    result = fetch_sjc_24h()
    if result is None:
        logger.warning("Fallback sang vietstock.vn...")
        result = fetch_sjc_vietstock()
    return result


# ──────────────────────────────────────────────
# Nguồn 3: Tỷ giá USD/VND
# ──────────────────────────────────────────────
@retry
def fetch_usd_vnd() -> float | None:
    """
    Lấy tỷ giá USD/VND từ exchangerate-api.com (miễn phí).

    Returns:
        Tỷ giá USD→VND hoặc None.
    """
    url = "https://open.er-api.com/v6/latest/USD"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    rate = data["rates"].get("VND")
    if rate:
        logger.info(f"[exchangerate-api] USD/VND = {rate}")
        return rate
    raise ValueError("Không tìm thấy tỷ giá VND trong phản hồi API")


# ──────────────────────────────────────────────
# Nguồn 4: VN-Index từ DNSE Entrade
# ──────────────────────────────────────────────
@retry
def fetch_vnindex() -> float | None:
    """
    Lấy điểm VN-Index hiện tại từ DNSE Entrade API.

    Returns:
        Điểm VN-Index hoặc None.
    """
    url = "https://api.entrade.com.vn/api/v2/quotes/realtime?symbols=VNINDEX"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data and "data" in data and len(data["data"]) > 0:
        price = data["data"][0].get("lastPrice") or data["data"][0].get("close")
        if price:
            logger.info(f"[DNSE] VN-Index = {price}")
            return float(price)
    raise ValueError("Không tìm thấy dữ liệu VN-Index từ DNSE Entrade")


# ──────────────────────────────────────────────
# Hàm tích hợp & cập nhật CSV
# ──────────────────────────────────────────────
def load_existing_data() -> pd.DataFrame:
    """Tải dữ liệu hiện có từ CSV."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        logger.info(f"Đã tải {len(df)} dòng từ {DATA_PATH}")
        return df
    else:
        logger.warning(f"Chưa có file {DATA_PATH}. Sẽ tạo mới.")
        return pd.DataFrame()


def build_today_row(today: datetime.date) -> dict | None:
    """
    Xây dựng một dòng dữ liệu cho ngày hôm nay.

    Args:
        today: Ngày cần lấy dữ liệu.

    Returns:
        Dict chứa tất cả các trường hoặc None nếu thất bại nghiêm trọng.
    """
    start_str = today.strftime("%Y-%m-%d")
    end_str   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    row: dict = {"date": today}

    # --- yfinance ---
    yf_df = fetch_yfinance(start_str, end_str)
    if yf_df is not None and not yf_df.empty:
        for col in ["xauusd_close", "dxy_close", "vix_close", "sp500_close"]:
            row[col] = yf_df[col].iloc[-1] if col in yf_df.columns else None
    else:
        logger.error("Không thể lấy dữ liệu từ yfinance — hủy cập nhật hôm nay.")
        return None

    # --- SJC ---
    sjc = fetch_sjc_price()
    if sjc:
        row["buy_price"]  = sjc["buy_price"]
        row["sell_price"] = sjc["sell_price"]
    else:
        logger.warning("Không lấy được giá SJC. Dùng NaN.")
        row["buy_price"]  = None
        row["sell_price"] = None

    # --- USD/VND ---
    usd_vnd = fetch_usd_vnd()
    row["usd_vnd"] = usd_vnd

    # --- VN-Index ---
    vnindex = fetch_vnindex()
    row["vnindex"] = vnindex

    return row


def update_dataset() -> None:
    """
    Điểm nhập chính: kiểm tra ngày mới nhất, tải dữ liệu còn thiếu,
    và ghi lại vào CSV.
    """
    df = load_existing_data()
    today = datetime.utcnow().date()

    if not df.empty:
        last_date = pd.to_datetime(df["date"]).max().date()
        missing_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            end=today,
            freq="D",
        ).date.tolist()
    else:
        # Tải toàn bộ nếu chưa có file
        missing_dates = pd.date_range(start="2010-01-01", end=today, freq="D").date.tolist()

    if not missing_dates:
        logger.info("Dữ liệu đã cập nhật đến hôm nay. Không cần làm gì thêm.")
        return

    logger.info(f"Cần cập nhật {len(missing_dates)} ngày từ {missing_dates[0]} đến {missing_dates[-1]}")

    new_rows: list[dict] = []
    for date in missing_dates:
        logger.info(f"Đang xử lý {date}...")
        row = build_today_row(date)
        if row:
            new_rows.append(row)

    if not new_rows:
        logger.warning("Không có dòng mới nào được thêm.")
        return

    new_df = pd.DataFrame(new_rows)
    column_order = ["date", "buy_price", "sell_price", "xauusd_close",
                    "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
    for col in column_order:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[column_order]

    combined = pd.concat([df, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").drop_duplicates(subset="date")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    logger.info(f"✅ Đã lưu {len(combined)} dòng vào {DATA_PATH}")


if __name__ == "__main__":
    update_dataset()
