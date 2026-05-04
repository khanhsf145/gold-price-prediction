# # """
# # crawler.py — Thu thập dữ liệu tự động cho pipeline dự đoán giá vàng SJC.
# # Nguồn dữ liệu:
# #   - yfinance         : XAUUSD, DXY, VIX, SP500
# #   - 24h.com.vn       : Giá mua/bán vàng SJC (fallback: vietstock.vn)
# #   - exchangerate-api : Tỷ giá USD/VND
# #   - DNSE Entrade API : VN-Index
# # """
# #
# # import os
# # import time
# # import logging
# # import requests
# # import yfinance as yf
# # import pandas as pd
# # from datetime import datetime, timedelta
# # from bs4 import BeautifulSoup
# # from pathlib import Path
# #
# # # ──────────────────────────────────────────────
# # # Cấu hình logging
# # # ──────────────────────────────────────────────
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s [%(levelname)s] %(message)s",
# #     handlers=[
# #         logging.StreamHandler(),
# #         logging.FileHandler("crawler.log", encoding="utf-8"),
# #     ],
# # )
# # logger = logging.getLogger(__name__)
# #
# # # ──────────────────────────────────────────────
# # # Hằng số
# # # ──────────────────────────────────────────────
# # DATA_PATH = Path("data/gold_dataset_v2.csv")
# # MAX_RETRIES = 3
# # RETRY_DELAY = 5  # giây
# #
# # YFINANCE_SYMBOLS: dict[str, str] = {
# #     "xauusd_close": "GC=F",
# #     "dxy_close":    "DX-Y.NYB",
# #     "vix_close":    "^VIX",
# #     "sp500_close":  "^GSPC",
# # }
# #
# # HEADERS = {
# #     "User-Agent": (
# #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
# #         "AppleWebKit/537.36 (KHTML, like Gecko) "
# #         "Chrome/120.0.0.0 Safari/537.36"
# #     )
# # }
# #
# #
# # # ──────────────────────────────────────────────
# # # Tiện ích
# # # ──────────────────────────────────────────────
# # def retry(func):
# #     """Decorator thử lại tối đa MAX_RETRIES lần khi gặp lỗi mạng."""
# #     def wrapper(*args, **kwargs):
# #         for attempt in range(1, MAX_RETRIES + 1):
# #             try:
# #                 return func(*args, **kwargs)
# #             except Exception as exc:
# #                 logger.warning(f"[{func.__name__}] Lần {attempt}/{MAX_RETRIES} thất bại: {exc}")
# #                 if attempt < MAX_RETRIES:
# #                     time.sleep(RETRY_DELAY)
# #         logger.error(f"[{func.__name__}] Đã thất bại sau {MAX_RETRIES} lần thử.")
# #         return None
# #     return wrapper
# #
# #
# # # ──────────────────────────────────────────────
# # # Nguồn 1: yfinance
# # # ──────────────────────────────────────────────
# # @retry
# # def fetch_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
# #     """
# #     Tải XAUUSD, DXY, VIX, SP500 từ yfinance.
# #
# #     Args:
# #         start_date: Ngày bắt đầu (YYYY-MM-DD)
# #         end_date:   Ngày kết thúc  (YYYY-MM-DD)
# #
# #     Returns:
# #         DataFrame với các cột: date, xauusd_close, dxy_close, vix_close, sp500_close
# #     """
# #     logger.info(f"[yfinance] Tải dữ liệu từ {start_date} đến {end_date}...")
# #     frames: list[pd.DataFrame] = []
# #     for col, symbol in YFINANCE_SYMBOLS.items():
# #         ticker = yf.Ticker(symbol)
# #         df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
# #         if df.empty:
# #             logger.warning(f"[yfinance] Không có dữ liệu cho {symbol}")
# #             continue
# #         s = df["Close"].rename(col)
# #         s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
# #         frames.append(s)
# #
# #     if not frames:
# #         raise ValueError("yfinance không trả về dữ liệu nào.")
# #
# #     result = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "date", "Date": "date"})
# #     result["date"] = pd.to_datetime(result["date"]).dt.date
# #     logger.info(f"[yfinance] Đã tải {len(result)} dòng.")
# #     return result
# #
# #
# # # ──────────────────────────────────────────────
# # # Nguồn 2a: SJC từ 24h.com.vn
# # # ──────────────────────────────────────────────
# # @retry
# # def fetch_sjc_24h() -> dict[str, float] | None:
# #     """
# #     Lấy giá vàng SJC (mua/bán) từ 24h.com.vn.
# #
# #     Returns:
# #         Dict {'buy_price': float, 'sell_price': float} hoặc None nếu thất bại.
# #     """
# #     url = "https://www.24h.com.vn/gia-vang-hom-nay-c1000.html"
# #     resp = requests.get(url, headers=HEADERS, timeout=15)
# #     resp.raise_for_status()
# #     soup = BeautifulSoup(resp.text, "html.parser")
# #
# #     # Tìm bảng giá vàng SJC
# #     for row in soup.select("table tr"):
# #         cells = row.select("td")
# #         if len(cells) >= 3 and "SJC" in cells[0].get_text():
# #             try:
# #                 buy  = float(cells[1].get_text(strip=True).replace(",", "").replace(".", "")) / 1_000
# #                 sell = float(cells[2].get_text(strip=True).replace(",", "").replace(".", "")) / 1_000
# #                 logger.info(f"[24h.com.vn] SJC buy={buy}, sell={sell}")
# #                 return {"buy_price": buy, "sell_price": sell}
# #             except ValueError:
# #                 pass
# #     raise ValueError("Không tìm thấy dữ liệu SJC trên 24h.com.vn")
# #
# #
# # # ──────────────────────────────────────────────
# # # Nguồn 2b: SJC từ vietstock.vn (fallback)
# # # ──────────────────────────────────────────────
# # @retry
# # def fetch_sjc_vietstock() -> dict[str, float] | None:
# #     """
# #     Fallback: Lấy giá vàng SJC từ vietstock.vn.
# #
# #     Returns:
# #         Dict {'buy_price': float, 'sell_price': float} hoặc None.
# #     """
# #     url = "https://finance.vietstock.vn/gia-vang.htm"
# #     resp = requests.get(url, headers=HEADERS, timeout=15)
# #     resp.raise_for_status()
# #     soup = BeautifulSoup(resp.text, "html.parser")
# #
# #     for row in soup.select("table.table-gold tr"):
# #         cells = row.find_all("td")
# #         if len(cells) >= 3 and "SJC" in cells[0].get_text():
# #             try:
# #                 buy  = float(cells[1].get_text(strip=True).replace(",", "")) / 1_000
# #                 sell = float(cells[2].get_text(strip=True).replace(",", "")) / 1_000
# #                 logger.info(f"[vietstock] SJC buy={buy}, sell={sell}")
# #                 return {"buy_price": buy, "sell_price": sell}
# #             except ValueError:
# #                 pass
# #     raise ValueError("Không tìm thấy dữ liệu SJC trên vietstock.vn")
# #
# #
# # def fetch_sjc_price() -> dict[str, float] | None:
# #     """Thử 24h.com.vn, fallback sang vietstock.vn."""
# #     result = fetch_sjc_24h()
# #     if result is None:
# #         logger.warning("Fallback sang vietstock.vn...")
# #         result = fetch_sjc_vietstock()
# #     return result
# #
# #
# # # ──────────────────────────────────────────────
# # # Nguồn 3: Tỷ giá USD/VND
# # # ──────────────────────────────────────────────
# # @retry
# # def fetch_usd_vnd() -> float | None:
# #     """
# #     Lấy tỷ giá USD/VND từ exchangerate-api.com (miễn phí).
# #
# #     Returns:
# #         Tỷ giá USD→VND hoặc None.
# #     """
# #     url = "https://open.er-api.com/v6/latest/USD"
# #     resp = requests.get(url, timeout=15)
# #     resp.raise_for_status()
# #     data = resp.json()
# #     rate = data["rates"].get("VND")
# #     if rate:
# #         logger.info(f"[exchangerate-api] USD/VND = {rate}")
# #         return rate
# #     raise ValueError("Không tìm thấy tỷ giá VND trong phản hồi API")
# #
# #
# # # ──────────────────────────────────────────────
# # # Nguồn 4: VN-Index từ DNSE Entrade
# # # ──────────────────────────────────────────────
# # @retry
# # def fetch_vnindex() -> float | None:
# #     """
# #     Lấy điểm VN-Index hiện tại từ DNSE Entrade API.
# #
# #     Returns:
# #         Điểm VN-Index hoặc None.
# #     """
# #     url = "https://api.entrade.com.vn/api/v2/quotes/realtime?symbols=VNINDEX"
# #     resp = requests.get(url, headers=HEADERS, timeout=15)
# #     resp.raise_for_status()
# #     data = resp.json()
# #     if data and "data" in data and len(data["data"]) > 0:
# #         price = data["data"][0].get("lastPrice") or data["data"][0].get("close")
# #         if price:
# #             logger.info(f"[DNSE] VN-Index = {price}")
# #             return float(price)
# #     raise ValueError("Không tìm thấy dữ liệu VN-Index từ DNSE Entrade")
# #
# #
# # # ──────────────────────────────────────────────
# # # Hàm tích hợp & cập nhật CSV
# # # ──────────────────────────────────────────────
# # def load_existing_data() -> pd.DataFrame:
# #     """Tải dữ liệu hiện có từ CSV."""
# #     if DATA_PATH.exists():
# #         df = pd.read_csv(DATA_PATH, parse_dates=["date"])
# #         df["date"] = pd.to_datetime(df["date"]).dt.date
# #         logger.info(f"Đã tải {len(df)} dòng từ {DATA_PATH}")
# #         return df
# #     else:
# #         logger.warning(f"Chưa có file {DATA_PATH}. Sẽ tạo mới.")
# #         return pd.DataFrame()
# #
# #
# # def build_today_row(today: datetime.date) -> dict | None:
# #     """
# #     Xây dựng một dòng dữ liệu cho ngày hôm nay.
# #
# #     Args:
# #         today: Ngày cần lấy dữ liệu.
# #
# #     Returns:
# #         Dict chứa tất cả các trường hoặc None nếu thất bại nghiêm trọng.
# #     """
# #     start_str = today.strftime("%Y-%m-%d")
# #     end_str   = (today + timedelta(days=1)).strftime("%Y-%m-%d")
# #
# #     row: dict = {"date": today}
# #
# #     # --- yfinance ---
# #     yf_df = fetch_yfinance(start_str, end_str)
# #     if yf_df is not None and not yf_df.empty:
# #         for col in ["xauusd_close", "dxy_close", "vix_close", "sp500_close"]:
# #             row[col] = yf_df[col].iloc[-1] if col in yf_df.columns else None
# #     else:
# #         logger.error("Không thể lấy dữ liệu từ yfinance — hủy cập nhật hôm nay.")
# #         return None
# #
# #     # --- SJC ---
# #     sjc = fetch_sjc_price()
# #     if sjc:
# #         row["buy_price"]  = sjc["buy_price"]
# #         row["sell_price"] = sjc["sell_price"]
# #     else:
# #         logger.warning("Không lấy được giá SJC. Dùng NaN.")
# #         row["buy_price"]  = None
# #         row["sell_price"] = None
# #
# #     # --- USD/VND ---
# #     usd_vnd = fetch_usd_vnd()
# #     row["usd_vnd"] = usd_vnd
# #
# #     # --- VN-Index ---
# #     vnindex = fetch_vnindex()
# #     row["vnindex"] = vnindex
# #
# #     return row
# #
# #
# # def update_dataset() -> None:
# #     """
# #     Điểm nhập chính: kiểm tra ngày mới nhất, tải dữ liệu còn thiếu,
# #     và ghi lại vào CSV.
# #     """
# #     df = load_existing_data()
# #     today = datetime.utcnow().date()
# #
# #     if not df.empty:
# #         last_date = pd.to_datetime(df["date"]).max().date()
# #         missing_dates = pd.date_range(
# #             start=last_date + timedelta(days=1),
# #             end=today,
# #             freq="D",
# #         ).date.tolist()
# #     else:
# #         # Tải toàn bộ nếu chưa có file
# #         missing_dates = pd.date_range(start="2010-01-01", end=today, freq="D").date.tolist()
# #
# #     if not missing_dates:
# #         logger.info("Dữ liệu đã cập nhật đến hôm nay. Không cần làm gì thêm.")
# #         return
# #
# #     logger.info(f"Cần cập nhật {len(missing_dates)} ngày từ {missing_dates[0]} đến {missing_dates[-1]}")
# #
# #     new_rows: list[dict] = []
# #     for date in missing_dates:
# #         logger.info(f"Đang xử lý {date}...")
# #         row = build_today_row(date)
# #         if row:
# #             new_rows.append(row)
# #
# #     if not new_rows:
# #         logger.warning("Không có dòng mới nào được thêm.")
# #         return
# #
# #     new_df = pd.DataFrame(new_rows)
# #     column_order = ["date", "buy_price", "sell_price", "xauusd_close",
# #                     "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
# #     for col in column_order:
# #         if col not in new_df.columns:
# #             new_df[col] = None
# #     new_df = new_df[column_order]
# #
# #     combined = pd.concat([df, new_df], ignore_index=True)
# #     combined["date"] = pd.to_datetime(combined["date"])
# #     combined = combined.sort_values("date").drop_duplicates(subset="date")
# #
# #     DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
# #     combined.to_csv(DATA_PATH, index=False)
# #     logger.info(f"✅ Đã lưu {len(combined)} dòng vào {DATA_PATH}")
# #
# #
# # if __name__ == "__main__":
# #     update_dataset()
#
#
# import requests
# import json
# import time
# import datetime
# import logging
# from typing import Optional
#
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger("AURUM.Crawler")
#
# # ── Timeout mặc định (giây) ─────────────────────────────────
# TIMEOUT = 12
# HEADERS = {
#     "User-Agent": (
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) "
#         "Chrome/124.0.0.0 Safari/537.36"
#     ),
#     "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
# }
#
#
# # ══════════════════════════════════════════════════════════════
# # 1. GIÁ VÀNG SJC
# # ══════════════════════════════════════════════════════════════
# def fetch_sjc() -> Optional[dict]:
#     """
#     Trả về {"buy": float, "sell": float} theo đơn vị triệu VNĐ/lượng.
#     Nguồn chính: 24h.com.vn — scrape từ bảng giá theo data-seach="sjc"
#     Đơn vị gốc trên web: x1000đ/lượng  →  chia 1000 = triệu VNĐ/lượng
#     """
#     try:
#         from bs4 import BeautifulSoup
#         url = "https://www.24h.com.vn/gia-vang-hom-nay-c425.html"
#         r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
#         r.raise_for_status()
#         soup = BeautifulSoup(r.text, "html.parser")
#
#         sjc_row = soup.find("tr", attrs={"data-seach": "sjc"})
#         if sjc_row:
#             # Lấy tất cả các span.fixW (chứa giá)
#             prices = sjc_row.find_all("span", class_="fixW")
#             if len(prices) >= 2:
#                 # prices[0] = giá mua, prices[1] = giá bán
#                 buy_text = prices[0].get_text(strip=True).replace(",", "").replace(".", "")
#                 sell_text = prices[1].get_text(strip=True).replace(",", "").replace(".", "")
#
#                 # Chuyển đơn vị từ nghìn VNĐ/lượng thành triệu VNĐ/lượng
#                 buy = float(buy_text) / 1_000 if buy_text.isdigit() else None
#                 sell = float(sell_text) / 1_000 if sell_text.isdigit() else None
#
#                 if buy and sell and sell > 0:
#                     log.info(f"24h.com.vn SJC → mua={buy:.3f}M, bán={sell:.3f}M")
#                     return {"buy": buy, "sell": sell}
#
#         log.warning("24h.com.vn: không tìm thấy dòng SJC")
#         return None
#     except Exception as e:
#         log.warning(f"24h.com.vn lỗi ({e})")
#         return None
#
#
# # ══════════════════════════════════════════════════════════════
# # 2. XAU/USD, DXY, S&P 500, VIX, USD/VNĐ
# # ══════════════════════════════════════════════════════════════
# def _yahoo_latest(symbol: str) -> Optional[float]:
#     """
#     Lấy giá đóng cửa gần nhất từ Yahoo Finance Chart API (không cần thư viện yfinance).
#     Dùng endpoint /v8/finance/chart/ với interval=1d, range=5d.
#     """
#     url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
#     params = {
#         "interval": "1d",
#         "range": "5d",
#         "includePrePost": "false",
#     }
#     headers = {**HEADERS, "Accept": "application/json"}
#     try:
#         r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
#         r.raise_for_status()
#         body = r.json()
#
#         result = body["chart"]["result"][0]
#         closes = result["indicators"]["quote"][0]["close"]
#         # Lấy giá close cuối cùng có giá trị (không None)
#         val = next((v for v in reversed(closes) if v is not None), None)
#         if val:
#             log.info(f"Yahoo {symbol} → {val:.4f}")
#         else:
#             log.warning(f"Yahoo {symbol}: không có dữ liệu")
#         return val
#     except Exception as e:
#         log.warning(f"Yahoo {symbol} lỗi: {e}")
#         return None
#
#
# def fetch_xauusd() -> Optional[float]:
#     """Giá vàng thế giới XAU/USD (USD/oz) từ Yahoo Finance (GC=F hoặc XAUUSD=X)"""
#     val = _yahoo_latest("GC=F")  # Gold Futures — dữ liệu phổ biến nhất
#     if val is None:
#         val = _yahoo_latest("XAUUSD=X")  # Spot XAU/USD
#     return val
#
#
# def fetch_dxy() -> Optional[float]:
#     """Dollar Index DXY từ Yahoo Finance"""
#     return _yahoo_latest("DX-Y.NYB")
#
#
# def fetch_sp500() -> Optional[float]:
#     """S&P 500 Index từ Yahoo Finance"""
#     return _yahoo_latest("^GSPC")
#
#
# def fetch_vix() -> Optional[float]:
#     """VIX (CBOE Volatility Index) từ Yahoo Finance"""
#     return _yahoo_latest("^VIX")
#
#
# # ══════════════════════════════════════════════════════════════
# # 3. TỶ GIÁ USD/VNĐ
# # ══════════════════════════════════════════════════════════════
# def fetch_usd_vnd() -> Optional[float]:
#     # Nguồn 1: Yahoo Finance USDVND=X
#     val = _yahoo_latest("USDVND=X")
#     if val:
#         return val
#
#     # Nguồn 2: exchangerate-api.com (miễn phí, không cần key)
#     try:
#         r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=TIMEOUT)
#         r.raise_for_status()
#         val = r.json()["rates"].get("VND")
#         if val:
#             log.info(f"ExchangeRate-API USD/VNĐ → {val:.0f}")
#             return float(val)
#     except Exception as e:
#         log.warning(f"ExchangeRate-API lỗi: {e}")
#
#     return None
#
#
# # ══════════════════════════════════════════════════════════════
# # 4. VN-INDEX
# # ══════════════════════════════════════════════════════════════
#
# def fetch_vnindex() -> Optional[float]:
#     # Nguồn 1: API DNSE/Entrade
#     val = _fetch_vnindex_entrade()
#     if val:
#         return val
#
#     # Nguồn 2: vn.investing.com
#     return _fetch_vnindex_investing()
#
#
# def _fetch_vnindex_entrade() -> Optional[float]:
#     """
#     Lấy VN-Index từ API của DNSE/Entrade.
#     """
#     try:
#         import time
#         to_ts = int(time.time())
#         from_ts = to_ts - 7 * 24 * 3600
#         url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index?resolution=1D&symbol=VNINDEX&from={from_ts}&to={to_ts}"
#         r = requests.get(url, timeout=TIMEOUT)
#         r.raise_for_status()
#         data = r.json()
#         if "c" in data and len(data["c"]) > 0:
#             val = float(data["c"][-1])
#             log.info(f"Entrade VN-Index → {val:.2f}")
#             return val
#         else:
#             log.warning("Entrade VN-Index: không có dữ liệu")
#             return None
#     except Exception as e:
#         log.warning(f"Entrade VN-Index lỗi: {e}")
#         return None
#
#
# def _fetch_vnindex_investing() -> Optional[float]:
#     """
#     Scrape VN-Index từ https://vn.investing.com/indices/vn bằng BeautifulSoup.
#     Selector: [data-test="instrument-price-last"]
#     """
#     try:
#         from bs4 import BeautifulSoup
#         url = "https://vn.investing.com/indices/vn"
#         r = requests.get(url, headers={
#             **HEADERS,
#             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#         }, timeout=TIMEOUT)
#         r.raise_for_status()
#         soup = BeautifulSoup(r.text, "html.parser")
#
#         elem = soup.find("div", attrs={"data-test": "instrument-price-last"})
#         if elem:
#             text = elem.get_text(strip=True).replace(",", "").replace(".", "")
#             # Xử lý text như "1,853.29" → "1853.29"
#             # Thực tế sẽ là "185329" sau replace, cần xử lý hợp lý
#             # Text gốc từ 24h: "1,853.29" nhưng vn.investing có thể khác format
#             # Cách an toàn: lấy text gốc, thay "," thành "", rồi convert
#             raw_text = elem.get_text(strip=True)
#             # Có thể là "1,853.29" hoặc "1853.29"
#             cleaned = raw_text.replace(",", ".") if "," in raw_text else raw_text
#             # Nếu có 2 dấu chấm, xử lý theo chuẩn châu Âu (1.234,56)
#             if cleaned.count(".") > 1:
#                 cleaned = cleaned.replace(".", "").replace(",", ".")
#             val = float(cleaned)
#             if val > 0:
#                 log.info(f"vn.investing.com VN-Index → {val:.2f}")
#                 return val
#         else:
#             log.warning("vn.investing.com: không tìm thấy [data-test='instrument-price-last']")
#             return None
#
#     except Exception as e:
#         log.warning(f"vn.investing.com lỗi: {e}")
#         return None
#
#
# # ══════════════════════════════════════════════════════════════
# # MAIN — Crawl tất cả và trả về dict
# # ══════════════════════════════════════════════════════════════
# def crawl_latest() -> dict:
#     """
#     Crawl toàn bộ dữ liệu mới nhất.
#     Trả về dict với đầy đủ các trường, hoặc None cho trường không lấy được.
#
#     Args:
#         df: DataFrame lịch sử (dùng để carry-forward lãi suất & CPI)
#
#     Returns:
#         dict: {
#             "date"         : str (YYYY-MM-DD),
#             "buy_price"    : float | None,
#             "sell_price"   : float | None,
#             "xauusd_close" : float | None,
#             "dxy_close"    : float | None,
#             "sp500_close"  : float | None,
#             "vix_close"    : float | None,
#             "usd_vnd"      : float | None,
#             "vnindex"      : float | None,
#             "errors"       : list[str],   # danh sách trường bị lỗi
#             "warnings"     : list[str],   # trường dùng fallback/carry-forward
#         }
#     """
#     log.info("═" * 55)
#     log.info("Bắt đầu crawl dữ liệu mới nhất…")
#     log.info("═" * 55)
#
#     today = datetime.date.today().isoformat()
#     result = {"date": today, "errors": [], "warnings": []}
#
#     # ── 1. Vàng SJC ──────────────────────────────────────────
#     log.info("1/7  Giá vàng SJC (24h.com.vn)")
#     sjc = fetch_sjc()
#     if sjc:
#         result["buy_price"] = sjc["buy"]
#         result["sell_price"] = sjc["sell"]
#     else:
#         result["buy_price"] = None
#         result["sell_price"] = None
#         result["errors"].append("sjc")
#         log.error("FAIL: Không lấy được giá SJC")
#
#     time.sleep(0.5)  # Lịch sự với server
#
#     # ── 2. XAU/USD ───────────────────────────────────────────
#     log.info("2/7  XAU/USD (Yahoo Finance GC=F)")
#     result["xauusd_close"] = fetch_xauusd()
#     if result["xauusd_close"] is None:
#         result["errors"].append("xauusd")
#
#     time.sleep(0.3)
#
#     # ── 3. DXY ───────────────────────────────────────────────
#     log.info("3/7  Dollar Index DXY (Yahoo Finance)")
#     result["dxy_close"] = fetch_dxy()
#     if result["dxy_close"] is None:
#         result["errors"].append("dxy")
#
#     time.sleep(0.3)
#
#     # ── 4. S&P 500 ───────────────────────────────────────────
#     log.info("4/7  S&P 500 (Yahoo Finance ^GSPC)")
#     result["sp500_close"] = fetch_sp500()
#     if result["sp500_close"] is None:
#         result["errors"].append("sp500")
#
#     time.sleep(0.3)
#
#     # ── 5. VIX ───────────────────────────────────────────────
#     log.info("5/7  VIX (Yahoo Finance ^VIX)")
#     result["vix_close"] = fetch_vix()
#     if result["vix_close"] is None:
#         result["errors"].append("vix")
#
#     time.sleep(0.3)
#
#     # ── 6. USD/VNĐ ───────────────────────────────────────────
#     log.info("6/7  Tỷ giá USD/VNĐ (Yahoo Finance USDVND=X)")
#     result["usd_vnd"] = fetch_usd_vnd()
#     if result["usd_vnd"] is None:
#         result["errors"].append("usd_vnd")
#
#     time.sleep(0.3)
#
#     # ── 7. VN-Index ──────────────────────────────────────────
#     log.info("7/7  VN-Index (DNSE/Entrade API, vn.investing.com)")
#     result["vnindex"] = fetch_vnindex()
#     if result["vnindex"] is None:
#         result["errors"].append("vnindex")
#
#     # ── Tổng kết ─────────────────────────────────────────────
#     ok_fields = [k for k, v in result.items()
#                  if k not in ("date", "errors", "warnings") and v is not None]
#     fail_fields = result["errors"]
#
#     log.info("═" * 55)
#     log.info(f"Crawl xong: {len(ok_fields)}/7 trường thành công")
#     if fail_fields:
#         log.warning(f"Lỗi: {fail_fields}")
#     log.info("═" * 55)
#
#     return result
#
#
# # ══════════════════════════════════════════════════════════════
# # UPDATE DATASET — Thêm dòng mới vào CSV
# # ══════════════════════════════════════════════════════════════
# def update_dataset(csv_path: str = "data/gold_dataset_v2.csv") -> dict:
#     """
#     Crawl dữ liệu hôm nay và append/update vào file CSV.
#     Trả về dict kết quả crawl.
#     """
#     import pandas as pd
#     import os
#
#     # Đọc dataset hiện tại
#     df = None
#     if os.path.exists(csv_path):
#         df = pd.read_csv(csv_path, encoding="utf-8-sig")
#         df["date"] = pd.to_datetime(df["date"])
#         log.info(f"Dataset hiện tại: {len(df)} dòng, đến {df['date'].max().date()}")
#     else:
#         log.warning(f"Không tìm thấy {csv_path}, sẽ tạo mới")
#
#     # Crawl
#     data = crawl_latest()
#     today_str = data["date"]
#
#     # Cập nhật (ghi đè) nếu đã có ngày hôm nay trong dataset để lấy giá mới nhất
#     if df is not None:
#         existing = df[df["date"] == pd.Timestamp(today_str)]
#         if not existing.empty:
#             log.info(f"Ngày {today_str} đã có trong dataset. Cập nhật dòng bằng dữ liệu mới nhất.")
#             df = df[df["date"] != pd.Timestamp(today_str)]
#
#     # Điền carry-forward cho trường còn None (từ dòng cuối dataset)
#     if df is not None:
#         last = df.iloc[-1]
#         for field in ["xauusd_close", "dxy_close", "sp500_close", "vix_close",
#                       "usd_vnd", "vnindex"]:
#             if data.get(field) is None:
#                 data[field] = float(last[field])
#                 data["warnings"].append(f"{field} (carry-forward vì lỗi crawl)")
#                 log.warning(f"{field}: carry-forward từ {last['date'].date()} = {last[field]}")
#
#     # Giá SJC bắt buộc phải có
#     if data.get("sell_price") is None:
#         log.error("Không thể cập nhật dataset: không có giá SJC")
#         return {"status": "failed_no_sjc", **data}
#
#     # Tạo dòng mới
#     new_row = {
#         "date": today_str,
#         "buy_price": round(data["buy_price"], 3),
#         "sell_price": round(data["sell_price"], 3),
#         "xauusd_close": round(data["xauusd_close"], 4),
#         "dxy_close": round(data["dxy_close"], 4),
#         "sp500_close": round(data["sp500_close"], 4),
#         "vix_close": round(data["vix_close"], 4),
#         "usd_vnd": round(data["usd_vnd"], 0),
#         "vnindex": round(data["vnindex"], 2),
#     }
#
#     # Append vào CSV
#     new_df = pd.DataFrame([new_row])
#     if df is not None:
#         combined = pd.concat([df.assign(date=df["date"].dt.strftime("%Y-%m-%d")),
#                               new_df], ignore_index=True)
#     else:
#         combined = new_df
#
#     combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
#     log.info(f"✅ Đã thêm ngày {today_str} vào {csv_path} ({len(combined)} dòng tổng)")
#
#     return {"status": "success", "new_row": new_row, **data}
#
#
# # ══════════════════════════════════════════════════════════════
# # CHẠY THỬ TRỰC TIẾP
# # ══════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     result = update_dataset("data/gold_dataset_v2.csv")
#     print("\n=== KẾT QUẢ CRAWL ===")
#     for k, v in result.items():
#         if k not in ("errors", "warnings", "new_row"):
#             print(f"  {k:20s}: {v}")
#     if result.get("errors"):
#         print(f"\n⚠ Lỗi   : {result['errors']}")
#     if result.get("warnings"):
#         print(f"⚠ Cảnh báo: {result['warnings']}")






#!/usr/bin/env python3
"""
crawler.py — Thu thập dữ liệu giá vàng SJC và các chỉ số thị trường.
Hỗ trợ fallback nhiều nguồn, retry, log đầy đủ.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

# Đảm bảo file log luôn tồn tại
with open("crawler.log", "w", encoding="utf-8") as f:
    f.write(f"Crawl started at {datetime.utcnow().isoformat()}\n")

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Crawler")

DATA_PATH = Path("data/gold_dataset_v2.csv")
TIMEOUT = 15
MAX_RETRIES = 2
RETRY_DELAY = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ------------------------------
# 1. Giá vàng SJC (24h + vietstock fallback)
# ------------------------------
def fetch_sjc_24h() -> Optional[Dict[str, float]]:
    """Lấy từ 24h.com.vn (dùng selector hiện tại)"""
    try:
        from bs4 import BeautifulSoup
        url = "https://www.24h.com.vn/gia-vang-hom-nay-c425.html"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        row = soup.find("tr", attrs={"data-seach": "sjc"})
        if row:
            prices = row.find_all("span", class_="fixW")
            if len(prices) >= 2:
                buy_text = prices[0].get_text(strip=True).replace(",", "").replace(".", "")
                sell_text = prices[1].get_text(strip=True).replace(",", "").replace(".", "")
                buy = float(buy_text) / 1000 if buy_text.isdigit() else None
                sell = float(sell_text) / 1000 if sell_text.isdigit() else None
                if buy and sell:
                    logger.info(f"SJC 24h: mua={buy:.1f}, bán={sell:.1f}")
                    return {"buy": buy, "sell": sell}
        logger.warning("24h: không tìm thấy SJC")
        return None
    except Exception as e:
        logger.warning(f"24h lỗi: {e}")
        return None

def fetch_sjc_vietstock() -> Optional[Dict[str, float]]:
    """Fallback từ vietstock.vn"""
    try:
        from bs4 import BeautifulSoup
        url = "https://finance.vietstock.vn/gia-vang.htm"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for row in soup.select("table.table-gold tr"):
            cells = row.find_all("td")
            if len(cells) >= 3 and "SJC" in cells[0].get_text():
                buy = float(cells[1].get_text(strip=True).replace(",", "")) / 1000
                sell = float(cells[2].get_text(strip=True).replace(",", "")) / 1000
                logger.info(f"SJC vietstock: mua={buy:.1f}, bán={sell:.1f}")
                return {"buy": buy, "sell": sell}
        logger.warning("vietstock: không tìm thấy SJC")
        return None
    except Exception as e:
        logger.warning(f"vietstock lỗi: {e}")
        return None

def fetch_sjc() -> Optional[Dict[str, float]]:
    res = fetch_sjc_24h()
    if not res:
        res = fetch_sjc_vietstock()
    return res

# ------------------------------
# 2. Yahoo Finance (dùng API trực tiếp, không yfinance)
# ------------------------------
def _yahoo_latest(symbol: str) -> Optional[float]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": "5d"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        val = next((v for v in reversed(closes) if v is not None), None)
        if val:
            logger.info(f"Yahoo {symbol} → {val:.4f}")
        return val
    except Exception as e:
        logger.warning(f"Yahoo {symbol} lỗi: {e}")
        return None

def fetch_xauusd() -> Optional[float]:
    val = _yahoo_latest("GC=F")
    if not val:
        val = _yahoo_latest("XAUUSD=X")
    return val

def fetch_dxy() -> Optional[float]:
    return _yahoo_latest("DX-Y.NYB")

def fetch_sp500() -> Optional[float]:
    return _yahoo_latest("^GSPC")

def fetch_vix() -> Optional[float]:
    return _yahoo_latest("^VIX")

# ------------------------------
# 3. USD/VND
# ------------------------------
def fetch_usd_vnd() -> Optional[float]:
    # Yahoo
    val = _yahoo_latest("USDVND=X")
    if val:
        return val
    # Fallback exchangerate-api
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=TIMEOUT)
        r.raise_for_status()
        val = r.json()["rates"].get("VND")
        if val:
            logger.info(f"USD/VND API → {val:.0f}")
            return float(val)
    except Exception as e:
        logger.warning(f"USD/VND fallback lỗi: {e}")
    return None

# ------------------------------
# 4. VN-Index
# ------------------------------
def fetch_vnindex() -> Optional[float]:
    # DNSE/Entrade
    try:
        ts = int(time.time())
        url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index?resolution=1D&symbol=VNINDEX&from={ts-7*86400}&to={ts}"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if "c" in data and data["c"]:
            val = float(data["c"][-1])
            logger.info(f"VN-Index Entrade → {val:.2f}")
            return val
    except Exception as e:
        logger.warning(f"Entrade VN-Index lỗi: {e}")
    # Fallback investing.com
    try:
        from bs4 import BeautifulSoup
        url = "https://vn.investing.com/indices/vn"
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(r.text, "html.parser")
        elem = soup.find("div", attrs={"data-test": "instrument-price-last"})
        if elem:
            text = elem.get_text(strip=True).replace(",", ".").replace(".", "")
            val = float(text)
            logger.info(f"VN-Index investing → {val:.2f}")
            return val
    except Exception as e:
        logger.warning(f"Investing VN-Index lỗi: {e}")
    return None

# ------------------------------
# 5. Crawl và cập nhật CSV
# ------------------------------
def load_existing_data(df_path: Path) -> pd.DataFrame:
    if df_path.exists():
        df = pd.read_csv(df_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        logger.info(f"Đã đọc {len(df)} dòng từ {df_path}")
        return df
    return pd.DataFrame()

def update_dataset(csv_path: str = "data/gold_dataset_v2.csv") -> Dict[str, Any]:
    """Crawl dữ liệu hôm nay, append vào CSV, trả về kết quả."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.utcnow().date().isoformat()
    result = {
        "date": today,
        "buy_price": None,
        "sell_price": None,
        "xauusd_close": None,
        "dxy_close": None,
        "sp500_close": None,
        "vix_close": None,
        "usd_vnd": None,
        "vnindex": None,
        "errors": [],
        "warnings": []
    }

    # 1. SJC
    sjc = fetch_sjc()
    if sjc:
        result["buy_price"] = sjc["buy"]
        result["sell_price"] = sjc["sell"]
    else:
        result["errors"].append("sjc")

    # 2-7. Các chỉ số
    result["xauusd_close"] = fetch_xauusd()
    result["dxy_close"] = fetch_dxy()
    result["sp500_close"] = fetch_sp500()
    result["vix_close"] = fetch_vix()
    result["usd_vnd"] = fetch_usd_vnd()
    result["vnindex"] = fetch_vnindex()

    # Kiểm tra lỗi
    for field in ["xauusd_close", "dxy_close", "sp500_close", "vix_close", "usd_vnd", "vnindex"]:
        if result[field] is None:
            result["errors"].append(field)

    # Nếu không có SJC -> bỏ qua
    if result["sell_price"] is None:
        logger.error("Không lấy được giá SJC, bỏ qua cập nhật hôm nay.")
        return {"status": "failed_no_sjc", **result}

    # Đọc dữ liệu cũ
    old_df = load_existing_data(path)
    new_row = {
        "date": today,
        "buy_price": result["buy_price"],
        "sell_price": result["sell_price"],
        "xauusd_close": result["xauusd_close"],
        "dxy_close": result["dxy_close"],
        "sp500_close": result["sp500_close"],
        "vix_close": result["vix_close"],
        "usd_vnd": result["usd_vnd"],
        "vnindex": result["vnindex"],
    }

    # Nếu đã có ngày hôm nay trong CSV, xóa dòng cũ
    if not old_df.empty:
        old_df = old_df[old_df["date"] != pd.Timestamp(today)]

    new_df = pd.DataFrame([new_row])
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").drop_duplicates(subset="date")
    combined.to_csv(path, index=False, encoding="utf-8-sig")

    logger.info(f"✅ Đã lưu {len(combined)} dòng vào {path}")
    return {"status": "success", "new_row": new_row, **result}

if __name__ == "__main__":
    res = update_dataset()
    print("\n=== KẾT QUẢ CRAWL ===")
    for k, v in res.items():
        if k not in ("errors", "warnings"):
            print(f"  {k:20s}: {v}")
    if res.get("errors"):
        print(f"\n⚠️ LỖI: {res['errors']}")