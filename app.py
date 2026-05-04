# """
# app.py — Ứng dụng Streamlit dự báo giá vàng SJC.
# Tất cả văn bản UI bằng tiếng Việt.
# Tải mô hình đã huấn luyện sẵn từ models/ để suy luận nhanh.
# """
#
# import json
# import warnings
# import logging
# from pathlib import Path
# from datetime import datetime
#
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
# from statsmodels.tsa.stattools import adfuller
#
# warnings.filterwarnings("ignore")
#
# # ──────────────────────────────────────────────
# # Cấu hình trang
# # ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="🥇 Dự Báo Giá Vàng SJC",
#     page_icon="🥇",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
#
# # ──────────────────────────────────────────────
# # Logging
# # ──────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # ──────────────────────────────────────────────
# # Hằng số
# # ──────────────────────────────────────────────
# DATA_PATH  = Path("data/gold_dataset_v2.csv")
# MODELS_DIR = Path("models")
# IMG_DIR    = Path("img")
# HORIZONS   = [1, 3, 7]
# TARGET_COL = "sell_price"
# EXOG_COLS  = ["xauusd_close", "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
#
# HORIZON_LABELS = {1: "1 ngày", 3: "3 ngày", 7: "7 ngày"}
#
# # ──────────────────────────────────────────────
# # CSS tùy chỉnh
# # ──────────────────────────────────────────────
# st.markdown("""
# <style>
#     .main-title { font-size: 2.2rem; font-weight: 700; color: #FFB703; }
#     .metric-card {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#         border: 1px solid #FFB703;
#         border-radius: 12px;
#         padding: 1rem;
#         text-align: center;
#         margin: 0.3rem;
#     }
#     .metric-value { font-size: 1.6rem; font-weight: 700; color: #FFB703; }
#     .metric-label { font-size: 0.85rem; color: #aaa; }
#     .info-box {
#         background: #0f3460;
#         border-left: 4px solid #FFB703;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .warn-box {
#         background: #3d1515;
#         border-left: 4px solid #E63946;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .success-box {
#         background: #0d3b2e;
#         border-left: 4px solid #2A9D8F;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # ══════════════════════════════════════════════
# # HÀM TIỆN ÍCH
# # ══════════════════════════════════════════════
#
# @st.cache_data(show_spinner="Đang tải dữ liệu...")
# def load_data() -> pd.DataFrame:
#     """Tải và tiền xử lý dataset."""
#     df = pd.read_csv(DATA_PATH, parse_dates=["date"])
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values("date").set_index("date")
#     full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
#     df = df.reindex(full_range).ffill().bfill()
#     return df
#
#
# @st.cache_resource(show_spinner="Đang tải mô hình...")
# def load_model_artifacts(horizon: int) -> dict:
#     """
#     Tải scaler, RandomForest, LightGBM và metadata cho một horizon.
#
#     Args:
#         horizon: Horizon h (1, 3 hoặc 7).
#
#     Returns:
#         Dict {'scaler', 'rf', 'lgbm', 'metadata'} hoặc {} nếu chưa có model.
#     """
#     prefix = MODELS_DIR / f"h{horizon}"
#     artifacts = {}
#     try:
#         artifacts["scaler"]   = joblib.load(f"{prefix}_scaler.joblib")
#         artifacts["rf"]       = joblib.load(f"{prefix}_rf.joblib")
#         artifacts["lgbm"]     = joblib.load(f"{prefix}_lgbm.joblib")
#         with open(f"{prefix}_metadata.json", "r", encoding="utf-8") as f:
#             artifacts["metadata"] = json.load(f)
#     except FileNotFoundError:
#         logger.warning(f"Chưa tìm thấy model artifacts cho H={horizon}.")
#     return artifacts
#
#
# def build_features_for_inference(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
#     """
#     Xây dựng đặc trưng CHỐNG RÒ RỈ — logic y hệt pipeline.py.
#     Hàm này phải GIỐNG HỆT hàm build_features() trong pipeline.py.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         DataFrame đặc trưng (1 hàng cuối là điểm dự báo hiện tại).
#     """
#     feat = df.copy()
#
#     for col in EXOG_COLS:
#         feat[f"{col}_s1"] = feat[col].shift(1)
#
#     feat["lag_1"] = feat[TARGET_COL].shift(1)
#     feat["lag_2"] = feat[TARGET_COL].shift(2)
#     feat["lag_3"] = feat[TARGET_COL].shift(3)
#
#     target_shifted = feat[TARGET_COL].shift(1)
#     feat["rolling_mean_7"]  = target_shifted.rolling(7,  min_periods=1).mean()
#     feat["rolling_mean_30"] = target_shifted.rolling(30, min_periods=1).mean()
#     feat["rolling_std_7"]   = target_shifted.rolling(7,  min_periods=2).std()
#     feat["rolling_std_30"]  = target_shifted.rolling(30, min_periods=2).std()
#     feat["rolling_max_7"]   = target_shifted.rolling(7,  min_periods=1).max()
#     feat["rolling_min_7"]   = target_shifted.rolling(7,  min_periods=1).min()
#
#     for col in EXOG_COLS:
#         s = feat[f"{col}_s1"]
#         feat[f"{col}_rm7"]  = s.rolling(7,  min_periods=1).mean()
#         feat[f"{col}_rm30"] = s.rolling(30, min_periods=1).mean()
#
#     feat["day_of_week"] = feat.index.dayofweek
#     feat["month"]       = feat.index.month
#     feat["is_weekend"]  = (feat.index.dayofweek >= 5).astype(int)
#     feat["quarter"]     = feat.index.quarter
#     feat["day_of_year"] = feat.index.dayofyear
#     feat["residual"]    = feat[TARGET_COL] - feat[TARGET_COL].shift(1)
#
#     cols_to_drop = EXOG_COLS + [TARGET_COL, "buy_price"]
#     feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])
#     return feat.dropna()
#
#
# def predict_price(df: pd.DataFrame, horizon: int) -> dict:
#     """
#     Suy luận trên hàng dữ liệu mới nhất.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         Dict {'lgbm_pred', 'rf_pred', 'naive_pred', 'last_price',
#               'last_date', 'model_loaded'}.
#     """
#     arts = load_model_artifacts(horizon)
#     result = {
#         "lgbm_pred":    None,
#         "rf_pred":      None,
#         "naive_pred":   None,
#         "last_price":   df[TARGET_COL].iloc[-1],
#         "last_date":    df.index[-1],
#         "model_loaded": bool(arts),
#     }
#
#     # Naive
#     result["naive_pred"] = df[TARGET_COL].iloc[-(horizon)]
#
#     if not arts:
#         return result
#
#     meta         = arts["metadata"]
#     feature_cols = meta["feature_list"]
#     scaler       = arts["scaler"]
#
#     feat_df = build_features_for_inference(df, horizon)
#     if feat_df.empty:
#         return result
#
#     last_row = feat_df[feature_cols].iloc[[-1]]
#     try:
#         last_scaled = pd.DataFrame(
#             scaler.transform(last_row),
#             columns=feature_cols,
#         )
#     except Exception as exc:
#         logger.error(f"Lỗi scaling: {exc}")
#         return result
#
#     # Residual predictions
#     e_hat_lgb = arts["lgbm"].predict(last_scaled)[0]
#     e_hat_rf  = arts["rf"].predict(last_scaled)[0]
#
#     naive_base           = result["naive_pred"]
#     result["lgbm_pred"]  = naive_base + e_hat_lgb
#     result["rf_pred"]    = naive_base + e_hat_rf
#
#     return result
#
#
# # ══════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════
# def render_sidebar(df: pd.DataFrame) -> None:
#     """Thanh bên với thông tin nhanh về dataset."""
#     with st.sidebar:
#         st.markdown("## 🥇 Giá Vàng SJC")
#         st.markdown("---")
#
#         last_row   = df.iloc[-1]
#         last_date  = df.index[-1].strftime("%d/%m/%Y")
#         sell_price = last_row[TARGET_COL]
#         buy_price  = last_row.get("buy_price", np.nan)
#         xauusd     = last_row.get("xauusd_close", np.nan)
#
#         st.markdown(f"**📅 Ngày cập nhật:** {last_date}")
#         st.markdown(f"**💰 Giá bán SJC:** `{sell_price:.1f}` triệu VND/lượng")
#         st.markdown(f"**💵 XAUUSD:** `{xauusd:.0f}` USD/oz")
#         st.markdown("---")
#         st.markdown(f"**Tổng số ngày:** {len(df):,}")
#         st.markdown(f"**Từ:** {df.index.min().strftime('%d/%m/%Y')}")
#         st.markdown(f"**Đến:** {df.index.max().strftime('%d/%m/%Y')}")
#         st.markdown("---")
#         st.caption("© 2024 Luận văn tốt nghiệp CNTT")
#
#
# # ══════════════════════════════════════════════
# # TAB 1 — TỔNG QUAN
# # ══════════════════════════════════════════════
# def tab_overview(df: pd.DataFrame) -> None:
#     """Tab 1: Giới thiệu dự án và biện pháp chống rò rỉ."""
#     st.markdown('<p class="main-title">🥇 Dự Báo Giá Vàng SJC</p>', unsafe_allow_html=True)
#     st.markdown("#### Hệ thống dự báo đa horizon · Chống rò rỉ dữ liệu · Học phần dư")
#
#     col1, col2, col3, col4 = st.columns(4)
#     metrics = [
#         ("📅", "Số ngày dữ liệu", f"{len(df):,}"),
#         ("🏆", "Mô hình tốt nhất", "LightGBM"),
#         ("⏱️", "Horizon tối đa", "7 ngày"),
#         ("🛡️", "Chiến lược CV", "TimeSeriesSplit"),
#     ]
#     for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics):
#         with col:
#             st.markdown(
#                 f'<div class="metric-card"><div style="font-size:1.5rem">{icon}</div>'
#                 f'<div class="metric-value">{value}</div>'
#                 f'<div class="metric-label">{label}</div></div>',
#                 unsafe_allow_html=True,
#             )
#
#     st.markdown("---")
#
#     col_left, col_right = st.columns([1, 1])
#
#     with col_left:
#         st.markdown("### 🎯 Mục tiêu dự án")
#         st.markdown("""
#         Xây dựng pipeline dự báo **giá vàng SJC** (Công ty Vàng bạc Đá quý Sài Gòn)
#         với các horizon **1 ngày, 3 ngày và 7 ngày** sử dụng:
#
#         - 📈 **Dữ liệu thị trường:** XAUUSD, DXY, VIX, S&P500, VN-Index, USD/VND
#         - 🤖 **Mô hình ML:** Random Forest & LightGBM với **Residual Learning**
#         - 📊 **Tối ưu tự động:** Optuna + TimeSeriesSplit Cross-Validation
#         - 🔍 **Giải thích:** SHAP values
#         """)
#
#         st.markdown("### 📐 Dataset")
#         st.markdown(f"""
#         | Thuộc tính | Giá trị |
#         |---|---|
#         | **Tập huấn luyện** | 01/01/2010 → 31/12/2023 |
#         | **Tập kiểm tra** | 01/01/2024 → hiện tại |
#         | **Target** | `sell_price` (Triệu VND/lượng) |
#         | **Số đặc trưng** | ~30 sau feature engineering |
#         """)
#
#     with col_right:
#         st.markdown("### 🛡️ Biện pháp chống rò rỉ dữ liệu")
#         measures = [
#             ("✅", "Shift tất cả biến ngoại sinh `.shift(1)` trước khi tạo lag/rolling"),
#             ("✅", "Rolling window chỉ tính trên dữ liệu đến t-1 (sau khi shift)"),
#             ("✅", "Scaler **chỉ fit trên tập train**, transform riêng tập test"),
#             ("✅", "TimeSeriesSplit — **không shuffle** dữ liệu"),
#             ("✅", "Multi-horizon target: y[t+h-1] với đặc trưng chỉ từ t-1"),
#             ("✅", "Residual learning: e(t) = y(t) − y(t-1) với X tại t-1"),
#             ("✅", "Không có thông tin tập test nào được dùng khi huấn luyện"),
#         ]
#         for icon, text in measures:
#             st.markdown(
#                 f'<div class="success-box">{icon} {text}</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ tổng quan giá
#     st.markdown("---")
#     st.markdown("### 📈 Lịch sử giá vàng SJC (2010 – nay)")
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df.index, y=df[TARGET_COL],
#         mode="lines", name="Giá bán SJC",
#         line=dict(color="#FFB703", width=1.5),
#         fill="tozeroy", fillcolor="rgba(255,183,3,0.08)",
#     ))
#     # Đường phân cách train/test
#     fig.add_vline(x="2024-01-01", line_dash="dash", line_color="red",
#                   annotation_text="Test set bắt đầu", annotation_position="top left")
#     fig.update_layout(
#         height=350, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title="Triệu VND/lượng",
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 2 — PHÂN TÍCH GIÁ
# # ══════════════════════════════════════════════
# def tab_analysis(df: pd.DataFrame) -> None:
#     """Tab 2: EDA, tương quan, kiểm định ADF."""
#     st.markdown("## 📊 Phân tích giá & thị trường")
#
#     # -- Biểu đồ đa biến --
#     st.markdown("### 📈 So sánh các chỉ số thị trường")
#     selected_cols = st.multiselect(
#         "Chọn biến để so sánh:",
#         options=EXOG_COLS + [TARGET_COL],
#         default=["sell_price", "xauusd_close", "dxy_close"],
#     )
#     if selected_cols:
#         fig = go.Figure()
#         colors = px.colors.qualitative.Plotly
#         for i, col in enumerate(selected_cols):
#             norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
#             fig.add_trace(go.Scatter(
#                 x=df.index, y=norm,
#                 mode="lines", name=col,
#                 line=dict(color=colors[i % len(colors)], width=1.2),
#             ))
#         fig.update_layout(
#             height=350, template="plotly_dark",
#             yaxis_title="Chuẩn hóa Min-Max",
#             xaxis_title="Ngày",
#             margin=dict(l=10, r=10, t=20, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)
#
#     col_left, col_right = st.columns([1, 1])
#
#     # -- Heatmap tương quan --
#     with col_left:
#         st.markdown("### 🔗 Ma trận tương quan")
#         corr_cols = [TARGET_COL, "buy_price"] + EXOG_COLS
#         corr = df[corr_cols].corr()
#
#         fig, ax = plt.subplots(figsize=(8, 6))
#         fig.patch.set_facecolor("#0e1117")
#         ax.set_facecolor("#0e1117")
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
#                     center=0, square=True, linewidths=0.4,
#                     annot_kws={"size": 8, "color": "white"},
#                     ax=ax)
#         ax.set_title("Hệ số tương quan Pearson", color="white", fontsize=11)
#         plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
#         plt.yticks(color="white", fontsize=8)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     # -- ADF Test --
#     with col_right:
#         st.markdown("### 📉 Kiểm định tính dừng (ADF)")
#         adf_rows = []
#         for col in [TARGET_COL] + EXOG_COLS:
#             series = df[col].dropna()
#             stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
#             adf_rows.append({
#                 "Biến":         col,
#                 "ADF Statistic": round(stat, 3),
#                 "p-value":       round(pval, 5),
#                 "Dừng (p<0.05)": "✅ Có" if pval < 0.05 else "❌ Không",
#                 "CV 5%":         round(crit["5%"], 3),
#             })
#         adf_df = pd.DataFrame(adf_rows)
#
#         def color_stationary(val: str) -> str:
#             return "color: #2A9D8F" if "✅" in str(val) else "color: #E63946"
#
#         st.dataframe(
#             adf_df.style.applymap(color_stationary, subset=["Dừng (p<0.05)"]),
#             height=280,
#         )
#         st.markdown(
#             '<div class="info-box">💡 Chuỗi không dừng (p ≥ 0.05) gợi ý nên '
#             'dùng sai phân (differencing). Pipeline sử dụng <b>Residual Learning</b> '
#             'để ổn định hóa target.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # -- Phân phối target --
#     st.markdown("### 📊 Phân phối & biến động giá bán SJC")
#     col_a, col_b = st.columns(2)
#     with col_a:
#         fig = px.histogram(
#             df, x=TARGET_COL, nbins=60,
#             title="Phân phối giá bán SJC",
#             color_discrete_sequence=["#FFB703"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#     with col_b:
#         daily_change = df[TARGET_COL].diff()
#         fig = px.histogram(
#             daily_change.dropna(), x=TARGET_COL, nbins=60,
#             title="Phân phối thay đổi giá ngày hôm nay - hôm qua (Δ)",
#             color_discrete_sequence=["#457B9D"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 3 — DỰ ĐOÁN
# # ══════════════════════════════════════════════
# def tab_prediction(df: pd.DataFrame) -> None:
#     """Tab 3: Giao diện dự báo giá vàng tương tác."""
#     st.markdown("## 🔮 Dự báo giá vàng SJC")
#
#     col_left, col_right = st.columns([1, 2])
#
#     with col_left:
#         st.markdown("### ⚙️ Cài đặt dự báo")
#         horizon_label = st.selectbox(
#             "Chọn horizon dự báo:",
#             options=list(HORIZON_LABELS.values()),
#         )
#         horizon = [k for k, v in HORIZON_LABELS.items() if v == horizon_label][0]
#
#         model_choice = st.radio(
#             "Chọn mô hình:",
#             options=["🤖 LightGBM (Khuyến nghị)", "🌲 Random Forest", "📏 Naive Baseline"],
#         )
#
#         run_predict = st.button("▶️ Chạy dự báo", type="primary", use_container_width=True)
#
#     with col_right:
#         if run_predict:
#             with st.spinner("Đang tính toán dự báo..."):
#                 result = predict_price(df, horizon)
#
#             if not result["model_loaded"]:
#                 st.markdown(
#                     '<div class="warn-box">⚠️ Chưa tìm thấy file model. '
#                     'Vui lòng chạy <code>python pipeline.py</code> trước.</div>',
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 # Chọn giá trị dự báo theo mô hình được chọn
#                 if "LightGBM" in model_choice:
#                     pred_val = result["lgbm_pred"]
#                     model_name = "LightGBM"
#                 elif "Random Forest" in model_choice:
#                     pred_val = result["rf_pred"]
#                     model_name = "Random Forest"
#                 else:
#                     pred_val = result["naive_pred"]
#                     model_name = "Naive Baseline"
#
#                 last_price = result["last_price"]
#                 delta      = (pred_val - last_price) if pred_val else 0
#                 delta_pct  = (delta / last_price * 100) if last_price else 0
#
#                 # Khoảng tin cậy đơn giản: ±2 * std của residuals test
#                 std_resid = df[TARGET_COL].diff().dropna().std()
#                 ci_low    = pred_val - 2 * std_resid * np.sqrt(horizon)
#                 ci_high   = pred_val + 2 * std_resid * np.sqrt(horizon)
#
#                 st.markdown("### 📌 Kết quả dự báo")
#                 c1, c2, c3 = st.columns(3)
#                 with c1:
#                     st.metric(
#                         label=f"Dự báo {horizon_label} ({model_name})",
#                         value=f"{pred_val:.2f} tr.VND",
#                         delta=f"{delta:+.2f} ({delta_pct:+.2f}%)",
#                     )
#                 with c2:
#                     st.metric(label="Giá hiện tại",     value=f"{last_price:.2f} tr.VND")
#                 with c3:
#                     st.metric(label="Ngày dữ liệu mới nhất",
#                               value=result["last_date"].strftime("%d/%m/%Y"))
#
#                 # Khoảng tin cậy
#                 st.markdown(
#                     f'<div class="info-box">📐 Khoảng tin cậy 95% (ước tính): '
#                     f'<b>{ci_low:.2f}</b> – <b>{ci_high:.2f}</b> triệu VND/lượng</div>',
#                     unsafe_allow_html=True,
#                 )
#
#                 # Biểu đồ 90 ngày gần nhất + dự báo
#                 st.markdown("---")
#                 st.markdown("### 📈 Biểu đồ giá 90 ngày gần nhất + dự báo")
#                 recent = df.tail(90)
#                 pred_date = recent.index[-1] + pd.Timedelta(days=horizon)
#
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=recent.index, y=recent[TARGET_COL],
#                     mode="lines", name="Lịch sử giá",
#                     line=dict(color="#FFB703", width=2),
#                 ))
#                 fig.add_trace(go.Scatter(
#                     x=[recent.index[-1], pred_date],
#                     y=[last_price, pred_val],
#                     mode="lines+markers",
#                     name=f"Dự báo H={horizon}",
#                     line=dict(color="#E63946", width=2.5, dash="dot"),
#                     marker=dict(size=[8, 12]),
#                 ))
#                 # CI
#                 fig.add_trace(go.Scatter(
#                     x=[pred_date, pred_date],
#                     y=[ci_low, ci_high],
#                     mode="lines", name="CI 95%",
#                     line=dict(color="#2A9D8F", width=6),
#                     opacity=0.6,
#                 ))
#                 fig.update_layout(
#                     height=350, template="plotly_dark",
#                     xaxis_title="Ngày",
#                     yaxis_title="Giá bán SJC (Triệu VND/lượng)",
#                     margin=dict(l=10, r=10, t=20, b=10),
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("👆 Chọn horizon & mô hình, sau đó nhấn **▶️ Chạy dự báo**")
#
#
# # ══════════════════════════════════════════════
# # TAB 4 — SO SÁNH MÔ HÌNH
# # ══════════════════════════════════════════════
# def tab_comparison(df: pd.DataFrame) -> None:
#     """Tab 4: Metrics đánh giá và biểu đồ so sánh."""
#     st.markdown("## 📋 So sánh mô hình")
#
#     metrics_path = IMG_DIR / "evaluation_metrics.csv"
#
#     if metrics_path.exists():
#         metrics_df = pd.read_csv(metrics_path)
#         st.markdown("### 📊 Bảng kết quả đánh giá (tập kiểm tra)")
#
#         def highlight_best(s: pd.Series) -> list[str]:
#             if s.name in ["MAE", "RMSE", "MAPE"]:
#                 best_idx = s.idxmin()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             elif s.name == "R2":
#                 best_idx = s.idxmax()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             return [""] * len(s)
#
#         st.dataframe(
#             metrics_df.style.apply(highlight_best).format(
#                 {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"}
#             ),
#             use_container_width=True,
#         )
#
#         # Biểu đồ so sánh MAE
#         h1_df = metrics_df[metrics_df["Model"].str.contains("H=1")]
#         if not h1_df.empty:
#             col_a, col_b = st.columns(2)
#             for ax_col, metric in zip([col_a, col_b], ["MAE", "RMSE"]):
#                 with ax_col:
#                     fig = px.bar(
#                         h1_df.sort_values(metric),
#                         x=metric, y="Model",
#                         orientation="h",
#                         title=f"So sánh {metric} (H=1)",
#                         color=metric,
#                         color_continuous_scale="RdYlGn_r",
#                         template="plotly_dark",
#                     )
#                     fig.update_layout(
#                         height=300, coloraxis_showscale=False,
#                         margin=dict(l=10, r=10, t=40, b=10),
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.markdown(
#             '<div class="warn-box">⚠️ Chưa tìm thấy kết quả đánh giá. '
#             'Hãy chạy <code>python pipeline.py</code> để tạo metrics.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # Biểu đồ dự báo đã lưu
#     st.markdown("---")
#     st.markdown("### 📈 Biểu đồ dự báo so với thực tế")
#     for h in HORIZONS:
#         img_path = IMG_DIR / f"predictions_h{h}.png"
#         if img_path.exists():
#             st.markdown(f"**Horizon H={h}:**")
#             st.image(str(img_path), use_column_width=True)
#
#     # SHAP plots
#     st.markdown("---")
#     st.markdown("### 🔍 SHAP Feature Importance (LightGBM)")
#     for h in HORIZONS:
#         shap_path = IMG_DIR / f"shap_summary_h{h}.png"
#         if shap_path.exists():
#             st.markdown(f"**H={h}:**")
#             col_s1, col_s2 = st.columns(2)
#             with col_s1:
#                 st.image(str(IMG_DIR / f"shap_summary_h{h}.png"), use_column_width=True)
#             with col_s2:
#                 bar_path = IMG_DIR / f"shap_bar_h{h}.png"
#                 if bar_path.exists():
#                     st.image(str(bar_path), use_column_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 5 — TIN TỨC & THỊ TRƯỜNG
# # ══════════════════════════════════════════════
# def tab_market(df: pd.DataFrame) -> None:
#     """Tab 5: Thông tin thị trường hiện tại."""
#     st.markdown("## 📡 Tin tức & Thị trường")
#
#     last = df.iloc[-1]
#     last_date = df.index[-1].strftime("%d/%m/%Y")
#     st.markdown(f"*Dữ liệu cập nhật đến: **{last_date}***")
#     st.markdown("---")
#
#     # Chỉ số thị trường
#     st.markdown("### 📊 Chỉ số thị trường hiện tại")
#     market_data = [
#         ("🥇 XAUUSD",    "xauusd_close",  "USD/oz",         "#FFB703"),
#         ("💵 DXY",        "dxy_close",     "Điểm",           "#457B9D"),
#         ("😨 VIX",        "vix_close",     "Điểm",           "#E63946"),
#         ("📈 S&P500",     "sp500_close",   "USD",            "#2A9D8F"),
#         ("🇻🇳 VN-Index",   "vnindex",       "Điểm",           "#F4A261"),
#         ("💱 USD/VND",    "usd_vnd",       "VND",            "#6A4C93"),
#     ]
#     cols = st.columns(3)
#     for i, (label, col, unit, color) in enumerate(market_data):
#         val = last.get(col, np.nan)
#         prev_val = df[col].iloc[-2] if len(df) > 1 else val
#         chg = val - prev_val
#         chg_pct = (chg / prev_val * 100) if prev_val else 0
#
#         with cols[i % 3]:
#             direction = "▲" if chg >= 0 else "▼"
#             chg_color = "#2A9D8F" if chg >= 0 else "#E63946"
#             st.markdown(
#                 f'<div class="metric-card">'
#                 f'<div class="metric-label">{label}</div>'
#                 f'<div class="metric-value" style="color:{color}">'
#                 f'{val:,.1f}</div>'
#                 f'<div class="metric-label">{unit}</div>'
#                 f'<div style="color:{chg_color};font-size:0.9rem">'
#                 f'{direction} {abs(chg):,.2f} ({chg_pct:+.2f}%)</div>'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ 30 ngày
#     st.markdown("---")
#     st.markdown("### 📈 Diễn biến 30 ngày gần nhất")
#     recent30 = df.tail(30)
#     chart_options = {
#         "Giá vàng SJC (Bán)": TARGET_COL,
#         "XAUUSD":              "xauusd_close",
#         "DXY":                 "dxy_close",
#         "VIX":                 "vix_close",
#         "VN-Index":            "vnindex",
#     }
#     selected = st.selectbox("Chọn chỉ số:", list(chart_options.keys()))
#     col_name  = chart_options[selected]
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=recent30.index, y=recent30[col_name],
#         mode="lines+markers",
#         name=selected,
#         line=dict(width=2.5, color="#FFB703"),
#         marker=dict(size=4),
#     ))
#     fig.update_layout(
#         height=300, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title=selected,
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Tương quan gần đây với SJC
#     st.markdown("---")
#     st.markdown("### 🔗 Tương quan 90 ngày gần nhất với Giá bán SJC")
#     recent90   = df.tail(90)
#     corr_vals  = recent90[EXOG_COLS].corrwith(recent90[TARGET_COL]).sort_values(ascending=False)
#     fig = px.bar(
#         corr_vals,
#         orientation="v",
#         title="Tương quan Pearson với sell_price (90 ngày gần nhất)",
#         color=corr_vals.values,
#         color_continuous_scale="RdYlGn",
#         range_color=[-1, 1],
#         template="plotly_dark",
#     )
#     fig.update_layout(
#         height=300,
#         coloraxis_showscale=True,
#         yaxis_title="Hệ số tương quan",
#         margin=dict(l=10, r=10, t=50, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # ĐIỂM NHẬP CHÍNH
# # ══════════════════════════════════════════════
# def main() -> None:
#     """Chạy ứng dụng Streamlit."""
#     # Tải dữ liệu
#     if not DATA_PATH.exists():
#         st.error(f"❌ Không tìm thấy file dữ liệu tại `{DATA_PATH}`. "
#                  "Vui lòng đặt file `gold_dataset_v2.csv` vào thư mục `data/`.")
#         st.stop()
#
#     df = load_data()
#     render_sidebar(df)
#
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "🏠 Tổng quan",
#         "📊 Phân tích giá",
#         "🔮 Dự đoán",
#         "📋 So sánh mô hình",
#         "📡 Tin tức & Thị trường",
#     ])
#
#     with tab1:
#         tab_overview(df)
#     with tab2:
#         tab_analysis(df)
#     with tab3:
#         tab_prediction(df)
#     with tab4:
#         tab_comparison(df)
#     with tab5:
#         tab_market(df)
#
#
# if __name__ == "__main__":
#     main()





# """
# app.py — Ứng dụng Streamlit dự báo giá vàng SJC.
# Tất cả văn bản UI bằng tiếng Việt.
# Tải mô hình đã huấn luyện sẵn từ models/ để suy luận nhanh.
# """

# import json
# import warnings
# import logging
# from pathlib import Path
# from datetime import datetime
#
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
#
# warnings.filterwarnings("ignore")
#
# # Lazy import statsmodels để tránh lỗi scipy trên Python 3.13
# try:
#     from statsmodels.tsa.stattools import adfuller as _adfuller
#     STATSMODELS_OK = True
# except Exception:
#     STATSMODELS_OK = False
#
# def adfuller(series, autolag="AIC"):
#     """Wrapper: dùng statsmodels nếu có, fallback trả về kết quả giả."""
#     if STATSMODELS_OK:
#         return _adfuller(series, autolag=autolag)
#     # Fallback đơn giản: trả về tuple tương thích (stat, pval, lags, nobs, crit, ic)
#     return (0.0, 1.0, 0, len(series), {"1%": -3.43, "5%": -2.86, "10%": -2.57}, 0.0)
#
# # ──────────────────────────────────────────────
# # Cấu hình trang
# # ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="🥇 Dự Báo Giá Vàng SJC",
#     page_icon="🥇",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
#
# # ──────────────────────────────────────────────
# # Logging
# # ──────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # ──────────────────────────────────────────────
# # Hằng số
# # ──────────────────────────────────────────────
# DATA_PATH  = Path("data/gold_dataset_v2.csv")
# MODELS_DIR = Path("models")
# IMG_DIR    = Path("img")
# HORIZONS   = [1, 3, 7]
# TARGET_COL = "sell_price"
# EXOG_COLS  = ["xauusd_close", "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
#
# HORIZON_LABELS = {1: "1 ngày", 3: "3 ngày", 7: "7 ngày"}
#
# # ──────────────────────────────────────────────
# # CSS tùy chỉnh
# # ──────────────────────────────────────────────
# st.markdown("""
# <style>
#     .main-title { font-size: 2.2rem; font-weight: 700; color: #FFB703; }
#     .metric-card {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#         border: 1px solid #FFB703;
#         border-radius: 12px;
#         padding: 1rem;
#         text-align: center;
#         margin: 0.3rem;
#     }
#     .metric-value { font-size: 1.6rem; font-weight: 700; color: #FFB703; }
#     .metric-label { font-size: 0.85rem; color: #aaa; }
#     .info-box {
#         background: #0f3460;
#         border-left: 4px solid #FFB703;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .warn-box {
#         background: #3d1515;
#         border-left: 4px solid #E63946;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .success-box {
#         background: #0d3b2e;
#         border-left: 4px solid #2A9D8F;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # ══════════════════════════════════════════════
# # HÀM TIỆN ÍCH
# # ══════════════════════════════════════════════
#
# @st.cache_data(show_spinner="Đang tải dữ liệu...")
# def load_data() -> pd.DataFrame:
#     """Tải và tiền xử lý dataset."""
#     df = pd.read_csv(DATA_PATH, parse_dates=["date"])
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values("date").set_index("date")
#     full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
#     df = df.reindex(full_range).ffill().bfill()
#     return df
#
#
# @st.cache_resource(show_spinner="Đang tải mô hình...")
# def load_model_artifacts(horizon: int) -> dict:
#     """
#     Tải scaler, RandomForest, LightGBM và metadata cho một horizon.
#
#     Args:
#         horizon: Horizon h (1, 3 hoặc 7).
#
#     Returns:
#         Dict {'scaler', 'rf', 'lgbm', 'metadata'} hoặc {} nếu chưa có model.
#     """
#     prefix = MODELS_DIR / f"h{horizon}"
#     artifacts = {}
#     try:
#         artifacts["scaler"]   = joblib.load(f"{prefix}_scaler.joblib")
#         artifacts["rf"]       = joblib.load(f"{prefix}_rf.joblib")
#         artifacts["lgbm"]     = joblib.load(f"{prefix}_lgbm.joblib")
#         with open(f"{prefix}_metadata.json", "r", encoding="utf-8") as f:
#             artifacts["metadata"] = json.load(f)
#     except FileNotFoundError:
#         logger.warning(f"Chưa tìm thấy model artifacts cho H={horizon}.")
#     return artifacts
#
#
# def build_features_for_inference(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
#     """
#     Xây dựng đặc trưng CHỐNG RÒ RỈ — logic y hệt pipeline.py.
#     Hàm này phải GIỐNG HỆT hàm build_features() trong pipeline.py.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         DataFrame đặc trưng (1 hàng cuối là điểm dự báo hiện tại).
#     """
#     feat = df.copy()
#
#     for col in EXOG_COLS:
#         feat[f"{col}_s1"] = feat[col].shift(1)
#
#     feat["lag_1"] = feat[TARGET_COL].shift(1)
#     feat["lag_2"] = feat[TARGET_COL].shift(2)
#     feat["lag_3"] = feat[TARGET_COL].shift(3)
#
#     target_shifted = feat[TARGET_COL].shift(1)
#     feat["rolling_mean_7"]  = target_shifted.rolling(7,  min_periods=1).mean()
#     feat["rolling_mean_30"] = target_shifted.rolling(30, min_periods=1).mean()
#     feat["rolling_std_7"]   = target_shifted.rolling(7,  min_periods=2).std()
#     feat["rolling_std_30"]  = target_shifted.rolling(30, min_periods=2).std()
#     feat["rolling_max_7"]   = target_shifted.rolling(7,  min_periods=1).max()
#     feat["rolling_min_7"]   = target_shifted.rolling(7,  min_periods=1).min()
#
#     for col in EXOG_COLS:
#         s = feat[f"{col}_s1"]
#         feat[f"{col}_rm7"]  = s.rolling(7,  min_periods=1).mean()
#         feat[f"{col}_rm30"] = s.rolling(30, min_periods=1).mean()
#
#     feat["day_of_week"] = feat.index.dayofweek
#     feat["month"]       = feat.index.month
#     feat["is_weekend"]  = (feat.index.dayofweek >= 5).astype(int)
#     feat["quarter"]     = feat.index.quarter
#     feat["day_of_year"] = feat.index.dayofyear
#     feat["residual"]    = feat[TARGET_COL] - feat[TARGET_COL].shift(1)
#
#     cols_to_drop = EXOG_COLS + [TARGET_COL, "buy_price"]
#     feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])
#     return feat.dropna()
#
#
# def predict_price(df: pd.DataFrame, horizon: int) -> dict:
#     """
#     Suy luận trên hàng dữ liệu mới nhất.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         Dict {'lgbm_pred', 'rf_pred', 'naive_pred', 'last_price',
#               'last_date', 'model_loaded'}.
#     """
#     arts = load_model_artifacts(horizon)
#     result = {
#         "lgbm_pred":    None,
#         "rf_pred":      None,
#         "naive_pred":   None,
#         "last_price":   df[TARGET_COL].iloc[-1],
#         "last_date":    df.index[-1],
#         "model_loaded": bool(arts),
#     }
#
#     # Naive
#     result["naive_pred"] = df[TARGET_COL].iloc[-(horizon)]
#
#     if not arts:
#         return result
#
#     meta         = arts["metadata"]
#     feature_cols = meta["feature_list"]
#     scaler       = arts["scaler"]
#
#     feat_df = build_features_for_inference(df, horizon)
#     if feat_df.empty:
#         return result
#
#     last_row = feat_df[feature_cols].iloc[[-1]]
#     try:
#         last_scaled = pd.DataFrame(
#             scaler.transform(last_row),
#             columns=feature_cols,
#         )
#     except Exception as exc:
#         logger.error(f"Lỗi scaling: {exc}")
#         return result
#
#     # Residual predictions
#     e_hat_lgb = arts["lgbm"].predict(last_scaled)[0]
#     e_hat_rf  = arts["rf"].predict(last_scaled)[0]
#
#     naive_base           = result["naive_pred"]
#     result["lgbm_pred"]  = naive_base + e_hat_lgb
#     result["rf_pred"]    = naive_base + e_hat_rf
#
#     return result
#
#
# # ══════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════
# def render_sidebar(df: pd.DataFrame) -> None:
#     """Thanh bên với thông tin nhanh về dataset."""
#     with st.sidebar:
#         st.markdown("## 🥇 Giá Vàng SJC")
#         st.markdown("---")
#
#         last_row   = df.iloc[-1]
#         last_date  = df.index[-1].strftime("%d/%m/%Y")
#         sell_price = last_row[TARGET_COL]
#         buy_price  = last_row.get("buy_price", np.nan)
#         xauusd     = last_row.get("xauusd_close", np.nan)
#
#         st.markdown(f"**📅 Ngày cập nhật:** {last_date}")
#         st.markdown(f"**💰 Giá bán SJC:** `{sell_price:.1f}` triệu VND/lượng")
#         st.markdown(f"**💵 XAUUSD:** `{xauusd:.0f}` USD/oz")
#         st.markdown("---")
#         st.markdown(f"**Tổng số ngày:** {len(df):,}")
#         st.markdown(f"**Từ:** {df.index.min().strftime('%d/%m/%Y')}")
#         st.markdown(f"**Đến:** {df.index.max().strftime('%d/%m/%Y')}")
#         st.markdown("---")
#         st.caption("© 2024 Luận văn tốt nghiệp CNTT")
#
#
# # ══════════════════════════════════════════════
# # TAB 1 — TỔNG QUAN
# # ══════════════════════════════════════════════
# def tab_overview(df: pd.DataFrame) -> None:
#     """Tab 1: Giới thiệu dự án và biện pháp chống rò rỉ."""
#     st.markdown('<p class="main-title">🥇 Dự Báo Giá Vàng SJC</p>', unsafe_allow_html=True)
#     st.markdown("#### Hệ thống dự báo đa horizon · Chống rò rỉ dữ liệu · Học phần dư")
#
#     col1, col2, col3, col4 = st.columns(4)
#     metrics = [
#         ("📅", "Số ngày dữ liệu", f"{len(df):,}"),
#         ("🏆", "Mô hình tốt nhất", "LightGBM"),
#         ("⏱️", "Horizon tối đa", "7 ngày"),
#         ("🛡️", "Chiến lược CV", "TimeSeriesSplit"),
#     ]
#     for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics):
#         with col:
#             st.markdown(
#                 f'<div class="metric-card"><div style="font-size:1.5rem">{icon}</div>'
#                 f'<div class="metric-value">{value}</div>'
#                 f'<div class="metric-label">{label}</div></div>',
#                 unsafe_allow_html=True,
#             )
#
#     st.markdown("---")
#
#     col_left, col_right = st.columns([1, 1])
#
#     with col_left:
#         st.markdown("### 🎯 Mục tiêu dự án")
#         st.markdown("""
#         Xây dựng pipeline dự báo **giá vàng SJC** (Công ty Vàng bạc Đá quý Sài Gòn)
#         với các horizon **1 ngày, 3 ngày và 7 ngày** sử dụng:
#
#         - 📈 **Dữ liệu thị trường:** XAUUSD, DXY, VIX, S&P500, VN-Index, USD/VND
#         - 🤖 **Mô hình ML:** Random Forest & LightGBM với **Residual Learning**
#         - 📊 **Tối ưu tự động:** Optuna + TimeSeriesSplit Cross-Validation
#         - 🔍 **Giải thích:** SHAP values
#         """)
#
#         st.markdown("### 📐 Dataset")
#         st.markdown(f"""
#         | Thuộc tính | Giá trị |
#         |---|---|
#         | **Tập huấn luyện** | 01/01/2010 → 31/12/2023 |
#         | **Tập kiểm tra** | 01/01/2024 → hiện tại |
#         | **Target** | `sell_price` (Triệu VND/lượng) |
#         | **Số đặc trưng** | ~30 sau feature engineering |
#         """)
#
#     with col_right:
#         st.markdown("### 🛡️ Biện pháp chống rò rỉ dữ liệu")
#         measures = [
#             ("✅", "Shift tất cả biến ngoại sinh `.shift(1)` trước khi tạo lag/rolling"),
#             ("✅", "Rolling window chỉ tính trên dữ liệu đến t-1 (sau khi shift)"),
#             ("✅", "Scaler **chỉ fit trên tập train**, transform riêng tập test"),
#             ("✅", "TimeSeriesSplit — **không shuffle** dữ liệu"),
#             ("✅", "Multi-horizon target: y[t+h-1] với đặc trưng chỉ từ t-1"),
#             ("✅", "Residual learning: e(t) = y(t) − y(t-1) với X tại t-1"),
#             ("✅", "Không có thông tin tập test nào được dùng khi huấn luyện"),
#         ]
#         for icon, text in measures:
#             st.markdown(
#                 f'<div class="success-box">{icon} {text}</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ tổng quan giá
#     st.markdown("---")
#     st.markdown("### 📈 Lịch sử giá vàng SJC (2010 – nay)")
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df.index, y=df[TARGET_COL],
#         mode="lines", name="Giá bán SJC",
#         line=dict(color="#FFB703", width=1.5),
#         fill="tozeroy", fillcolor="rgba(255,183,3,0.08)",
#     ))
#     # Đường phân cách train/test
#     fig.add_vline(x="2024-01-01", line_dash="dash", line_color="red",
#                   annotation_text="Test set bắt đầu", annotation_position="top left")
#     fig.update_layout(
#         height=350, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title="Triệu VND/lượng",
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 2 — PHÂN TÍCH GIÁ
# # ══════════════════════════════════════════════
# def tab_analysis(df: pd.DataFrame) -> None:
#     """Tab 2: EDA, tương quan, kiểm định ADF."""
#     st.markdown("## 📊 Phân tích giá & thị trường")
#
#     # -- Biểu đồ đa biến --
#     st.markdown("### 📈 So sánh các chỉ số thị trường")
#     selected_cols = st.multiselect(
#         "Chọn biến để so sánh:",
#         options=EXOG_COLS + [TARGET_COL],
#         default=["sell_price", "xauusd_close", "dxy_close"],
#     )
#     if selected_cols:
#         fig = go.Figure()
#         colors = px.colors.qualitative.Plotly
#         for i, col in enumerate(selected_cols):
#             norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
#             fig.add_trace(go.Scatter(
#                 x=df.index, y=norm,
#                 mode="lines", name=col,
#                 line=dict(color=colors[i % len(colors)], width=1.2),
#             ))
#         fig.update_layout(
#             height=350, template="plotly_dark",
#             yaxis_title="Chuẩn hóa Min-Max",
#             xaxis_title="Ngày",
#             margin=dict(l=10, r=10, t=20, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)
#
#     col_left, col_right = st.columns([1, 1])
#
#     # -- Heatmap tương quan --
#     with col_left:
#         st.markdown("### 🔗 Ma trận tương quan")
#         corr_cols = [TARGET_COL, "buy_price"] + EXOG_COLS
#         corr = df[corr_cols].corr()
#
#         fig, ax = plt.subplots(figsize=(8, 6))
#         fig.patch.set_facecolor("#0e1117")
#         ax.set_facecolor("#0e1117")
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
#                     center=0, square=True, linewidths=0.4,
#                     annot_kws={"size": 8, "color": "white"},
#                     ax=ax)
#         ax.set_title("Hệ số tương quan Pearson", color="white", fontsize=11)
#         plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
#         plt.yticks(color="white", fontsize=8)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     # -- ADF Test --
#     with col_right:
#         st.markdown("### 📉 Kiểm định tính dừng (ADF)")
#         if not STATSMODELS_OK:
#             st.warning(
#                 "⚠️ **statsmodels** không tương thích với Python 3.13 trên máy này. "
#                 "Kết quả ADF bên dưới chỉ mang tính minh hoạ.\n\n"
#                 "**Cách sửa:** `pip install statsmodels --upgrade` hoặc dùng Python 3.11/3.12."
#             )
#         adf_rows = []
#         for col in [TARGET_COL] + EXOG_COLS:
#             series = df[col].dropna()
#             stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
#             adf_rows.append({
#                 "Biến":         col,
#                 "ADF Statistic": round(stat, 3),
#                 "p-value":       round(pval, 5),
#                 "Dừng (p<0.05)": "✅ Có" if pval < 0.05 else "❌ Không",
#                 "CV 5%":         round(crit["5%"], 3),
#             })
#         adf_df = pd.DataFrame(adf_rows)
#
#         def color_stationary(val: str) -> str:
#             return "color: #2A9D8F" if "✅" in str(val) else "color: #E63946"
#
#         st.dataframe(
#             adf_df.style.applymap(color_stationary, subset=["Dừng (p<0.05)"]),
#             height=280,
#         )
#         st.markdown(
#             '<div class="info-box">💡 Chuỗi không dừng (p ≥ 0.05) gợi ý nên '
#             'dùng sai phân (differencing). Pipeline sử dụng <b>Residual Learning</b> '
#             'để ổn định hóa target.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # -- Phân phối target --
#     st.markdown("### 📊 Phân phối & biến động giá bán SJC")
#     col_a, col_b = st.columns(2)
#     with col_a:
#         fig = px.histogram(
#             df, x=TARGET_COL, nbins=60,
#             title="Phân phối giá bán SJC",
#             color_discrete_sequence=["#FFB703"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#     with col_b:
#         daily_change = df[TARGET_COL].diff()
#         fig = px.histogram(
#             daily_change.dropna(), x=TARGET_COL, nbins=60,
#             title="Phân phối thay đổi giá ngày hôm nay - hôm qua (Δ)",
#             color_discrete_sequence=["#457B9D"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 3 — DỰ ĐOÁN
# # ══════════════════════════════════════════════
# def tab_prediction(df: pd.DataFrame) -> None:
#     """Tab 3: Giao diện dự báo giá vàng tương tác."""
#     st.markdown("## 🔮 Dự báo giá vàng SJC")
#
#     col_left, col_right = st.columns([1, 2])
#
#     with col_left:
#         st.markdown("### ⚙️ Cài đặt dự báo")
#         horizon_label = st.selectbox(
#             "Chọn horizon dự báo:",
#             options=list(HORIZON_LABELS.values()),
#         )
#         horizon = [k for k, v in HORIZON_LABELS.items() if v == horizon_label][0]
#
#         model_choice = st.radio(
#             "Chọn mô hình:",
#             options=["🤖 LightGBM (Khuyến nghị)", "🌲 Random Forest", "📏 Naive Baseline"],
#         )
#
#         run_predict = st.button("▶️ Chạy dự báo", type="primary", use_container_width=True)
#
#     with col_right:
#         if run_predict:
#             with st.spinner("Đang tính toán dự báo..."):
#                 result = predict_price(df, horizon)
#
#             if not result["model_loaded"]:
#                 st.markdown(
#                     '<div class="warn-box">⚠️ Chưa tìm thấy file model. '
#                     'Vui lòng chạy <code>python pipeline.py</code> trước.</div>',
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 # Chọn giá trị dự báo theo mô hình được chọn
#                 if "LightGBM" in model_choice:
#                     pred_val = result["lgbm_pred"]
#                     model_name = "LightGBM"
#                 elif "Random Forest" in model_choice:
#                     pred_val = result["rf_pred"]
#                     model_name = "Random Forest"
#                 else:
#                     pred_val = result["naive_pred"]
#                     model_name = "Naive Baseline"
#
#                 last_price = result["last_price"]
#                 delta      = (pred_val - last_price) if pred_val else 0
#                 delta_pct  = (delta / last_price * 100) if last_price else 0
#
#                 # Khoảng tin cậy đơn giản: ±2 * std của residuals test
#                 std_resid = df[TARGET_COL].diff().dropna().std()
#                 ci_low    = pred_val - 2 * std_resid * np.sqrt(horizon)
#                 ci_high   = pred_val + 2 * std_resid * np.sqrt(horizon)
#
#                 st.markdown("### 📌 Kết quả dự báo")
#                 c1, c2, c3 = st.columns(3)
#                 with c1:
#                     st.metric(
#                         label=f"Dự báo {horizon_label} ({model_name})",
#                         value=f"{pred_val:.2f} tr.VND",
#                         delta=f"{delta:+.2f} ({delta_pct:+.2f}%)",
#                     )
#                 with c2:
#                     st.metric(label="Giá hiện tại",     value=f"{last_price:.2f} tr.VND")
#                 with c3:
#                     st.metric(label="Ngày dữ liệu mới nhất",
#                               value=result["last_date"].strftime("%d/%m/%Y"))
#
#                 # Khoảng tin cậy
#                 st.markdown(
#                     f'<div class="info-box">📐 Khoảng tin cậy 95% (ước tính): '
#                     f'<b>{ci_low:.2f}</b> – <b>{ci_high:.2f}</b> triệu VND/lượng</div>',
#                     unsafe_allow_html=True,
#                 )
#
#                 # Biểu đồ 90 ngày gần nhất + dự báo
#                 st.markdown("---")
#                 st.markdown("### 📈 Biểu đồ giá 90 ngày gần nhất + dự báo")
#                 recent = df.tail(90)
#                 pred_date = recent.index[-1] + pd.Timedelta(days=horizon)
#
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=recent.index, y=recent[TARGET_COL],
#                     mode="lines", name="Lịch sử giá",
#                     line=dict(color="#FFB703", width=2),
#                 ))
#                 fig.add_trace(go.Scatter(
#                     x=[recent.index[-1], pred_date],
#                     y=[last_price, pred_val],
#                     mode="lines+markers",
#                     name=f"Dự báo H={horizon}",
#                     line=dict(color="#E63946", width=2.5, dash="dot"),
#                     marker=dict(size=[8, 12]),
#                 ))
#                 # CI
#                 fig.add_trace(go.Scatter(
#                     x=[pred_date, pred_date],
#                     y=[ci_low, ci_high],
#                     mode="lines", name="CI 95%",
#                     line=dict(color="#2A9D8F", width=6),
#                     opacity=0.6,
#                 ))
#                 fig.update_layout(
#                     height=350, template="plotly_dark",
#                     xaxis_title="Ngày",
#                     yaxis_title="Giá bán SJC (Triệu VND/lượng)",
#                     margin=dict(l=10, r=10, t=20, b=10),
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("👆 Chọn horizon & mô hình, sau đó nhấn **▶️ Chạy dự báo**")
#
#
# # ══════════════════════════════════════════════
# # TAB 4 — SO SÁNH MÔ HÌNH
# # ══════════════════════════════════════════════
# def tab_comparison(df: pd.DataFrame) -> None:
#     """Tab 4: Metrics đánh giá và biểu đồ so sánh."""
#     st.markdown("## 📋 So sánh mô hình")
#
#     metrics_path = IMG_DIR / "evaluation_metrics.csv"
#
#     if metrics_path.exists():
#         metrics_df = pd.read_csv(metrics_path)
#         st.markdown("### 📊 Bảng kết quả đánh giá (tập kiểm tra)")
#
#         def highlight_best(s: pd.Series) -> list[str]:
#             if s.name in ["MAE", "RMSE", "MAPE"]:
#                 best_idx = s.idxmin()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             elif s.name == "R2":
#                 best_idx = s.idxmax()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             return [""] * len(s)
#
#         st.dataframe(
#             metrics_df.style.apply(highlight_best).format(
#                 {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"}
#             ),
#             use_container_width=True,
#         )
#
#         # Biểu đồ so sánh MAE
#         h1_df = metrics_df[metrics_df["Model"].str.contains("H=1")]
#         if not h1_df.empty:
#             col_a, col_b = st.columns(2)
#             for ax_col, metric in zip([col_a, col_b], ["MAE", "RMSE"]):
#                 with ax_col:
#                     fig = px.bar(
#                         h1_df.sort_values(metric),
#                         x=metric, y="Model",
#                         orientation="h",
#                         title=f"So sánh {metric} (H=1)",
#                         color=metric,
#                         color_continuous_scale="RdYlGn_r",
#                         template="plotly_dark",
#                     )
#                     fig.update_layout(
#                         height=300, coloraxis_showscale=False,
#                         margin=dict(l=10, r=10, t=40, b=10),
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.markdown(
#             '<div class="warn-box">⚠️ Chưa tìm thấy kết quả đánh giá. '
#             'Hãy chạy <code>python pipeline.py</code> để tạo metrics.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # Biểu đồ dự báo đã lưu
#     st.markdown("---")
#     st.markdown("### 📈 Biểu đồ dự báo so với thực tế")
#     for h in HORIZONS:
#         img_path = IMG_DIR / f"predictions_h{h}.png"
#         if img_path.exists():
#             st.markdown(f"**Horizon H={h}:**")
#             st.image(str(img_path), use_column_width=True)
#
#     # SHAP plots
#     st.markdown("---")
#     st.markdown("### 🔍 SHAP Feature Importance (LightGBM)")
#     for h in HORIZONS:
#         shap_path = IMG_DIR / f"shap_summary_h{h}.png"
#         if shap_path.exists():
#             st.markdown(f"**H={h}:**")
#             col_s1, col_s2 = st.columns(2)
#             with col_s1:
#                 st.image(str(IMG_DIR / f"shap_summary_h{h}.png"), use_column_width=True)
#             with col_s2:
#                 bar_path = IMG_DIR / f"shap_bar_h{h}.png"
#                 if bar_path.exists():
#                     st.image(str(bar_path), use_column_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 5 — TIN TỨC & THỊ TRƯỜNG
# # ══════════════════════════════════════════════
# def tab_market(df: pd.DataFrame) -> None:
#     """Tab 5: Thông tin thị trường hiện tại."""
#     st.markdown("## 📡 Tin tức & Thị trường")
#
#     last = df.iloc[-1]
#     last_date = df.index[-1].strftime("%d/%m/%Y")
#     st.markdown(f"*Dữ liệu cập nhật đến: **{last_date}***")
#     st.markdown("---")
#
#     # Chỉ số thị trường
#     st.markdown("### 📊 Chỉ số thị trường hiện tại")
#     market_data = [
#         ("🥇 XAUUSD",    "xauusd_close",  "USD/oz",         "#FFB703"),
#         ("💵 DXY",        "dxy_close",     "Điểm",           "#457B9D"),
#         ("😨 VIX",        "vix_close",     "Điểm",           "#E63946"),
#         ("📈 S&P500",     "sp500_close",   "USD",            "#2A9D8F"),
#         ("🇻🇳 VN-Index",   "vnindex",       "Điểm",           "#F4A261"),
#         ("💱 USD/VND",    "usd_vnd",       "VND",            "#6A4C93"),
#     ]
#     cols = st.columns(3)
#     for i, (label, col, unit, color) in enumerate(market_data):
#         val = last.get(col, np.nan)
#         prev_val = df[col].iloc[-2] if len(df) > 1 else val
#         chg = val - prev_val
#         chg_pct = (chg / prev_val * 100) if prev_val else 0
#
#         with cols[i % 3]:
#             direction = "▲" if chg >= 0 else "▼"
#             chg_color = "#2A9D8F" if chg >= 0 else "#E63946"
#             st.markdown(
#                 f'<div class="metric-card">'
#                 f'<div class="metric-label">{label}</div>'
#                 f'<div class="metric-value" style="color:{color}">'
#                 f'{val:,.1f}</div>'
#                 f'<div class="metric-label">{unit}</div>'
#                 f'<div style="color:{chg_color};font-size:0.9rem">'
#                 f'{direction} {abs(chg):,.2f} ({chg_pct:+.2f}%)</div>'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ 30 ngày
#     st.markdown("---")
#     st.markdown("### 📈 Diễn biến 30 ngày gần nhất")
#     recent30 = df.tail(30)
#     chart_options = {
#         "Giá vàng SJC (Bán)": TARGET_COL,
#         "XAUUSD":              "xauusd_close",
#         "DXY":                 "dxy_close",
#         "VIX":                 "vix_close",
#         "VN-Index":            "vnindex",
#     }
#     selected = st.selectbox("Chọn chỉ số:", list(chart_options.keys()))
#     col_name  = chart_options[selected]
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=recent30.index, y=recent30[col_name],
#         mode="lines+markers",
#         name=selected,
#         line=dict(width=2.5, color="#FFB703"),
#         marker=dict(size=4),
#     ))
#     fig.update_layout(
#         height=300, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title=selected,
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Tương quan gần đây với SJC
#     st.markdown("---")
#     st.markdown("### 🔗 Tương quan 90 ngày gần nhất với Giá bán SJC")
#     recent90   = df.tail(90)
#     corr_vals  = recent90[EXOG_COLS].corrwith(recent90[TARGET_COL]).sort_values(ascending=False)
#     fig = px.bar(
#         corr_vals,
#         orientation="v",
#         title="Tương quan Pearson với sell_price (90 ngày gần nhất)",
#         color=corr_vals.values,
#         color_continuous_scale="RdYlGn",
#         range_color=[-1, 1],
#         template="plotly_dark",
#     )
#     fig.update_layout(
#         height=300,
#         coloraxis_showscale=True,
#         yaxis_title="Hệ số tương quan",
#         margin=dict(l=10, r=10, t=50, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # ĐIỂM NHẬP CHÍNH
# # ══════════════════════════════════════════════
# def main() -> None:
#     """Chạy ứng dụng Streamlit."""
#     # Tải dữ liệu
#     if not DATA_PATH.exists():
#         st.error(f"❌ Không tìm thấy file dữ liệu tại `{DATA_PATH}`. "
#                  "Vui lòng đặt file `gold_dataset_v2.csv` vào thư mục `data/`.")
#         st.stop()
#
#     df = load_data()
#     render_sidebar(df)
#
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "🏠 Tổng quan",
#         "📊 Phân tích giá",
#         "🔮 Dự đoán",
#         "📋 So sánh mô hình",
#         "📡 Tin tức & Thị trường",
#     ])
#
#     with tab1:
#         tab_overview(df)
#     with tab2:
#         tab_analysis(df)
#     with tab3:
#         tab_prediction(df)
#     with tab4:
#         tab_comparison(df)
#     with tab5:
#         tab_market(df)
#
#
# if __name__ == "__main__":
#     main()






#
# """
# app.py — Ứng dụng Streamlit dự báo giá vàng SJC.
# Tất cả văn bản UI bằng tiếng Việt.
# Tải mô hình đã huấn luyện sẵn từ models/ để suy luận nhanh.
# """

# import json
# import warnings
# import logging
# from pathlib import Path
# from datetime import datetime
#
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
#
# warnings.filterwarnings("ignore")
#
# # Lazy import statsmodels để tránh lỗi scipy trên Python 3.13
# try:
#     from statsmodels.tsa.stattools import adfuller as _adfuller
#     STATSMODELS_OK = True
# except Exception:
#     STATSMODELS_OK = False
#
# def adfuller(series, autolag="AIC"):
#     """Wrapper: dùng statsmodels nếu có, fallback trả về kết quả giả."""
#     if STATSMODELS_OK:
#         return _adfuller(series, autolag=autolag)
#     # Fallback đơn giản: trả về tuple tương thích (stat, pval, lags, nobs, crit, ic)
#     return (0.0, 1.0, 0, len(series), {"1%": -3.43, "5%": -2.86, "10%": -2.57}, 0.0)
#
# # ──────────────────────────────────────────────
# # Cấu hình trang
# # ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="🥇 Dự Báo Giá Vàng SJC",
#     page_icon="🥇",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
#
# # ──────────────────────────────────────────────
# # Logging
# # ──────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # ──────────────────────────────────────────────
# # Hằng số
# # ──────────────────────────────────────────────
# DATA_PATH  = Path("data/gold_dataset_v2.csv")
# MODELS_DIR = Path("models")
# IMG_DIR    = Path("img")
# HORIZONS   = [1, 3, 7]
# TARGET_COL = "sell_price"
# EXOG_COLS  = ["xauusd_close", "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
#
# HORIZON_LABELS = {1: "1 ngày", 3: "3 ngày", 7: "7 ngày"}
#
# # ──────────────────────────────────────────────
# # CSS tùy chỉnh
# # ──────────────────────────────────────────────
# st.markdown("""
# <style>
#     .main-title { font-size: 2.2rem; font-weight: 700; color: #FFB703; }
#     .metric-card {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#         border: 1px solid #FFB703;
#         border-radius: 12px;
#         padding: 1rem;
#         text-align: center;
#         margin: 0.3rem;
#     }
#     .metric-value { font-size: 1.6rem; font-weight: 700; color: #FFB703; }
#     .metric-label { font-size: 0.85rem; color: #aaa; }
#     .info-box {
#         background: #0f3460;
#         border-left: 4px solid #FFB703;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .warn-box {
#         background: #3d1515;
#         border-left: 4px solid #E63946;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .success-box {
#         background: #0d3b2e;
#         border-left: 4px solid #2A9D8F;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # ══════════════════════════════════════════════
# # HÀM TIỆN ÍCH
# # ══════════════════════════════════════════════
#
# @st.cache_data(show_spinner="Đang tải dữ liệu...")
# def load_data() -> pd.DataFrame:
#     """Tải và tiền xử lý dataset."""
#     df = pd.read_csv(DATA_PATH, parse_dates=["date"])
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values("date").set_index("date")
#     full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
#     df = df.reindex(full_range).ffill().bfill()
#     return df
#
#
# @st.cache_resource(show_spinner="Đang tải mô hình...")
# def load_model_artifacts(horizon: int) -> dict:
#     """
#     Tải scaler, RandomForest, LightGBM và metadata cho một horizon.
#
#     Args:
#         horizon: Horizon h (1, 3 hoặc 7).
#
#     Returns:
#         Dict {'scaler', 'rf', 'lgbm', 'metadata'} hoặc {} nếu chưa có model.
#     """
#     prefix = MODELS_DIR / f"h{horizon}"
#     artifacts = {}
#     try:
#         artifacts["scaler"]   = joblib.load(f"{prefix}_scaler.joblib")
#         artifacts["rf"]       = joblib.load(f"{prefix}_rf.joblib")
#         artifacts["lgbm"]     = joblib.load(f"{prefix}_lgbm.joblib")
#         with open(f"{prefix}_metadata.json", "r", encoding="utf-8") as f:
#             artifacts["metadata"] = json.load(f)
#     except FileNotFoundError:
#         logger.warning(f"Chưa tìm thấy model artifacts cho H={horizon}.")
#     return artifacts
#
#
# def build_features_for_inference(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
#     """
#     Xây dựng đặc trưng CHỐNG RÒ RỈ — logic y hệt pipeline.py.
#     Hàm này phải GIỐNG HỆT hàm build_features() trong pipeline.py.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         DataFrame đặc trưng (1 hàng cuối là điểm dự báo hiện tại).
#     """
#     feat = df.copy()
#
#     for col in EXOG_COLS:
#         feat[f"{col}_s1"] = feat[col].shift(1)
#
#     feat["lag_1"] = feat[TARGET_COL].shift(1)
#     feat["lag_2"] = feat[TARGET_COL].shift(2)
#     feat["lag_3"] = feat[TARGET_COL].shift(3)
#
#     target_shifted = feat[TARGET_COL].shift(1)
#     feat["rolling_mean_7"]  = target_shifted.rolling(7,  min_periods=1).mean()
#     feat["rolling_mean_30"] = target_shifted.rolling(30, min_periods=1).mean()
#     feat["rolling_std_7"]   = target_shifted.rolling(7,  min_periods=2).std()
#     feat["rolling_std_30"]  = target_shifted.rolling(30, min_periods=2).std()
#     feat["rolling_max_7"]   = target_shifted.rolling(7,  min_periods=1).max()
#     feat["rolling_min_7"]   = target_shifted.rolling(7,  min_periods=1).min()
#
#     for col in EXOG_COLS:
#         s = feat[f"{col}_s1"]
#         feat[f"{col}_rm7"]  = s.rolling(7,  min_periods=1).mean()
#         feat[f"{col}_rm30"] = s.rolling(30, min_periods=1).mean()
#
#     feat["day_of_week"] = feat.index.dayofweek
#     feat["month"]       = feat.index.month
#     feat["is_weekend"]  = (feat.index.dayofweek >= 5).astype(int)
#     feat["quarter"]     = feat.index.quarter
#     feat["day_of_year"] = feat.index.dayofyear
#     feat["residual"]    = feat[TARGET_COL] - feat[TARGET_COL].shift(1)
#
#     cols_to_drop = EXOG_COLS + [TARGET_COL, "buy_price"]
#     feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])
#     return feat.dropna()
#
#
# def predict_price(df: pd.DataFrame, horizon: int) -> dict:
#     """
#     Suy luận trên hàng dữ liệu mới nhất.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         Dict {'lgbm_pred', 'rf_pred', 'naive_pred', 'last_price',
#               'last_date', 'model_loaded'}.
#     """
#     arts = load_model_artifacts(horizon)
#     result = {
#         "lgbm_pred":    None,
#         "rf_pred":      None,
#         "naive_pred":   None,
#         "last_price":   df[TARGET_COL].iloc[-1],
#         "last_date":    df.index[-1],
#         "model_loaded": bool(arts),
#     }
#
#     # Naive
#     result["naive_pred"] = df[TARGET_COL].iloc[-(horizon)]
#
#     if not arts:
#         return result
#
#     meta         = arts["metadata"]
#     feature_cols = meta["feature_list"]
#     scaler       = arts["scaler"]
#
#     feat_df = build_features_for_inference(df, horizon)
#     if feat_df.empty:
#         return result
#
#     last_row = feat_df[feature_cols].iloc[[-1]]
#     try:
#         last_scaled = pd.DataFrame(
#             scaler.transform(last_row),
#             columns=feature_cols,
#         )
#     except Exception as exc:
#         logger.error(f"Lỗi scaling: {exc}")
#         return result
#
#     # Residual predictions
#     e_hat_lgb = arts["lgbm"].predict(last_scaled)[0]
#     e_hat_rf  = arts["rf"].predict(last_scaled)[0]
#
#     naive_base           = result["naive_pred"]
#     result["lgbm_pred"]  = naive_base + e_hat_lgb
#     result["rf_pred"]    = naive_base + e_hat_rf
#
#     return result
#
#
# # ══════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════
# def render_sidebar(df: pd.DataFrame) -> None:
#     """Thanh bên với thông tin nhanh về dataset."""
#     with st.sidebar:
#         st.markdown("## 🥇 Giá Vàng SJC")
#         st.markdown("---")
#
#         last_row   = df.iloc[-1]
#         last_date  = df.index[-1].strftime("%d/%m/%Y")
#         sell_price = last_row[TARGET_COL]
#         buy_price  = last_row.get("buy_price", np.nan)
#         xauusd     = last_row.get("xauusd_close", np.nan)
#
#         st.markdown(f"**📅 Ngày cập nhật:** {last_date}")
#         st.markdown(f"**💰 Giá bán SJC:** `{sell_price:.1f}` triệu VND/lượng")
#         st.markdown(f"**💵 XAUUSD:** `{xauusd:.0f}` USD/oz")
#         st.markdown("---")
#         st.markdown(f"**Tổng số ngày:** {len(df):,}")
#         st.markdown(f"**Từ:** {df.index.min().strftime('%d/%m/%Y')}")
#         st.markdown(f"**Đến:** {df.index.max().strftime('%d/%m/%Y')}")
#         st.markdown("---")
#         st.caption("© 2024 Luận văn tốt nghiệp CNTT")
#
#
# # ══════════════════════════════════════════════
# # TAB 1 — TỔNG QUAN
# # ══════════════════════════════════════════════
# def tab_overview(df: pd.DataFrame) -> None:
#     """Tab 1: Giới thiệu dự án và biện pháp chống rò rỉ."""
#     st.markdown('<p class="main-title">🥇 Dự Báo Giá Vàng SJC</p>', unsafe_allow_html=True)
#     st.markdown("#### Hệ thống dự báo đa horizon · Chống rò rỉ dữ liệu · Học phần dư")
#
#     col1, col2, col3, col4 = st.columns(4)
#     metrics = [
#         ("📅", "Số ngày dữ liệu", f"{len(df):,}"),
#         ("🏆", "Mô hình tốt nhất", "LightGBM"),
#         ("⏱️", "Horizon tối đa", "7 ngày"),
#         ("🛡️", "Chiến lược CV", "TimeSeriesSplit"),
#     ]
#     for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics):
#         with col:
#             st.markdown(
#                 f'<div class="metric-card"><div style="font-size:1.5rem">{icon}</div>'
#                 f'<div class="metric-value">{value}</div>'
#                 f'<div class="metric-label">{label}</div></div>',
#                 unsafe_allow_html=True,
#             )
#
#     st.markdown("---")
#
#     col_left, col_right = st.columns([1, 1])
#
#     with col_left:
#         st.markdown("### 🎯 Mục tiêu dự án")
#         st.markdown("""
#         Xây dựng pipeline dự báo **giá vàng SJC** (Công ty Vàng bạc Đá quý Sài Gòn)
#         với các horizon **1 ngày, 3 ngày và 7 ngày** sử dụng:
#
#         - 📈 **Dữ liệu thị trường:** XAUUSD, DXY, VIX, S&P500, VN-Index, USD/VND
#         - 🤖 **Mô hình ML:** Random Forest & LightGBM với **Residual Learning**
#         - 📊 **Tối ưu tự động:** Optuna + TimeSeriesSplit Cross-Validation
#         - 🔍 **Giải thích:** SHAP values
#         """)
#
#         st.markdown("### 📐 Dataset")
#         st.markdown(f"""
#         | Thuộc tính | Giá trị |
#         |---|---|
#         | **Tập huấn luyện** | 01/01/2010 → 31/12/2023 |
#         | **Tập kiểm tra** | 01/01/2024 → hiện tại |
#         | **Target** | `sell_price` (Triệu VND/lượng) |
#         | **Số đặc trưng** | ~30 sau feature engineering |
#         """)
#
#     with col_right:
#         st.markdown("### 🛡️ Biện pháp chống rò rỉ dữ liệu")
#         measures = [
#             ("✅", "Shift tất cả biến ngoại sinh `.shift(1)` trước khi tạo lag/rolling"),
#             ("✅", "Rolling window chỉ tính trên dữ liệu đến t-1 (sau khi shift)"),
#             ("✅", "Scaler **chỉ fit trên tập train**, transform riêng tập test"),
#             ("✅", "TimeSeriesSplit — **không shuffle** dữ liệu"),
#             ("✅", "Multi-horizon target: y[t+h-1] với đặc trưng chỉ từ t-1"),
#             ("✅", "Residual learning: e(t) = y(t) − y(t-1) với X tại t-1"),
#             ("✅", "Không có thông tin tập test nào được dùng khi huấn luyện"),
#         ]
#         for icon, text in measures:
#             st.markdown(
#                 f'<div class="success-box">{icon} {text}</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ tổng quan giá
#     st.markdown("---")
#     st.markdown("### 📈 Lịch sử giá vàng SJC (2010 – nay)")
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df.index, y=df[TARGET_COL],
#         mode="lines", name="Giá bán SJC",
#         line=dict(color="#FFB703", width=1.5),
#         fill="tozeroy", fillcolor="rgba(255,183,3,0.08)",
#     ))
#     # Đường phân cách train/test (dùng add_shape thay vì add_vline để tránh lỗi plotly)
#     fig.add_shape(
#         type="line",
#         x0=pd.Timestamp("2024-01-01"), x1=pd.Timestamp("2024-01-01"),
#         y0=0, y1=1, yref="paper",
#         line=dict(color="red", dash="dash", width=1.5),
#     )
#     fig.add_annotation(
#         x=pd.Timestamp("2024-01-01"), y=0.97, yref="paper",
#         text="Test set bắt đầu", showarrow=False,
#         font=dict(color="red", size=11),
#         xanchor="left", bgcolor="rgba(0,0,0,0.5)",
#     )
#     fig.update_layout(
#         height=350, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title="Triệu VND/lượng",
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 2 — PHÂN TÍCH GIÁ
# # ══════════════════════════════════════════════
# def tab_analysis(df: pd.DataFrame) -> None:
#     """Tab 2: EDA, tương quan, kiểm định ADF."""
#     st.markdown("## 📊 Phân tích giá & thị trường")
#
#     # -- Biểu đồ đa biến --
#     st.markdown("### 📈 So sánh các chỉ số thị trường")
#     selected_cols = st.multiselect(
#         "Chọn biến để so sánh:",
#         options=EXOG_COLS + [TARGET_COL],
#         default=["sell_price", "xauusd_close", "dxy_close"],
#     )
#     if selected_cols:
#         fig = go.Figure()
#         colors = px.colors.qualitative.Plotly
#         for i, col in enumerate(selected_cols):
#             norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
#             fig.add_trace(go.Scatter(
#                 x=df.index, y=norm,
#                 mode="lines", name=col,
#                 line=dict(color=colors[i % len(colors)], width=1.2),
#             ))
#         fig.update_layout(
#             height=350, template="plotly_dark",
#             yaxis_title="Chuẩn hóa Min-Max",
#             xaxis_title="Ngày",
#             margin=dict(l=10, r=10, t=20, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)
#
#     col_left, col_right = st.columns([1, 1])
#
#     # -- Heatmap tương quan --
#     with col_left:
#         st.markdown("### 🔗 Ma trận tương quan")
#         corr_cols = [TARGET_COL, "buy_price"] + EXOG_COLS
#         corr = df[corr_cols].corr()
#
#         fig, ax = plt.subplots(figsize=(8, 6))
#         fig.patch.set_facecolor("#0e1117")
#         ax.set_facecolor("#0e1117")
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
#                     center=0, square=True, linewidths=0.4,
#                     annot_kws={"size": 8, "color": "white"},
#                     ax=ax)
#         ax.set_title("Hệ số tương quan Pearson", color="white", fontsize=11)
#         plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
#         plt.yticks(color="white", fontsize=8)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     # -- ADF Test --
#     with col_right:
#         st.markdown("### 📉 Kiểm định tính dừng (ADF)")
#         if not STATSMODELS_OK:
#             st.warning(
#                 "⚠️ **statsmodels** không tương thích với Python 3.13 trên máy này. "
#                 "Kết quả ADF bên dưới chỉ mang tính minh hoạ.\n\n"
#                 "**Cách sửa:** `pip install statsmodels --upgrade` hoặc dùng Python 3.11/3.12."
#             )
#         adf_rows = []
#         for col in [TARGET_COL] + EXOG_COLS:
#             series = df[col].dropna()
#             stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
#             adf_rows.append({
#                 "Biến":         col,
#                 "ADF Statistic": round(stat, 3),
#                 "p-value":       round(pval, 5),
#                 "Dừng (p<0.05)": "✅ Có" if pval < 0.05 else "❌ Không",
#                 "CV 5%":         round(crit["5%"], 3),
#             })
#         adf_df = pd.DataFrame(adf_rows)
#
#         def color_stationary(val: str) -> str:
#             return "color: #2A9D8F" if "✅" in str(val) else "color: #E63946"
#
#         st.dataframe(
#             adf_df.style.applymap(color_stationary, subset=["Dừng (p<0.05)"]),
#             height=280,
#         )
#         st.markdown(
#             '<div class="info-box">💡 Chuỗi không dừng (p ≥ 0.05) gợi ý nên '
#             'dùng sai phân (differencing). Pipeline sử dụng <b>Residual Learning</b> '
#             'để ổn định hóa target.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # -- Phân phối target --
#     st.markdown("### 📊 Phân phối & biến động giá bán SJC")
#     col_a, col_b = st.columns(2)
#     with col_a:
#         fig = px.histogram(
#             df, x=TARGET_COL, nbins=60,
#             title="Phân phối giá bán SJC",
#             color_discrete_sequence=["#FFB703"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#     with col_b:
#         daily_change = df[TARGET_COL].diff()
#         fig = px.histogram(
#             daily_change.dropna(), x=TARGET_COL, nbins=60,
#             title="Phân phối thay đổi giá ngày hôm nay - hôm qua (Δ)",
#             color_discrete_sequence=["#457B9D"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 3 — DỰ ĐOÁN
# # ══════════════════════════════════════════════
# def tab_prediction(df: pd.DataFrame) -> None:
#     """Tab 3: Giao diện dự báo giá vàng tương tác."""
#     st.markdown("## 🔮 Dự báo giá vàng SJC")
#
#     col_left, col_right = st.columns([1, 2])
#
#     with col_left:
#         st.markdown("### ⚙️ Cài đặt dự báo")
#         horizon_label = st.selectbox(
#             "Chọn horizon dự báo:",
#             options=list(HORIZON_LABELS.values()),
#         )
#         horizon = [k for k, v in HORIZON_LABELS.items() if v == horizon_label][0]
#
#         model_choice = st.radio(
#             "Chọn mô hình:",
#             options=["🤖 LightGBM (Khuyến nghị)", "🌲 Random Forest", "📏 Naive Baseline"],
#         )
#
#         run_predict = st.button("▶️ Chạy dự báo", type="primary", use_container_width=True)
#
#     with col_right:
#         if run_predict:
#             with st.spinner("Đang tính toán dự báo..."):
#                 result = predict_price(df, horizon)
#
#             if not result["model_loaded"]:
#                 st.markdown(
#                     '<div class="warn-box">⚠️ Chưa tìm thấy file model. '
#                     'Vui lòng chạy <code>python pipeline.py</code> trước.</div>',
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 # Chọn giá trị dự báo theo mô hình được chọn
#                 if "LightGBM" in model_choice:
#                     pred_val = result["lgbm_pred"]
#                     model_name = "LightGBM"
#                 elif "Random Forest" in model_choice:
#                     pred_val = result["rf_pred"]
#                     model_name = "Random Forest"
#                 else:
#                     pred_val = result["naive_pred"]
#                     model_name = "Naive Baseline"
#
#                 last_price = result["last_price"]
#                 delta      = (pred_val - last_price) if pred_val else 0
#                 delta_pct  = (delta / last_price * 100) if last_price else 0
#
#                 # Khoảng tin cậy đơn giản: ±2 * std của residuals test
#                 std_resid = df[TARGET_COL].diff().dropna().std()
#                 ci_low    = pred_val - 2 * std_resid * np.sqrt(horizon)
#                 ci_high   = pred_val + 2 * std_resid * np.sqrt(horizon)
#
#                 st.markdown("### 📌 Kết quả dự báo")
#                 c1, c2, c3 = st.columns(3)
#                 with c1:
#                     st.metric(
#                         label=f"Dự báo {horizon_label} ({model_name})",
#                         value=f"{pred_val:.2f} tr.VND",
#                         delta=f"{delta:+.2f} ({delta_pct:+.2f}%)",
#                     )
#                 with c2:
#                     st.metric(label="Giá hiện tại",     value=f"{last_price:.2f} tr.VND")
#                 with c3:
#                     st.metric(label="Ngày dữ liệu mới nhất",
#                               value=result["last_date"].strftime("%d/%m/%Y"))
#
#                 # Khoảng tin cậy
#                 st.markdown(
#                     f'<div class="info-box">📐 Khoảng tin cậy 95% (ước tính): '
#                     f'<b>{ci_low:.2f}</b> – <b>{ci_high:.2f}</b> triệu VND/lượng</div>',
#                     unsafe_allow_html=True,
#                 )
#
#                 # Biểu đồ 90 ngày gần nhất + dự báo
#                 st.markdown("---")
#                 st.markdown("### 📈 Biểu đồ giá 90 ngày gần nhất + dự báo")
#                 recent = df.tail(90)
#                 pred_date = recent.index[-1] + pd.Timedelta(days=horizon)
#
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=recent.index, y=recent[TARGET_COL],
#                     mode="lines", name="Lịch sử giá",
#                     line=dict(color="#FFB703", width=2),
#                 ))
#                 fig.add_trace(go.Scatter(
#                     x=[recent.index[-1], pred_date],
#                     y=[last_price, pred_val],
#                     mode="lines+markers",
#                     name=f"Dự báo H={horizon}",
#                     line=dict(color="#E63946", width=2.5, dash="dot"),
#                     marker=dict(size=[8, 12]),
#                 ))
#                 # CI
#                 fig.add_trace(go.Scatter(
#                     x=[pred_date, pred_date],
#                     y=[ci_low, ci_high],
#                     mode="lines", name="CI 95%",
#                     line=dict(color="#2A9D8F", width=6),
#                     opacity=0.6,
#                 ))
#                 fig.update_layout(
#                     height=350, template="plotly_dark",
#                     xaxis_title="Ngày",
#                     yaxis_title="Giá bán SJC (Triệu VND/lượng)",
#                     margin=dict(l=10, r=10, t=20, b=10),
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("👆 Chọn horizon & mô hình, sau đó nhấn **▶️ Chạy dự báo**")
#
#
# # ══════════════════════════════════════════════
# # TAB 4 — SO SÁNH MÔ HÌNH
# # ══════════════════════════════════════════════
# def tab_comparison(df: pd.DataFrame) -> None:
#     """Tab 4: Metrics đánh giá và biểu đồ so sánh."""
#     st.markdown("## 📋 So sánh mô hình")
#
#     metrics_path = IMG_DIR / "evaluation_metrics.csv"
#
#     if metrics_path.exists():
#         metrics_df = pd.read_csv(metrics_path)
#         st.markdown("### 📊 Bảng kết quả đánh giá (tập kiểm tra)")
#
#         def highlight_best(s: pd.Series) -> list[str]:
#             if s.name in ["MAE", "RMSE", "MAPE"]:
#                 best_idx = s.idxmin()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             elif s.name == "R2":
#                 best_idx = s.idxmax()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             return [""] * len(s)
#
#         st.dataframe(
#             metrics_df.style.apply(highlight_best).format(
#                 {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"}
#             ),
#             use_container_width=True,
#         )
#
#         # Biểu đồ so sánh MAE
#         h1_df = metrics_df[metrics_df["Model"].str.contains("H=1")]
#         if not h1_df.empty:
#             col_a, col_b = st.columns(2)
#             for ax_col, metric in zip([col_a, col_b], ["MAE", "RMSE"]):
#                 with ax_col:
#                     fig = px.bar(
#                         h1_df.sort_values(metric),
#                         x=metric, y="Model",
#                         orientation="h",
#                         title=f"So sánh {metric} (H=1)",
#                         color=metric,
#                         color_continuous_scale="RdYlGn_r",
#                         template="plotly_dark",
#                     )
#                     fig.update_layout(
#                         height=300, coloraxis_showscale=False,
#                         margin=dict(l=10, r=10, t=40, b=10),
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.markdown(
#             '<div class="warn-box">⚠️ Chưa tìm thấy kết quả đánh giá. '
#             'Hãy chạy <code>python pipeline.py</code> để tạo metrics.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # Biểu đồ dự báo đã lưu
#     st.markdown("---")
#     st.markdown("### 📈 Biểu đồ dự báo so với thực tế")
#     for h in HORIZONS:
#         img_path = IMG_DIR / f"predictions_h{h}.png"
#         if img_path.exists():
#             st.markdown(f"**Horizon H={h}:**")
#             st.image(str(img_path), use_column_width=True)
#
#     # SHAP plots
#     st.markdown("---")
#     st.markdown("### 🔍 SHAP Feature Importance (LightGBM)")
#     for h in HORIZONS:
#         shap_path = IMG_DIR / f"shap_summary_h{h}.png"
#         if shap_path.exists():
#             st.markdown(f"**H={h}:**")
#             col_s1, col_s2 = st.columns(2)
#             with col_s1:
#                 st.image(str(IMG_DIR / f"shap_summary_h{h}.png"), use_column_width=True)
#             with col_s2:
#                 bar_path = IMG_DIR / f"shap_bar_h{h}.png"
#                 if bar_path.exists():
#                     st.image(str(bar_path), use_column_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 5 — TIN TỨC & THỊ TRƯỜNG
# # ══════════════════════════════════════════════
# def tab_market(df: pd.DataFrame) -> None:
#     """Tab 5: Thông tin thị trường hiện tại."""
#     st.markdown("## 📡 Tin tức & Thị trường")
#
#     last = df.iloc[-1]
#     last_date = df.index[-1].strftime("%d/%m/%Y")
#     st.markdown(f"*Dữ liệu cập nhật đến: **{last_date}***")
#     st.markdown("---")
#
#     # Chỉ số thị trường
#     st.markdown("### 📊 Chỉ số thị trường hiện tại")
#     market_data = [
#         ("🥇 XAUUSD",    "xauusd_close",  "USD/oz",         "#FFB703"),
#         ("💵 DXY",        "dxy_close",     "Điểm",           "#457B9D"),
#         ("😨 VIX",        "vix_close",     "Điểm",           "#E63946"),
#         ("📈 S&P500",     "sp500_close",   "USD",            "#2A9D8F"),
#         ("🇻🇳 VN-Index",   "vnindex",       "Điểm",           "#F4A261"),
#         ("💱 USD/VND",    "usd_vnd",       "VND",            "#6A4C93"),
#     ]
#     cols = st.columns(3)
#     for i, (label, col, unit, color) in enumerate(market_data):
#         val = last.get(col, np.nan)
#         prev_val = df[col].iloc[-2] if len(df) > 1 else val
#         chg = val - prev_val
#         chg_pct = (chg / prev_val * 100) if prev_val else 0
#
#         with cols[i % 3]:
#             direction = "▲" if chg >= 0 else "▼"
#             chg_color = "#2A9D8F" if chg >= 0 else "#E63946"
#             st.markdown(
#                 f'<div class="metric-card">'
#                 f'<div class="metric-label">{label}</div>'
#                 f'<div class="metric-value" style="color:{color}">'
#                 f'{val:,.1f}</div>'
#                 f'<div class="metric-label">{unit}</div>'
#                 f'<div style="color:{chg_color};font-size:0.9rem">'
#                 f'{direction} {abs(chg):,.2f} ({chg_pct:+.2f}%)</div>'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ 30 ngày
#     st.markdown("---")
#     st.markdown("### 📈 Diễn biến 30 ngày gần nhất")
#     recent30 = df.tail(30)
#     chart_options = {
#         "Giá vàng SJC (Bán)": TARGET_COL,
#         "XAUUSD":              "xauusd_close",
#         "DXY":                 "dxy_close",
#         "VIX":                 "vix_close",
#         "VN-Index":            "vnindex",
#     }
#     selected = st.selectbox("Chọn chỉ số:", list(chart_options.keys()))
#     col_name  = chart_options[selected]
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=recent30.index, y=recent30[col_name],
#         mode="lines+markers",
#         name=selected,
#         line=dict(width=2.5, color="#FFB703"),
#         marker=dict(size=4),
#     ))
#     fig.update_layout(
#         height=300, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title=selected,
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Tương quan gần đây với SJC
#     st.markdown("---")
#     st.markdown("### 🔗 Tương quan 90 ngày gần nhất với Giá bán SJC")
#     recent90   = df.tail(90)
#     corr_vals  = recent90[EXOG_COLS].corrwith(recent90[TARGET_COL]).sort_values(ascending=False)
#     fig = px.bar(
#         corr_vals,
#         orientation="v",
#         title="Tương quan Pearson với sell_price (90 ngày gần nhất)",
#         color=corr_vals.values,
#         color_continuous_scale="RdYlGn",
#         range_color=[-1, 1],
#         template="plotly_dark",
#     )
#     fig.update_layout(
#         height=300,
#         coloraxis_showscale=True,
#         yaxis_title="Hệ số tương quan",
#         margin=dict(l=10, r=10, t=50, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # ĐIỂM NHẬP CHÍNH
# # ══════════════════════════════════════════════
# def main() -> None:
#     """Chạy ứng dụng Streamlit."""
#     # Tải dữ liệu
#     if not DATA_PATH.exists():
#         st.error(f"❌ Không tìm thấy file dữ liệu tại `{DATA_PATH}`. "
#                  "Vui lòng đặt file `gold_dataset_v2.csv` vào thư mục `data/`.")
#         st.stop()
#
#     df = load_data()
#     render_sidebar(df)
#
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "🏠 Tổng quan",
#         "📊 Phân tích giá",
#         "🔮 Dự đoán",
#         "📋 So sánh mô hình",
#         "📡 Tin tức & Thị trường",
#     ])
#
#     with tab1:
#         tab_overview(df)
#     with tab2:
#         tab_analysis(df)
#     with tab3:
#         tab_prediction(df)
#     with tab4:
#         tab_comparison(df)
#     with tab5:
#         tab_market(df)
#
#
# if __name__ == "__main__":
#     main()







"""
app.py — Ứng dụng Streamlit dự báo giá vàng SJC.
Tất cả văn bản UI bằng tiếng Việt.
Tải mô hình đã huấn luyện sẵn từ models/ để suy luận nhanh.
"""

# import json
# import warnings
# import logging
# from pathlib import Path
# from datetime import datetime
#
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
#
# warnings.filterwarnings("ignore")
#
# # Lazy import statsmodels để tránh lỗi scipy trên Python 3.13
# try:
#     from statsmodels.tsa.stattools import adfuller as _adfuller
#     STATSMODELS_OK = True
# except Exception:
#     STATSMODELS_OK = False
#
# def adfuller(series, autolag="AIC"):
#     """Wrapper: dùng statsmodels nếu có, fallback trả về kết quả giả."""
#     if STATSMODELS_OK:
#         return _adfuller(series, autolag=autolag)
#     # Fallback đơn giản: trả về tuple tương thích (stat, pval, lags, nobs, crit, ic)
#     return (0.0, 1.0, 0, len(series), {"1%": -3.43, "5%": -2.86, "10%": -2.57}, 0.0)
#
# # ──────────────────────────────────────────────
# # Cấu hình trang
# # ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="🥇 Dự Báo Giá Vàng SJC",
#     page_icon="🥇",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
#
# # ──────────────────────────────────────────────
# # Logging
# # ──────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # ──────────────────────────────────────────────
# # Hằng số
# # ──────────────────────────────────────────────
# DATA_PATH  = Path("data/gold_dataset_v2.csv")
# MODELS_DIR = Path("models")
# IMG_DIR    = Path("img")
# HORIZONS   = [1, 3, 7]
# TARGET_COL = "sell_price"
# EXOG_COLS  = ["xauusd_close", "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
#
# HORIZON_LABELS = {1: "1 ngày", 3: "3 ngày", 7: "7 ngày"}
#
# # ──────────────────────────────────────────────
# # CSS tùy chỉnh
# # ──────────────────────────────────────────────
# st.markdown("""
# <style>
#     .main-title { font-size: 2.2rem; font-weight: 700; color: #FFB703; }
#     .metric-card {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#         border: 1px solid #FFB703;
#         border-radius: 12px;
#         padding: 1rem;
#         text-align: center;
#         margin: 0.3rem;
#     }
#     .metric-value { font-size: 1.6rem; font-weight: 700; color: #FFB703; }
#     .metric-label { font-size: 0.85rem; color: #aaa; }
#     .info-box {
#         background: #0f3460;
#         border-left: 4px solid #FFB703;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .warn-box {
#         background: #3d1515;
#         border-left: 4px solid #E63946;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
#     .success-box {
#         background: #0d3b2e;
#         border-left: 4px solid #2A9D8F;
#         border-radius: 6px;
#         padding: 0.8rem 1rem;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # ══════════════════════════════════════════════
# # HÀM TIỆN ÍCH
# # ══════════════════════════════════════════════
#
# @st.cache_data(show_spinner="Đang tải dữ liệu...")
# def load_data() -> pd.DataFrame:
#     """Tải và tiền xử lý dataset."""
#     df = pd.read_csv(DATA_PATH, parse_dates=["date"])
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values("date").set_index("date")
#     full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
#     df = df.reindex(full_range).ffill().bfill()
#     return df
#
#
# @st.cache_resource(show_spinner="Đang tải mô hình...")
# def load_model_artifacts(horizon: int) -> dict:
#     """
#     Tải scaler, RandomForest, LightGBM và metadata cho một horizon.
#
#     Args:
#         horizon: Horizon h (1, 3 hoặc 7).
#
#     Returns:
#         Dict {'scaler', 'rf', 'lgbm', 'metadata'} hoặc {} nếu chưa có model.
#     """
#     prefix = MODELS_DIR / f"h{horizon}"
#     artifacts = {}
#     try:
#         artifacts["scaler"]   = joblib.load(f"{prefix}_scaler.joblib")
#         artifacts["rf"]       = joblib.load(f"{prefix}_rf.joblib")
#         artifacts["lgbm"]     = joblib.load(f"{prefix}_lgbm.joblib")
#         with open(f"{prefix}_metadata.json", "r", encoding="utf-8") as f:
#             artifacts["metadata"] = json.load(f)
#     except FileNotFoundError:
#         logger.warning(f"Chưa tìm thấy model artifacts cho H={horizon}.")
#     return artifacts
#
#
# def build_features_for_inference(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
#     """
#     Xây dựng đặc trưng CHỐNG RÒ RỈ — logic y hệt pipeline.py.
#     Hàm này phải GIỐNG HỆT hàm build_features() trong pipeline.py.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         DataFrame đặc trưng (1 hàng cuối là điểm dự báo hiện tại).
#     """
#     feat = df.copy()
#
#     for col in EXOG_COLS:
#         feat[f"{col}_s1"] = feat[col].shift(1)
#
#     feat["lag_1"] = feat[TARGET_COL].shift(1)
#     feat["lag_2"] = feat[TARGET_COL].shift(2)
#     feat["lag_3"] = feat[TARGET_COL].shift(3)
#
#     target_shifted = feat[TARGET_COL].shift(1)
#     feat["rolling_mean_7"]  = target_shifted.rolling(7,  min_periods=1).mean()
#     feat["rolling_mean_30"] = target_shifted.rolling(30, min_periods=1).mean()
#     feat["rolling_std_7"]   = target_shifted.rolling(7,  min_periods=2).std()
#     feat["rolling_std_30"]  = target_shifted.rolling(30, min_periods=2).std()
#     feat["rolling_max_7"]   = target_shifted.rolling(7,  min_periods=1).max()
#     feat["rolling_min_7"]   = target_shifted.rolling(7,  min_periods=1).min()
#
#     for col in EXOG_COLS:
#         s = feat[f"{col}_s1"]
#         feat[f"{col}_rm7"]  = s.rolling(7,  min_periods=1).mean()
#         feat[f"{col}_rm30"] = s.rolling(30, min_periods=1).mean()
#
#     feat["day_of_week"] = feat.index.dayofweek
#     feat["month"]       = feat.index.month
#     feat["is_weekend"]  = (feat.index.dayofweek >= 5).astype(int)
#     feat["quarter"]     = feat.index.quarter
#     feat["day_of_year"] = feat.index.dayofyear
#     feat["residual"]    = feat[TARGET_COL] - feat[TARGET_COL].shift(1)
#
#     cols_to_drop = EXOG_COLS + [TARGET_COL, "buy_price"]
#     feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])
#     return feat.dropna()
#
#
# def predict_price(df: pd.DataFrame, horizon: int) -> dict:
#     """
#     Suy luận trên hàng dữ liệu mới nhất.
#
#     Args:
#         df:      DataFrame đầy đủ.
#         horizon: Horizon h.
#
#     Returns:
#         Dict {'lgbm_pred', 'rf_pred', 'naive_pred', 'last_price',
#               'last_date', 'model_loaded'}.
#     """
#     arts = load_model_artifacts(horizon)
#     result = {
#         "lgbm_pred":    None,
#         "rf_pred":      None,
#         "naive_pred":   None,
#         "last_price":   df[TARGET_COL].iloc[-1],
#         "last_date":    df.index[-1],
#         "model_loaded": bool(arts),
#     }
#
#     # Naive
#     result["naive_pred"] = df[TARGET_COL].iloc[-(horizon)]
#
#     if not arts:
#         return result
#
#     meta         = arts["metadata"]
#     feature_cols = meta["feature_list"]
#     scaler       = arts["scaler"]
#
#     feat_df = build_features_for_inference(df, horizon)
#     if feat_df.empty:
#         return result
#
#     last_row = feat_df[feature_cols].iloc[[-1]]
#     try:
#         last_scaled = pd.DataFrame(
#             scaler.transform(last_row),
#             columns=feature_cols,
#         )
#     except Exception as exc:
#         logger.error(f"Lỗi scaling: {exc}")
#         return result
#
#     # Residual predictions
#     e_hat_lgb = arts["lgbm"].predict(last_scaled)[0]
#     e_hat_rf  = arts["rf"].predict(last_scaled)[0]
#
#     naive_base           = result["naive_pred"]
#     result["lgbm_pred"]  = naive_base + e_hat_lgb
#     result["rf_pred"]    = naive_base + e_hat_rf
#
#     return result
#
#
# # ══════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════
# def render_sidebar(df: pd.DataFrame) -> None:
#     """Thanh bên với thông tin nhanh về dataset."""
#     with st.sidebar:
#         st.markdown("## 🥇 Giá Vàng SJC")
#         st.markdown("---")
#
#         last_row   = df.iloc[-1]
#         last_date  = df.index[-1].strftime("%d/%m/%Y")
#         sell_price = last_row[TARGET_COL]
#         buy_price  = last_row.get("buy_price", np.nan)
#         xauusd     = last_row.get("xauusd_close", np.nan)
#
#         st.markdown(f"**📅 Ngày cập nhật:** {last_date}")
#         st.markdown(f"**💰 Giá bán SJC:** `{sell_price:.1f}` triệu VND/lượng")
#         st.markdown(f"**💵 XAUUSD:** `{xauusd:.0f}` USD/oz")
#         st.markdown("---")
#         st.markdown(f"**Tổng số ngày:** {len(df):,}")
#         st.markdown(f"**Từ:** {df.index.min().strftime('%d/%m/%Y')}")
#         st.markdown(f"**Đến:** {df.index.max().strftime('%d/%m/%Y')}")
#         st.markdown("---")
#         st.caption("© 2024 Luận văn tốt nghiệp CNTT")
#
#
# # ══════════════════════════════════════════════
# # TAB 1 — TỔNG QUAN
# # ══════════════════════════════════════════════
# def tab_overview(df: pd.DataFrame) -> None:
#     """Tab 1: Giới thiệu dự án và biện pháp chống rò rỉ."""
#     st.markdown('<p class="main-title">🥇 Dự Báo Giá Vàng SJC</p>', unsafe_allow_html=True)
#     st.markdown("#### Hệ thống dự báo đa horizon · Chống rò rỉ dữ liệu · Học phần dư")
#
#     col1, col2, col3, col4 = st.columns(4)
#     metrics = [
#         ("📅", "Số ngày dữ liệu", f"{len(df):,}"),
#         ("🏆", "Mô hình tốt nhất", "LightGBM"),
#         ("⏱️", "Horizon tối đa", "7 ngày"),
#         ("🛡️", "Chiến lược CV", "TimeSeriesSplit"),
#     ]
#     for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics):
#         with col:
#             st.markdown(
#                 f'<div class="metric-card"><div style="font-size:1.5rem">{icon}</div>'
#                 f'<div class="metric-value">{value}</div>'
#                 f'<div class="metric-label">{label}</div></div>',
#                 unsafe_allow_html=True,
#             )
#
#     st.markdown("---")
#
#     col_left, col_right = st.columns([1, 1])
#
#     with col_left:
#         st.markdown("### 🎯 Mục tiêu dự án")
#         st.markdown("""
#         Xây dựng pipeline dự báo **giá vàng SJC** (Công ty Vàng bạc Đá quý Sài Gòn)
#         với các horizon **1 ngày, 3 ngày và 7 ngày** sử dụng:
#
#         - 📈 **Dữ liệu thị trường:** XAUUSD, DXY, VIX, S&P500, VN-Index, USD/VND
#         - 🤖 **Mô hình ML:** Random Forest & LightGBM với **Residual Learning**
#         - 📊 **Tối ưu tự động:** Optuna + TimeSeriesSplit Cross-Validation
#         - 🔍 **Giải thích:** SHAP values
#         """)
#
#         st.markdown("### 📐 Dataset")
#         st.markdown(f"""
#         | Thuộc tính | Giá trị |
#         |---|---|
#         | **Tập huấn luyện** | 01/01/2010 → 31/12/2023 |
#         | **Tập kiểm tra** | 01/01/2024 → hiện tại |
#         | **Target** | `sell_price` (Triệu VND/lượng) |
#         | **Số đặc trưng** | ~30 sau feature engineering |
#         """)
#
#     with col_right:
#         st.markdown("### 🛡️ Biện pháp chống rò rỉ dữ liệu")
#         measures = [
#             ("✅", "Shift tất cả biến ngoại sinh `.shift(1)` trước khi tạo lag/rolling"),
#             ("✅", "Rolling window chỉ tính trên dữ liệu đến t-1 (sau khi shift)"),
#             ("✅", "Scaler **chỉ fit trên tập train**, transform riêng tập test"),
#             ("✅", "TimeSeriesSplit — **không shuffle** dữ liệu"),
#             ("✅", "Multi-horizon target: y[t+h-1] với đặc trưng chỉ từ t-1"),
#             ("✅", "Residual learning: e(t) = y(t) − y(t-1) với X tại t-1"),
#             ("✅", "Không có thông tin tập test nào được dùng khi huấn luyện"),
#         ]
#         for icon, text in measures:
#             st.markdown(
#                 f'<div class="success-box">{icon} {text}</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ tổng quan giá
#     st.markdown("---")
#     st.markdown("### 📈 Lịch sử giá vàng SJC (2010 – nay)")
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df.index, y=df[TARGET_COL],
#         mode="lines", name="Giá bán SJC",
#         line=dict(color="#FFB703", width=1.5),
#         fill="tozeroy", fillcolor="rgba(255,183,3,0.08)",
#     ))
#     # Đường phân cách train/test (dùng add_shape thay vì add_vline để tránh lỗi plotly)
#     fig.add_shape(
#         type="line",
#         x0=pd.Timestamp("2024-01-01"), x1=pd.Timestamp("2024-01-01"),
#         y0=0, y1=1, yref="paper",
#         line=dict(color="red", dash="dash", width=1.5),
#     )
#     fig.add_annotation(
#         x=pd.Timestamp("2024-01-01"), y=0.97, yref="paper",
#         text="Test set bắt đầu", showarrow=False,
#         font=dict(color="red", size=11),
#         xanchor="left", bgcolor="rgba(0,0,0,0.5)",
#     )
#     fig.update_layout(
#         height=350, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title="Triệu VND/lượng",
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 2 — PHÂN TÍCH GIÁ
# # ══════════════════════════════════════════════
# def tab_analysis(df: pd.DataFrame) -> None:
#     """Tab 2: EDA, tương quan, kiểm định ADF."""
#     st.markdown("## 📊 Phân tích giá & thị trường")
#
#     # -- Biểu đồ đa biến --
#     st.markdown("### 📈 So sánh các chỉ số thị trường")
#     selected_cols = st.multiselect(
#         "Chọn biến để so sánh:",
#         options=EXOG_COLS + [TARGET_COL],
#         default=["sell_price", "xauusd_close", "dxy_close"],
#     )
#     if selected_cols:
#         fig = go.Figure()
#         colors = px.colors.qualitative.Plotly
#         for i, col in enumerate(selected_cols):
#             norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
#             fig.add_trace(go.Scatter(
#                 x=df.index, y=norm,
#                 mode="lines", name=col,
#                 line=dict(color=colors[i % len(colors)], width=1.2),
#             ))
#         fig.update_layout(
#             height=350, template="plotly_dark",
#             yaxis_title="Chuẩn hóa Min-Max",
#             xaxis_title="Ngày",
#             margin=dict(l=10, r=10, t=20, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)
#
#     col_left, col_right = st.columns([1, 1])
#
#     # -- Heatmap tương quan --
#     with col_left:
#         st.markdown("### 🔗 Ma trận tương quan")
#         corr_cols = [TARGET_COL, "buy_price"] + EXOG_COLS
#         corr = df[corr_cols].corr()
#
#         fig, ax = plt.subplots(figsize=(8, 6))
#         fig.patch.set_facecolor("#0e1117")
#         ax.set_facecolor("#0e1117")
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
#                     center=0, square=True, linewidths=0.4,
#                     annot_kws={"size": 8, "color": "white"},
#                     ax=ax)
#         ax.set_title("Hệ số tương quan Pearson", color="white", fontsize=11)
#         plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
#         plt.yticks(color="white", fontsize=8)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     # -- ADF Test --
#     with col_right:
#         st.markdown("### 📉 Kiểm định tính dừng (ADF)")
#         if not STATSMODELS_OK:
#             st.warning(
#                 "⚠️ **statsmodels** không tương thích với Python 3.13 trên máy này. "
#                 "Kết quả ADF bên dưới chỉ mang tính minh hoạ.\n\n"
#                 "**Cách sửa:** `pip install statsmodels --upgrade` hoặc dùng Python 3.11/3.12."
#             )
#         adf_rows = []
#         for col in [TARGET_COL] + EXOG_COLS:
#             series = df[col].dropna()
#             stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
#             adf_rows.append({
#                 "Biến":         col,
#                 "ADF Statistic": round(stat, 3),
#                 "p-value":       round(pval, 5),
#                 "Dừng (p<0.05)": "✅ Có" if pval < 0.05 else "❌ Không",
#                 "CV 5%":         round(crit["5%"], 3),
#             })
#         adf_df = pd.DataFrame(adf_rows)
#
#         def color_stationary(val: str) -> str:
#             return "color: #2A9D8F" if "✅" in str(val) else "color: #E63946"
#
#         st.dataframe(
#             adf_df.style.map(color_stationary, subset=["Dừng (p<0.05)"]),
#             height=280,
#         )
#         st.markdown(
#             '<div class="info-box">💡 Chuỗi không dừng (p ≥ 0.05) gợi ý nên '
#             'dùng sai phân (differencing). Pipeline sử dụng <b>Residual Learning</b> '
#             'để ổn định hóa target.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # -- Phân phối target --
#     st.markdown("### 📊 Phân phối & biến động giá bán SJC")
#     col_a, col_b = st.columns(2)
#     with col_a:
#         fig = px.histogram(
#             df, x=TARGET_COL, nbins=60,
#             title="Phân phối giá bán SJC",
#             color_discrete_sequence=["#FFB703"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#     with col_b:
#         daily_change = df[TARGET_COL].diff()
#         fig = px.histogram(
#             daily_change.dropna(), x=TARGET_COL, nbins=60,
#             title="Phân phối thay đổi giá ngày hôm nay - hôm qua (Δ)",
#             color_discrete_sequence=["#457B9D"],
#             template="plotly_dark",
#         )
#         fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#         st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 3 — DỰ ĐOÁN
# # ══════════════════════════════════════════════
# def tab_prediction(df: pd.DataFrame) -> None:
#     """Tab 3: Giao diện dự báo giá vàng tương tác."""
#     st.markdown("## 🔮 Dự báo giá vàng SJC")
#
#     col_left, col_right = st.columns([1, 2])
#
#     with col_left:
#         st.markdown("### ⚙️ Cài đặt dự báo")
#         horizon_label = st.selectbox(
#             "Chọn horizon dự báo:",
#             options=list(HORIZON_LABELS.values()),
#         )
#         horizon = [k for k, v in HORIZON_LABELS.items() if v == horizon_label][0]
#
#         model_choice = st.radio(
#             "Chọn mô hình:",
#             options=["🤖 LightGBM (Khuyến nghị)", "🌲 Random Forest", "📏 Naive Baseline"],
#         )
#
#         run_predict = st.button("▶️ Chạy dự báo", type="primary", use_container_width=True)
#
#     with col_right:
#         if run_predict:
#             with st.spinner("Đang tính toán dự báo..."):
#                 result = predict_price(df, horizon)
#
#             if not result["model_loaded"]:
#                 st.markdown(
#                     '<div class="warn-box">⚠️ Chưa tìm thấy file model. '
#                     'Vui lòng chạy <code>python pipeline.py</code> trước.</div>',
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 # Chọn giá trị dự báo theo mô hình được chọn
#                 if "LightGBM" in model_choice:
#                     pred_val = result["lgbm_pred"]
#                     model_name = "LightGBM"
#                 elif "Random Forest" in model_choice:
#                     pred_val = result["rf_pred"]
#                     model_name = "Random Forest"
#                 else:
#                     pred_val = result["naive_pred"]
#                     model_name = "Naive Baseline"
#
#                 last_price = result["last_price"]
#                 delta      = (pred_val - last_price) if pred_val else 0
#                 delta_pct  = (delta / last_price * 100) if last_price else 0
#
#                 # Khoảng tin cậy đơn giản: ±2 * std của residuals test
#                 std_resid = df[TARGET_COL].diff().dropna().std()
#                 ci_low    = pred_val - 2 * std_resid * np.sqrt(horizon)
#                 ci_high   = pred_val + 2 * std_resid * np.sqrt(horizon)
#
#                 st.markdown("### 📌 Kết quả dự báo")
#                 c1, c2, c3 = st.columns(3)
#                 with c1:
#                     st.metric(
#                         label=f"Dự báo {horizon_label} ({model_name})",
#                         value=f"{pred_val:.2f} tr.VND",
#                         delta=f"{delta:+.2f} ({delta_pct:+.2f}%)",
#                     )
#                 with c2:
#                     st.metric(label="Giá hiện tại",     value=f"{last_price:.2f} tr.VND")
#                 with c3:
#                     st.metric(label="Ngày dữ liệu mới nhất",
#                               value=result["last_date"].strftime("%d/%m/%Y"))
#
#                 # Khoảng tin cậy
#                 st.markdown(
#                     f'<div class="info-box">📐 Khoảng tin cậy 95% (ước tính): '
#                     f'<b>{ci_low:.2f}</b> – <b>{ci_high:.2f}</b> triệu VND/lượng</div>',
#                     unsafe_allow_html=True,
#                 )
#
#                 # Biểu đồ 90 ngày gần nhất + dự báo
#                 st.markdown("---")
#                 st.markdown("### 📈 Biểu đồ giá 90 ngày gần nhất + dự báo")
#                 recent = df.tail(90)
#                 pred_date = recent.index[-1] + pd.Timedelta(days=horizon)
#
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=recent.index, y=recent[TARGET_COL],
#                     mode="lines", name="Lịch sử giá",
#                     line=dict(color="#FFB703", width=2),
#                 ))
#                 fig.add_trace(go.Scatter(
#                     x=[recent.index[-1], pred_date],
#                     y=[last_price, pred_val],
#                     mode="lines+markers",
#                     name=f"Dự báo H={horizon}",
#                     line=dict(color="#E63946", width=2.5, dash="dot"),
#                     marker=dict(size=[8, 12]),
#                 ))
#                 # CI
#                 fig.add_trace(go.Scatter(
#                     x=[pred_date, pred_date],
#                     y=[ci_low, ci_high],
#                     mode="lines", name="CI 95%",
#                     line=dict(color="#2A9D8F", width=6),
#                     opacity=0.6,
#                 ))
#                 fig.update_layout(
#                     height=350, template="plotly_dark",
#                     xaxis_title="Ngày",
#                     yaxis_title="Giá bán SJC (Triệu VND/lượng)",
#                     margin=dict(l=10, r=10, t=20, b=10),
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("👆 Chọn horizon & mô hình, sau đó nhấn **▶️ Chạy dự báo**")
#
#
# # ══════════════════════════════════════════════
# # TAB 4 — SO SÁNH MÔ HÌNH
# # ══════════════════════════════════════════════
# def tab_comparison(df: pd.DataFrame) -> None:
#     """Tab 4: Metrics đánh giá và biểu đồ so sánh."""
#     st.markdown("## 📋 So sánh mô hình")
#
#     metrics_path = IMG_DIR / "evaluation_metrics.csv"
#
#     if metrics_path.exists():
#         metrics_df = pd.read_csv(metrics_path)
#         st.markdown("### 📊 Bảng kết quả đánh giá (tập kiểm tra)")
#
#         def highlight_best(s: pd.Series) -> list[str]:
#             if s.name in ["MAE", "RMSE", "MAPE"]:
#                 best_idx = s.idxmin()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             elif s.name == "R2":
#                 best_idx = s.idxmax()
#                 return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
#             return [""] * len(s)
#
#         st.dataframe(
#             metrics_df.style.apply(highlight_best).format(
#                 {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"}
#             ),
#             use_container_width=True,
#         )
#
#         # Biểu đồ so sánh MAE
#         h1_df = metrics_df[metrics_df["Model"].str.contains("H=1")]
#         if not h1_df.empty:
#             col_a, col_b = st.columns(2)
#             for ax_col, metric in zip([col_a, col_b], ["MAE", "RMSE"]):
#                 with ax_col:
#                     fig = px.bar(
#                         h1_df.sort_values(metric),
#                         x=metric, y="Model",
#                         orientation="h",
#                         title=f"So sánh {metric} (H=1)",
#                         color=metric,
#                         color_continuous_scale="RdYlGn_r",
#                         template="plotly_dark",
#                     )
#                     fig.update_layout(
#                         height=300, coloraxis_showscale=False,
#                         margin=dict(l=10, r=10, t=40, b=10),
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.markdown(
#             '<div class="warn-box">⚠️ Chưa tìm thấy kết quả đánh giá. '
#             'Hãy chạy <code>python pipeline.py</code> để tạo metrics.</div>',
#             unsafe_allow_html=True,
#         )
#
#     # Biểu đồ dự báo đã lưu
#     st.markdown("---")
#     st.markdown("### 📈 Biểu đồ dự báo so với thực tế")
#     for h in HORIZONS:
#         img_path = IMG_DIR / f"predictions_h{h}.png"
#         if img_path.exists():
#             st.markdown(f"**Horizon H={h}:**")
#             st.image(str(img_path), use_column_width=True)
#
#     # SHAP plots
#     st.markdown("---")
#     st.markdown("### 🔍 SHAP Feature Importance (LightGBM)")
#     for h in HORIZONS:
#         shap_path = IMG_DIR / f"shap_summary_h{h}.png"
#         if shap_path.exists():
#             st.markdown(f"**H={h}:**")
#             col_s1, col_s2 = st.columns(2)
#             with col_s1:
#                 st.image(str(IMG_DIR / f"shap_summary_h{h}.png"), use_column_width=True)
#             with col_s2:
#                 bar_path = IMG_DIR / f"shap_bar_h{h}.png"
#                 if bar_path.exists():
#                     st.image(str(bar_path), use_column_width=True)
#
#
# # ══════════════════════════════════════════════
# # TAB 5 — TIN TỨC & THỊ TRƯỜNG
# # ══════════════════════════════════════════════
# def tab_market(df: pd.DataFrame) -> None:
#     """Tab 5: Thông tin thị trường hiện tại."""
#     st.markdown("## 📡 Tin tức & Thị trường")
#
#     last = df.iloc[-1]
#     last_date = df.index[-1].strftime("%d/%m/%Y")
#     st.markdown(f"*Dữ liệu cập nhật đến: **{last_date}***")
#     st.markdown("---")
#
#     # Chỉ số thị trường
#     st.markdown("### 📊 Chỉ số thị trường hiện tại")
#     market_data = [
#         ("🥇 XAUUSD",    "xauusd_close",  "USD/oz",         "#FFB703"),
#         ("💵 DXY",        "dxy_close",     "Điểm",           "#457B9D"),
#         ("😨 VIX",        "vix_close",     "Điểm",           "#E63946"),
#         ("📈 S&P500",     "sp500_close",   "USD",            "#2A9D8F"),
#         ("🇻🇳 VN-Index",   "vnindex",       "Điểm",           "#F4A261"),
#         ("💱 USD/VND",    "usd_vnd",       "VND",            "#6A4C93"),
#     ]
#     cols = st.columns(3)
#     for i, (label, col, unit, color) in enumerate(market_data):
#         val = last.get(col, np.nan)
#         prev_val = df[col].iloc[-2] if len(df) > 1 else val
#         chg = val - prev_val
#         chg_pct = (chg / prev_val * 100) if prev_val else 0
#
#         with cols[i % 3]:
#             direction = "▲" if chg >= 0 else "▼"
#             chg_color = "#2A9D8F" if chg >= 0 else "#E63946"
#             st.markdown(
#                 f'<div class="metric-card">'
#                 f'<div class="metric-label">{label}</div>'
#                 f'<div class="metric-value" style="color:{color}">'
#                 f'{val:,.1f}</div>'
#                 f'<div class="metric-label">{unit}</div>'
#                 f'<div style="color:{chg_color};font-size:0.9rem">'
#                 f'{direction} {abs(chg):,.2f} ({chg_pct:+.2f}%)</div>'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#
#     # Biểu đồ 30 ngày
#     st.markdown("---")
#     st.markdown("### 📈 Diễn biến 30 ngày gần nhất")
#     recent30 = df.tail(30)
#     chart_options = {
#         "Giá vàng SJC (Bán)": TARGET_COL,
#         "XAUUSD":              "xauusd_close",
#         "DXY":                 "dxy_close",
#         "VIX":                 "vix_close",
#         "VN-Index":            "vnindex",
#     }
#     selected = st.selectbox("Chọn chỉ số:", list(chart_options.keys()))
#     col_name  = chart_options[selected]
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=recent30.index, y=recent30[col_name],
#         mode="lines+markers",
#         name=selected,
#         line=dict(width=2.5, color="#FFB703"),
#         marker=dict(size=4),
#     ))
#     fig.update_layout(
#         height=300, template="plotly_dark",
#         xaxis_title="Ngày", yaxis_title=selected,
#         margin=dict(l=10, r=10, t=20, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Tương quan gần đây với SJC
#     st.markdown("---")
#     st.markdown("### 🔗 Tương quan 90 ngày gần nhất với Giá bán SJC")
#     recent90   = df.tail(90)
#     corr_vals  = recent90[EXOG_COLS].corrwith(recent90[TARGET_COL]).sort_values(ascending=False)
#     fig = px.bar(
#         corr_vals,
#         orientation="v",
#         title="Tương quan Pearson với sell_price (90 ngày gần nhất)",
#         color=corr_vals.values,
#         color_continuous_scale="RdYlGn",
#         range_color=[-1, 1],
#         template="plotly_dark",
#     )
#     fig.update_layout(
#         height=300,
#         coloraxis_showscale=True,
#         yaxis_title="Hệ số tương quan",
#         margin=dict(l=10, r=10, t=50, b=10),
#     )
#     st.plotly_chart(fig, use_container_width=True)
#
#
# # ══════════════════════════════════════════════
# # ĐIỂM NHẬP CHÍNH
# # ══════════════════════════════════════════════
# def main() -> None:
#     """Chạy ứng dụng Streamlit."""
#     # Tải dữ liệu
#     if not DATA_PATH.exists():
#         st.error(f"❌ Không tìm thấy file dữ liệu tại `{DATA_PATH}`. "
#                  "Vui lòng đặt file `gold_dataset_v2.csv` vào thư mục `data/`.")
#         st.stop()
#
#     df = load_data()
#     render_sidebar(df)
#
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "🏠 Tổng quan",
#         "📊 Phân tích giá",
#         "🔮 Dự đoán",
#         "📋 So sánh mô hình",
#         "📡 Tin tức & Thị trường",
#     ])
#
#     with tab1:
#         tab_overview(df)
#     with tab2:
#         tab_analysis(df)
#     with tab3:
#         tab_prediction(df)
#     with tab4:
#         tab_comparison(df)
#     with tab5:
#         tab_market(df)
#
#
# if __name__ == "__main__":
#     main()

# app.py — Ứng dụng Streamlit dự báo giá vàng SJC
# Toàn bộ UI bằng tiếng Việt, không icon trên tab.
# Tải mô hình đã huấn luyện từ thư mục models/

import json
import warnings
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ------------------------------
# Xử lý statsmodels (Python 3.13 compatibility)
# ------------------------------
try:
    from statsmodels.tsa.stattools import adfuller as _adfuller

    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False


def adfuller(series, autolag="AIC"):
    if STATSMODELS_OK:
        return _adfuller(series, autolag=autolag)
    # fallback
    return (0.0, 1.0, 0, len(series), {"1%": -3.43, "5%": -2.86, "10%": -2.57}, 0.0)


# ------------------------------
# Cấu hình trang
# ------------------------------
st.set_page_config(
    page_title="Dự Báo Giá Vàng SJC",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Hằng số đường dẫn
# ------------------------------
DATA_PATH = Path("data/gold_dataset_v2.csv")
MODELS_DIR = Path("models")
IMG_DIR = Path("img")
HORIZONS = [1, 3, 7]
TARGET_COL = "sell_price"
EXOG_COLS = ["xauusd_close", "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
HORIZON_LABELS = {1: "1 ngày", 3: "3 ngày", 7: "7 ngày"}

# ------------------------------
# CSS tùy chỉnh (giữ lại)
# ------------------------------
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #FFB703; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #FFB703;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #FFB703; }
    .metric-label { font-size: 0.85rem; color: #aaa; }
    .info-box {
        background: #0f3460;
        border-left: 4px solid #FFB703;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #3d1515;
        border-left: 4px solid #E63946;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #0d3b2e;
        border-left: 4px solid #2A9D8F;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------
# Hàm tải dữ liệu
# ------------------------------
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range).ffill().bfill()
    return df


# ------------------------------
# Tải model artifacts cho từng horizon
# ------------------------------
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model_artifacts(horizon: int) -> dict:
    prefix = MODELS_DIR / f"h{horizon}"
    artifacts = {}
    try:
        artifacts["scaler"] = joblib.load(f"{prefix}_scaler.joblib")
        artifacts["rf"] = joblib.load(f"{prefix}_rf.joblib")
        artifacts["lgbm"] = joblib.load(f"{prefix}_lgbm.joblib")
        with open(f"{prefix}_metadata.json", "r", encoding="utf-8") as f:
            artifacts["metadata"] = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Không tìm thấy model cho H={horizon}.")
    return artifacts


# ------------------------------
# Tái tạo đặc trưng (giống pipeline)
# ------------------------------
def build_features_for_inference(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    feat = df.copy()
    for col in EXOG_COLS:
        feat[f"{col}_s1"] = feat[col].shift(1)

    feat["lag_1"] = feat[TARGET_COL].shift(1)
    feat["lag_2"] = feat[TARGET_COL].shift(2)
    feat["lag_3"] = feat[TARGET_COL].shift(3)

    target_shifted = feat[TARGET_COL].shift(1)
    feat["rolling_mean_7"] = target_shifted.rolling(7, min_periods=1).mean()
    feat["rolling_mean_30"] = target_shifted.rolling(30, min_periods=1).mean()
    feat["rolling_std_7"] = target_shifted.rolling(7, min_periods=2).std()
    feat["rolling_std_30"] = target_shifted.rolling(30, min_periods=2).std()
    feat["rolling_max_7"] = target_shifted.rolling(7, min_periods=1).max()
    feat["rolling_min_7"] = target_shifted.rolling(7, min_periods=1).min()

    for col in EXOG_COLS:
        s = feat[f"{col}_s1"]
        feat[f"{col}_rm7"] = s.rolling(7, min_periods=1).mean()
        feat[f"{col}_rm30"] = s.rolling(30, min_periods=1).mean()

    feat["day_of_week"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["is_weekend"] = (feat.index.dayofweek >= 5).astype(int)
    feat["quarter"] = feat.index.quarter
    feat["day_of_year"] = feat.index.dayofyear
    feat["residual"] = feat[TARGET_COL] - feat[TARGET_COL].shift(1)

    cols_to_drop = EXOG_COLS + [TARGET_COL, "buy_price"]
    feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])
    return feat.dropna()


# ------------------------------
# Dự báo
# ------------------------------
def predict_price(df: pd.DataFrame, horizon: int) -> dict:
    arts = load_model_artifacts(horizon)
    result = {
        "lgbm_pred": None,
        "rf_pred": None,
        "naive_pred": None,
        "last_price": df[TARGET_COL].iloc[-1],
        "last_date": df.index[-1],
        "model_loaded": bool(arts),
    }

    # Naive baseline: lùi horizon ngày
    if len(df) >= horizon:
        result["naive_pred"] = df[TARGET_COL].iloc[-(horizon)]
    else:
        result["naive_pred"] = result["last_price"]

    if not arts:
        return result

    meta = arts["metadata"]
    feature_cols = meta["feature_list"]
    scaler = arts["scaler"]

    feat_df = build_features_for_inference(df, horizon)
    if feat_df.empty:
        return result

    last_row = feat_df[feature_cols].iloc[[-1]]
    try:
        last_scaled = pd.DataFrame(scaler.transform(last_row), columns=feature_cols)
    except Exception as exc:
        logger.error(f"Lỗi scaling: {exc}")
        return result

    e_hat_lgb = arts["lgbm"].predict(last_scaled)[0]
    e_hat_rf = arts["rf"].predict(last_scaled)[0]
    naive_base = result["naive_pred"]
    result["lgbm_pred"] = naive_base + e_hat_lgb
    result["rf_pred"] = naive_base + e_hat_rf
    return result


# ------------------------------
# Sidebar
# ------------------------------
def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("## Giá Vàng SJC")
        st.markdown("---")
        last_row = df.iloc[-1]
        last_date = df.index[-1].strftime("%d/%m/%Y")
        sell_price = last_row[TARGET_COL]
        xauusd = last_row.get("xauusd_close", np.nan)

        st.markdown(f"**📅 Ngày cập nhật:** {last_date}")
        st.markdown(f"**💰 Giá bán SJC:** `{sell_price:.1f}` triệu VND/lượng")
        st.markdown(f"**💵 XAUUSD:** `{xauusd:.0f}` USD/oz")
        st.markdown("---")
        st.markdown(f"**Tổng số ngày:** {len(df):,}")
        st.markdown(f"**Từ:** {df.index.min().strftime('%d/%m/%Y')}")
        st.markdown(f"**Đến:** {df.index.max().strftime('%d/%m/%Y')}")
        st.markdown("---")
        st.caption("Luận văn tốt nghiệp CNTT")


# ------------------------------
# Tab 1: Tổng quan
# ------------------------------
def tab_overview(df: pd.DataFrame) -> None:
    st.markdown('<p class="main-title">Dự Báo Giá Vàng SJC</p>', unsafe_allow_html=True)
    st.markdown("#### Hệ thống dự báo đa horizon · Chống rò rỉ dữ liệu · Học phần dư")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("📅", "Số ngày dữ liệu", f"{len(df):,}"),
        ("🏆", "Mô hình tốt nhất", "LightGBM"),
        ("⏱️", "Horizon tối đa", "7 ngày"),
        ("🛡️", "Chiến lược CV", "TimeSeriesSplit"),
    ]
    for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.5rem">{icon}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Mục tiêu dự án")
        st.markdown("""
        Xây dựng pipeline dự báo **giá vàng SJC** với các horizon **1, 3 và 7 ngày** sử dụng:
        - Dữ liệu thị trường: XAUUSD, DXY, VIX, S&P500, VN-Index, USD/VND
        - Mô hình ML: Random Forest & LightGBM với **Residual Learning**
        - Tối ưu tự động: Optuna + TimeSeriesSplit
        - Giải thích: SHAP values
        """)
        st.markdown("### Dataset")
        st.markdown(f"""
        | Thuộc tính | Giá trị |
        |---|---|
        | **Tập huấn luyện** | 01/01/2010 → 31/12/2023 |
        | **Tập kiểm tra** | 01/01/2024 → hiện tại |
        | **Target** | `sell_price` (triệu VND/lượng) |
        | **Số đặc trưng** | ~30 sau feature engineering |
        """)

    with col_right:
        st.markdown("### Biện pháp chống rò rỉ dữ liệu")
        measures = [
            ("✅", "Shift tất cả biến ngoại sinh `.shift(1)` trước khi tạo lag/rolling"),
            ("✅", "Rolling window chỉ tính trên dữ liệu đến t-1 (sau khi shift)"),
            ("✅", "Scaler **chỉ fit trên tập train**, transform riêng tập test"),
            ("✅", "TimeSeriesSplit — **không shuffle** dữ liệu"),
            ("✅", "Multi-horizon target: y[t+h-1] với đặc trưng chỉ từ t-1"),
            ("✅", "Residual learning: e(t) = y(t) − y(t-1) với X tại t-1"),
        ]
        for icon, text in measures:
            st.markdown(f'<div class="success-box">{icon} {text}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Lịch sử giá vàng SJC (2010 – nay)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[TARGET_COL],
        mode="lines", name="Giá bán SJC",
        line=dict(color="#FFB703", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,183,3,0.08)",
    ))
    fig.add_shape(
        type="line",
        x0=pd.Timestamp("2024-01-01"), x1=pd.Timestamp("2024-01-01"),
        y0=0, y1=1, yref="paper",
        line=dict(color="red", dash="dash", width=1.5),
    )
    fig.add_annotation(
        x=pd.Timestamp("2024-01-01"), y=0.97, yref="paper",
        text="Test set bắt đầu", showarrow=False,
        font=dict(color="red", size=11),
        xanchor="left", bgcolor="rgba(0,0,0,0.5)",
    )
    fig.update_layout(height=350, template="plotly_dark", xaxis_title="Ngày", yaxis_title="Triệu VND/lượng")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Tab 2: Phân tích giá
# ------------------------------
def tab_analysis(df: pd.DataFrame) -> None:
    st.markdown("## Phân tích giá & thị trường")

    st.markdown("### So sánh các chỉ số thị trường")
    selected_cols = st.multiselect(
        "Chọn biến để so sánh:",
        options=EXOG_COLS + [TARGET_COL],
        default=["sell_price", "xauusd_close", "dxy_close"],
    )
    if selected_cols:
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, col in enumerate(selected_cols):
            norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            fig.add_trace(go.Scatter(x=df.index, y=norm, mode="lines", name=col,
                                     line=dict(color=colors[i % len(colors)], width=1.2)))
        fig.update_layout(height=350, template="plotly_dark", yaxis_title="Chuẩn hóa Min-Max", xaxis_title="Ngày")
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Ma trận tương quan")
        corr_cols = [TARGET_COL, "buy_price"] + EXOG_COLS
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True, linewidths=0.4,
                    annot_kws={"size": 8, "color": "white"}, ax=ax)
        ax.set_title("Hệ số tương quan Pearson", color="white", fontsize=11)
        plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
        plt.yticks(color="white", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("### Kiểm định tính dừng (ADF)")
        if not STATSMODELS_OK:
            st.warning("⚠️ statsmodels không khả dụng. Kết quả ADF chỉ mang tính minh họa.")
        adf_rows = []
        for col in [TARGET_COL] + EXOG_COLS:
            series = df[col].dropna()
            stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
            adf_rows.append({
                "Biến": col,
                "ADF Statistic": round(stat, 3),
                "p-value": round(pval, 5),
                "Dừng (p<0.05)": "✅ Có" if pval < 0.05 else "❌ Không",
                "CV 5%": round(crit["5%"], 3),
            })
        adf_df = pd.DataFrame(adf_rows)

        def color_stationary(val):
            return "color: #2A9D8F" if "✅" in str(val) else "color: #E63946"

        st.dataframe(adf_df.style.map(color_stationary, subset=["Dừng (p<0.05)"]), height=280)
        st.markdown(
            '<div class="info-box">💡 Chuỗi không dừng gợi ý nên dùng sai phân. Pipeline sử dụng <b>Residual Learning</b> để ổn định hóa target.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Phân phối & biến động giá bán SJC")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x=TARGET_COL, nbins=60, title="Phân phối giá bán SJC",
                           color_discrete_sequence=["#FFB703"], template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        daily_change = df[TARGET_COL].diff()
        fig = px.histogram(daily_change.dropna(), x=TARGET_COL, nbins=60,
                           title="Phân phối thay đổi giá ngày hôm nay - hôm qua (Δ)",
                           color_discrete_sequence=["#457B9D"], template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Tab 3: Dự đoán
# ------------------------------
def tab_prediction(df: pd.DataFrame) -> None:
    st.markdown("## Dự báo giá vàng SJC")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Cài đặt dự báo")
        horizon_label = st.selectbox("Chọn horizon dự báo:", list(HORIZON_LABELS.values()))
        horizon = [k for k, v in HORIZON_LABELS.items() if v == horizon_label][0]
        model_choice = st.radio("Chọn mô hình:", ["LightGBM (Khuyến nghị)", "Random Forest", "Naive Baseline"])
        run_predict = st.button("Chạy dự báo", type="primary", use_container_width=True)

    with col_right:
        if run_predict:
            with st.spinner("Đang tính toán dự báo..."):
                result = predict_price(df, horizon)

            if not result["model_loaded"]:
                st.markdown('<div class="warn-box">⚠️ Chưa tìm thấy file model. Hãy chạy pipeline trước.</div>',
                            unsafe_allow_html=True)
            else:
                if "LightGBM" in model_choice:
                    pred_val = result["lgbm_pred"]
                    model_name = "LightGBM"
                elif "Random Forest" in model_choice:
                    pred_val = result["rf_pred"]
                    model_name = "Random Forest"
                else:
                    pred_val = result["naive_pred"]
                    model_name = "Naive Baseline"

                last_price = result["last_price"]
                delta = (pred_val - last_price) if pred_val else 0
                delta_pct = (delta / last_price * 100) if last_price else 0

                std_resid = df[TARGET_COL].diff().dropna().std()
                ci_low = pred_val - 2 * std_resid * np.sqrt(horizon)
                ci_high = pred_val + 2 * std_resid * np.sqrt(horizon)

                st.markdown("### Kết quả dự báo")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(label=f"Dự báo {horizon_label} ({model_name})", value=f"{pred_val:.2f} tr.VND",
                              delta=f"{delta:+.2f} ({delta_pct:+.2f}%)")
                with c2:
                    st.metric(label="Giá hiện tại", value=f"{last_price:.2f} tr.VND")
                with c3:
                    st.metric(label="Ngày dữ liệu mới nhất", value=result["last_date"].strftime("%d/%m/%Y"))

                st.markdown(
                    f'<div class="info-box">📐 Khoảng tin cậy 95% (ước tính): <b>{ci_low:.2f}</b> – <b>{ci_high:.2f}</b> triệu VND/lượng</div>',
                    unsafe_allow_html=True)

                st.markdown("### Biểu đồ giá 90 ngày gần nhất + dự báo")
                recent = df.tail(90)
                pred_date = recent.index[-1] + pd.Timedelta(days=horizon)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent.index, y=recent[TARGET_COL], mode="lines", name="Lịch sử giá",
                                         line=dict(color="#FFB703", width=2)))
                fig.add_trace(
                    go.Scatter(x=[recent.index[-1], pred_date], y=[last_price, pred_val], mode="lines+markers",
                               name=f"Dự báo H={horizon}", line=dict(color="#E63946", width=2.5, dash="dot"),
                               marker=dict(size=[8, 12])))
                fig.add_trace(go.Scatter(x=[pred_date, pred_date], y=[ci_low, ci_high], mode="lines", name="CI 95%",
                                         line=dict(color="#2A9D8F", width=6), opacity=0.6))
                fig.update_layout(height=350, template="plotly_dark", xaxis_title="Ngày",
                                  yaxis_title="Giá bán SJC (Triệu VND/lượng)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Chọn horizon & mô hình, sau đó nhấn **Chạy dự báo**")


# ------------------------------
# Tab 4: So sánh mô hình
# ------------------------------
def tab_comparison(df: pd.DataFrame) -> None:
    st.markdown("## So sánh mô hình")
    metrics_path = IMG_DIR / "evaluation_metrics.csv"

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.markdown("### Bảng kết quả đánh giá (tập kiểm tra)")

        def highlight_best(s: pd.Series):
            if s.name in ["MAE", "RMSE", "MAPE"]:
                best_idx = s.idxmin()
                return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
            elif s.name == "R2":
                best_idx = s.idxmax()
                return ["background-color: #0d3b2e" if i == best_idx else "" for i in s.index]
            return [""] * len(s)

        st.dataframe(
            metrics_df.style.apply(highlight_best).format(
                {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"}),
            use_container_width=True,
        )

        h1_df = metrics_df[metrics_df["Model"].str.contains("H=1")]
        if not h1_df.empty:
            col_a, col_b = st.columns(2)
            for ax_col, metric in zip([col_a, col_b], ["MAE", "RMSE"]):
                with ax_col:
                    fig = px.bar(h1_df.sort_values(metric), x=metric, y="Model", orientation="h",
                                 title=f"So sánh {metric} (H=1)", color=metric, color_continuous_scale="RdYlGn_r",
                                 template="plotly_dark")
                    fig.update_layout(height=300, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="warn-box">⚠️ Chưa tìm thấy kết quả đánh giá. Hãy chạy pipeline trước.</div>',
                    unsafe_allow_html=True)

    st.markdown("### Biểu đồ dự báo so với thực tế")
    for h in HORIZONS:
        img_path = IMG_DIR / f"predictions_h{h}.png"
        if img_path.exists():
            st.markdown(f"**Horizon H={h}:**")
            st.image(str(img_path), use_container_width=True)

    st.markdown("### SHAP Feature Importance (LightGBM)")
    for h in HORIZONS:
        shap_path = IMG_DIR / f"shap_summary_h{h}.png"
        if shap_path.exists():
            st.markdown(f"**H={h}:**")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.image(str(shap_path), use_container_width=True)
            with col_s2:
                bar_path = IMG_DIR / f"shap_bar_h{h}.png"
                if bar_path.exists():
                    st.image(str(bar_path), use_container_width=True)


# ------------------------------
# Tab 5: Tin tức & Thị trường
# ------------------------------
def tab_market(df: pd.DataFrame) -> None:
    st.markdown("## Tin tức & Thị trường")
    last = df.iloc[-1]
    last_date = df.index[-1].strftime("%d/%m/%Y")
    st.markdown(f"*Dữ liệu cập nhật đến: **{last_date}***")
    st.markdown("---")

    st.markdown("### Chỉ số thị trường hiện tại")
    market_data = [
        ("XAUUSD", "xauusd_close", "USD/oz", "#FFB703"),
        ("DXY", "dxy_close", "Điểm", "#457B9D"),
        ("VIX", "vix_close", "Điểm", "#E63946"),
        ("S&P500", "sp500_close", "USD", "#2A9D8F"),
        ("VN-Index", "vnindex", "Điểm", "#F4A261"),
        ("USD/VND", "usd_vnd", "VND", "#6A4C93"),
    ]
    cols = st.columns(3)
    for i, (label, col, unit, color) in enumerate(market_data):
        val = last.get(col, np.nan)
        prev_val = df[col].iloc[-2] if len(df) > 1 else val
        chg = val - prev_val
        chg_pct = (chg / prev_val * 100) if prev_val else 0
        direction = "▲" if chg >= 0 else "▼"
        chg_color = "#2A9D8F" if chg >= 0 else "#E63946"
        with cols[i % 3]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{color}">{val:,.1f}</div>'
                f'<div class="metric-label">{unit}</div>'
                f'<div style="color:{chg_color}">{direction} {abs(chg):,.2f} ({chg_pct:+.2f}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Diễn biến 30 ngày gần nhất")
    recent30 = df.tail(30)
    chart_options = {
        "Giá vàng SJC (Bán)": TARGET_COL,
        "XAUUSD": "xauusd_close",
        "DXY": "dxy_close",
        "VIX": "vix_close",
        "VN-Index": "vnindex",
    }
    selected = st.selectbox("Chọn chỉ số:", list(chart_options.keys()))
    col_name = chart_options[selected]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent30.index, y=recent30[col_name], mode="lines+markers", name=selected,
                             line=dict(width=2.5, color="#FFB703"), marker=dict(size=4)))
    fig.update_layout(height=300, template="plotly_dark", xaxis_title="Ngày", yaxis_title=selected)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tương quan 90 ngày gần nhất với Giá bán SJC")
    recent90 = df.tail(90)
    corr_vals = recent90[EXOG_COLS].corrwith(recent90[TARGET_COL]).sort_values(ascending=False)
    fig = px.bar(corr_vals, orientation="v", title="Tương quan Pearson với sell_price (90 ngày gần nhất)",
                 color=corr_vals.values, color_continuous_scale="RdYlGn", range_color=[-1, 1],
                 template="plotly_dark")
    fig.update_layout(height=300, coloraxis_showscale=True, yaxis_title="Hệ số tương quan")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Main
# ------------------------------
def main() -> None:
    if not DATA_PATH.exists():
        st.error(
            f"❌ Không tìm thấy file dữ liệu tại `{DATA_PATH}`. Vui lòng đặt file `gold_dataset_v2.csv` vào thư mục `data/`.")
        st.stop()

    df = load_data()
    render_sidebar(df)

    tabs = st.tabs(["Tổng quan", "Phân tích giá", "Dự đoán", "So sánh mô hình", "Tin tức & Thị trường"])
    with tabs[0]:
        tab_overview(df)
    with tabs[1]:
        tab_analysis(df)
    with tabs[2]:
        tab_prediction(df)
    with tabs[3]:
        tab_comparison(df)
    with tabs[4]:
        tab_market(df)


if __name__ == "__main__":
    main()