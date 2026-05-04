"""
pipeline.py — Pipeline huấn luyện chống rò rỉ dữ liệu (anti-leakage) cho dự báo
giá vàng SJC đa horizon (h=1, h=3, h=7).

Cách dùng:
    python pipeline.py

Sẽ tạo ra:
    - models/  : Các file .joblib (scaler, model, metadata cho mỗi horizon)
    - img/     : Các biểu đồ EDA, đánh giá, SHAP
"""

# ═══════════════════════════════════════════════════════
# 0. IMPORTS & CẤU HÌNH TOÀN CỤC
# ═══════════════════════════════════════════════════════
import warnings
import logging
import json
import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
import optuna
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP chưa được cài đặt. Bỏ qua phần SHAP.")

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Seed toàn cục
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Thư mục đầu ra
MODELS_DIR = Path("models")
IMG_DIR    = Path("img")
DATA_PATH  = Path("data/gold_dataset_v2.csv")
MODELS_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# Cấu hình train/test
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
HORIZONS    = [1, 3, 7]
N_SPLITS_CV = 3
OPTUNA_TRIALS = 50

EXOG_COLS = ["xauusd_close", "dxy_close", "vix_close", "usd_vnd", "vnindex", "sp500_close"]
TARGET_COL = "sell_price"


# ═══════════════════════════════════════════════════════
# 1. TẢI & TIỀN XỬ LÝ DỮ LIỆU
# ═══════════════════════════════════════════════════════
def load_and_preprocess(path: Path) -> pd.DataFrame:
    """
    Tải CSV, reindex theo ngày liên tục, xử lý NaN.

    Args:
        path: Đường dẫn đến file CSV.

    Returns:
        DataFrame đã được làm sạch với chỉ mục ngày liên tục.
    """
    logger.info(f"Đang tải dữ liệu từ {path}...")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Reindex toàn bộ khoảng ngày để bao gồm cuối tuần/ngày lễ
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)

    # Forward-fill cuối tuần & ngày lễ → sau đó backward-fill NaN đầu đoạn
    df = df.ffill().bfill()

    logger.info(f"Dataset: {df.shape[0]} hàng, {df.shape[1]} cột | "
                f"{df.index.min().date()} → {df.index.max().date()}")
    return df


# ═══════════════════════════════════════════════════════
# 2. EDA
# ═══════════════════════════════════════════════════════
def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """
    Thực hiện EDA cơ bản: thống kê mô tả, biểu đồ giá, heatmap tương quan,
    kiểm định ADF, và lưu ảnh vào img/.

    Args:
        df: DataFrame đã xử lý.

    Returns:
        Dict chứa kết quả ADF theo từng cột.
    """
    logger.info("=== EDA ===")

    # --- Thống kê mô tả ---
    desc = df.describe().T
    logger.info(f"\n{desc}")

    # --- Biểu đồ giá vàng SJC ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(df.index, df[TARGET_COL], color="#FFB703", linewidth=1.2, label="Giá bán SJC")
    axes[0].set_title("Giá vàng SJC (Triệu VND/lượng)", fontsize=13)
    axes[0].set_ylabel("Giá bán (Triệu VND)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["xauusd_close"], color="#023E8A", linewidth=1, label="XAUUSD")
    axes[1].set_title("Giá vàng thế giới XAUUSD (USD/oz)", fontsize=13)
    axes[1].set_ylabel("USD/oz")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "01_sjc_price_trend.png", dpi=150)
    plt.close()

    # --- Heatmap tương quan ---
    corr_cols = [TARGET_COL, "buy_price"] + EXOG_COLS
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.5, ax=ax, mask=False)
    ax.set_title("Ma trận tương quan — Các biến thị trường", fontsize=13)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close()

    # --- Kiểm định ADF ---
    adf_results: dict[str, Any] = {}
    for col in [TARGET_COL] + EXOG_COLS:
        series = df[col].dropna()
        stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
        adf_results[col] = {
            "adf_stat": round(stat, 4),
            "p_value":  round(pval, 6),
            "stationary": pval < 0.05,
        }
    adf_df = pd.DataFrame(adf_results).T
    logger.info(f"\n[ADF Test]\n{adf_df}")

    # Lưu kết quả ADF
    adf_df.to_csv(IMG_DIR / "adf_test_results.csv")

    return adf_results


# ═══════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — CHỐNG RÒ RỈ
# ═══════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Xây dựng đặc trưng CHỐNG RÒ RỈ cho một horizon cụ thể.

    Quy tắc chính:
    - Tất cả biến ngoại sinh được SHIFT(1) trước khi tạo lag/rolling.
    - Rolling window tính trên dữ liệu đã shift (không bao giờ chứa t).
    - Target cho horizon h: y[t+h-1] → dự báo h bước phía trước.
    - Residual e(t) = y(t) - y(t-1) được tính để học phần dư.

    Args:
        df:      DataFrame gốc (index là DatetimeIndex).
        horizon: Số bước dự báo (1, 3 hoặc 7).

    Returns:
        DataFrame đặc trưng đã sẵn sàng để train (chưa scale).
    """
    feat = df.copy()

    # ── 3.1 Shift biến ngoại sinh (chống rò rỉ cốt lõi) ──
    # Tại thời điểm T, ta CHỈ được biết giá trị ngoại sinh tại T-1 trở về trước.
    for col in EXOG_COLS:
        feat[f"{col}_s1"] = feat[col].shift(1)   # T-1

    # ── 3.2 Lag của target (sử dụng shift để đảm bảo không rò rỉ) ──
    feat["lag_1"] = feat[TARGET_COL].shift(1)   # y(t-1)
    feat["lag_2"] = feat[TARGET_COL].shift(2)   # y(t-2)
    feat["lag_3"] = feat[TARGET_COL].shift(3)   # y(t-3)

    # ── 3.3 Rolling statistics trên target đã shift ──
    # Lấy chuỗi shift(1) để rolling window không bao giờ thấy y(t)
    target_shifted = feat[TARGET_COL].shift(1)
    feat["rolling_mean_7"]  = target_shifted.rolling(7,  min_periods=1).mean()
    feat["rolling_mean_30"] = target_shifted.rolling(30, min_periods=1).mean()
    feat["rolling_std_7"]   = target_shifted.rolling(7,  min_periods=2).std()
    feat["rolling_std_30"]  = target_shifted.rolling(30, min_periods=2).std()
    feat["rolling_max_7"]   = target_shifted.rolling(7,  min_periods=1).max()
    feat["rolling_min_7"]   = target_shifted.rolling(7,  min_periods=1).min()

    # ── 3.4 Rolling stats trên biến ngoại sinh đã shift ──
    for col in EXOG_COLS:
        s = feat[f"{col}_s1"]
        feat[f"{col}_rm7"]  = s.rolling(7,  min_periods=1).mean()
        feat[f"{col}_rm30"] = s.rolling(30, min_periods=1).mean()

    # ── 3.5 Đặc trưng thời gian ──
    feat["day_of_week"] = feat.index.dayofweek          # 0=Thứ Hai
    feat["month"]       = feat.index.month
    feat["is_weekend"]  = (feat.index.dayofweek >= 5).astype(int)
    feat["quarter"]     = feat.index.quarter
    feat["day_of_year"] = feat.index.dayofyear

    # ── 3.6 Residual e(t) = y(t) - y(t-1) ──
    # Đây là TARGET cho các mô hình ML học phần dư.
    feat["residual"] = feat[TARGET_COL] - feat[TARGET_COL].shift(1)

    # ── 3.7 Target cho horizon h ──
    # Direct multi-step: dự báo y(t+h-1) sử dụng đặc trưng tại t-1
    feat[f"target_h{horizon}"] = feat[TARGET_COL].shift(-(horizon - 1))

    # ── 3.8 Loại cột gốc (chỉ giữ các cột shifted/engineered) ──
    cols_to_drop = EXOG_COLS + [TARGET_COL, "buy_price"]
    feat = feat.drop(columns=[c for c in cols_to_drop if c in feat.columns])

    feat = feat.dropna()
    return feat


def get_feature_columns(feat_df: pd.DataFrame, horizon: int) -> list[str]:
    """Trả về danh sách tên đặc trưng đầu vào (không bao gồm target/residual)."""
    exclude = {f"target_h{h}" for h in HORIZONS} | {"residual"}
    return [c for c in feat_df.columns if c not in exclude]


# ═══════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════
def split_data(
    feat_df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tách train/test theo mốc thời gian (KHÔNG shuffle).

    Args:
        feat_df: DataFrame đặc trưng đã xây dựng.
        horizon: Horizon hiện tại.

    Returns:
        X_train, X_test, y_train, y_test (residual targets)
    """
    feature_cols = get_feature_columns(feat_df, horizon)
    target_col   = f"target_h{horizon}"

    train = feat_df[feat_df.index <= TRAIN_END]
    test  = feat_df[feat_df.index >= TEST_START]

    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train = train["residual"]    # Train ML trên residual
    y_test  = test["residual"]

    logger.info(f"[H={horizon}] Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════
# 5. SCALING (fit CHỈ trên train)
# ═══════════════════════════════════════════════════════
def fit_scaler(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Fit MinMaxScaler trên X_train, transform cả train và test.

    Args:
        X_train: Đặc trưng tập train.
        X_test:  Đặc trưng tập test.

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    return X_train_scaled, X_test_scaled, scaler


# ═══════════════════════════════════════════════════════
# 6. MÔ HÌNH
# ═══════════════════════════════════════════════════════

# ─── 6a. Naive Forecast ───────────────────────────────
def naive_forecast(
    df_full: pd.DataFrame, horizon: int
) -> tuple[pd.Series, pd.Series]:
    """
    Baseline Naive: y_hat(t) = y(t - horizon).

    Args:
        df_full: DataFrame gốc đã xử lý (có TARGET_COL).
        horizon: Horizon (1, 3, 7).

    Returns:
        y_true, y_pred (cả hai trên tập test)
    """
    test_df = df_full[df_full.index >= TEST_START][[TARGET_COL]].copy()
    test_df["naive_pred"] = df_full[TARGET_COL].shift(horizon).loc[test_df.index]
    test_df = test_df.dropna()
    return test_df[TARGET_COL], test_df["naive_pred"]


# ─── 6b. Prophet ──────────────────────────────────────
def train_prophet(
    df_full: pd.DataFrame,
) -> tuple[Prophet, pd.DataFrame]:
    """
    Huấn luyện mô hình Prophet trên tập train.

    Args:
        df_full: DataFrame gốc.

    Returns:
        model, forecast_df (bao gồm cả test period)
    """
    logger.info("[Prophet] Đang huấn luyện...")
    train_df = df_full[df_full.index <= TRAIN_END][[TARGET_COL]].copy()
    train_df = train_df.reset_index().rename(columns={"index": "ds", TARGET_COL: "y"})
    train_df["ds"] = pd.to_datetime(train_df["ds"])

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )
    m.fit(train_df)

    future = m.make_future_dataframe(
        periods=(pd.to_datetime(df_full.index.max()) - pd.to_datetime(TRAIN_END)).days + 10
    )
    forecast = m.predict(future)
    logger.info("[Prophet] Hoàn thành.")
    return m, forecast


# ─── 6c. SARIMAX ─────────────────────────────────────
def train_sarimax(
    df_full: pd.DataFrame,
) -> tuple[Any, pd.Series]:
    """
    Huấn luyện SARIMAX(1,1,1)(1,1,1,7) trên tập train.

    Args:
        df_full: DataFrame gốc.

    Returns:
        model_fit, predictions (pd.Series trên test period)
    """
    logger.info("[SARIMAX] Đang huấn luyện — có thể mất vài phút...")
    train_series = df_full[df_full.index <= TRAIN_END][TARGET_COL]
    test_series  = df_full[df_full.index >= TEST_START][TARGET_COL]

    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)

    # Dự báo ngoài mẫu
    forecast = model_fit.forecast(steps=len(test_series))
    forecast.index = test_series.index
    logger.info("[SARIMAX] Hoàn thành.")
    return model_fit, forecast


# ─── 6d. Random Forest (Residual Learning) ────────────
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    """
    Huấn luyện Random Forest để dự báo residual.

    Args:
        X_train: Đặc trưng tập train (đã scale).
        y_train: Residual tập train.

    Returns:
        Mô hình đã được huấn luyện.
    """
    logger.info("[RandomForest] Đang huấn luyện...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    logger.info("[RandomForest] Hoàn thành.")
    return rf


# ─── 6e. LightGBM + Optuna (Residual Learning) ────────
def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
) -> dict[str, Any]:
    """
    Tối ưu siêu tham số LightGBM bằng Optuna với TimeSeriesSplit.

    Args:
        X_train: Đặc trưng train (đã scale).
        y_train: Residual train.
        horizon: Horizon hiện tại.

    Returns:
        Siêu tham số tốt nhất tìm được.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "regression",
            "metric":            "mae",
            "verbosity":         -1,
            "boosting_type":     "gbdt",
            "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state":      RANDOM_SEED,
        }
        mae_scores: list[float] = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))
        return float(np.mean(mae_scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    logger.info(f"[Optuna H={horizon}] Bắt đầu tối ưu ({OPTUNA_TRIALS} trials)...")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    best_params["objective"]    = "regression"
    best_params["metric"]       = "mae"
    best_params["verbosity"]    = -1
    logger.info(f"[Optuna H={horizon}] Best MAE={study.best_value:.4f}, params={best_params}")
    return best_params


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> lgb.LGBMRegressor:
    """
    Huấn luyện LightGBM với siêu tham số tối ưu trên toàn bộ tập train.

    Args:
        X_train: Đặc trưng train.
        y_train: Residual train.
        params:  Siêu tham số từ Optuna.

    Returns:
        Mô hình đã được huấn luyện.
    """
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════
# 7. RESIDUAL LEARNING — KẾT HỢP DỰ BÁO CUỐI
# ═══════════════════════════════════════════════════════
def predict_with_residual(
    ml_model: Any,
    X_test: pd.DataFrame,
    y_test_series_full: pd.Series,   # Chuỗi TARGET_COL đầy đủ (có cả train)
    horizon: int,
) -> pd.Series:
    """
    Kết hợp Naive Forecast với dự báo residual từ mô hình ML.

    Công thức:
        y_hat_final(t) = y_naive(t) + e_hat_ml(t)
        y_naive(t)     = y(t - horizon)          ← dự báo naive h bước
        e_hat_ml(t)    = ml_model.predict(X_test)

    Args:
        ml_model:            Mô hình ML (RF hoặc LightGBM).
        X_test:              Đặc trưng tập test.
        y_test_series_full:  Chuỗi TARGET_COL đầy đủ.
        horizon:             Horizon hiện tại.

    Returns:
        pd.Series dự báo cuối (index khớp với X_test).
    """
    # Naive baseline: y(t - horizon)
    naive = y_test_series_full.shift(horizon).loc[X_test.index]

    # Dự báo residual
    e_hat = pd.Series(ml_model.predict(X_test), index=X_test.index)

    final_pred = naive + e_hat
    return final_pred


# ═══════════════════════════════════════════════════════
# 8. ĐÁNH GIÁ
# ═══════════════════════════════════════════════════════
def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, model_name: str
) -> dict[str, float]:
    """
    Tính MAE, RMSE, MAPE, R².

    Args:
        y_true:     Giá trị thực.
        y_pred:     Giá trị dự báo.
        model_name: Tên mô hình (dùng cho log).

    Returns:
        Dict {'MAE', 'RMSE', 'MAPE', 'R2'}.
    """
    mask  = y_true.notna() & y_pred.notna()
    yt    = y_true[mask].values
    yp    = y_pred[mask].values

    mae   = mean_absolute_error(yt, yp)
    rmse  = np.sqrt(mean_squared_error(yt, yp))
    mape  = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
    r2    = r2_score(yt, yp)

    logger.info(f"[{model_name}] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}% | R²={r2:.4f}")
    return {"Model": model_name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "R2": round(r2, 4)}


def plot_predictions(
    y_true: pd.Series, preds_dict: dict[str, pd.Series], horizon: int
) -> None:
    """
    Vẽ và lưu biểu đồ so sánh dự báo vs. thực tế trên tập test.

    Args:
        y_true:     Giá thực tập test.
        preds_dict: Dict {model_name: y_pred}.
        horizon:    Horizon hiện tại.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(y_true.index, y_true.values, color="black", linewidth=1.5,
            label="Giá thực tế", zorder=5)

    colors = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261"]
    for (name, pred), color in zip(preds_dict.items(), colors):
        aligned = pred.reindex(y_true.index)
        ax.plot(aligned.index, aligned.values, linewidth=1, alpha=0.8,
                label=name, color=color)

    ax.set_title(f"Dự báo giá vàng SJC — Horizon H={horizon} | Tập kiểm tra 2024–2026",
                 fontsize=13)
    ax.set_ylabel("Giá bán SJC (Triệu VND/lượng)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"predictions_h{horizon}.png", dpi=150)
    plt.close()
    logger.info(f"Đã lưu biểu đồ dự báo H={horizon}")


def plot_shap(
    lgb_model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    horizon: int,
) -> None:
    """
    Tạo và lưu SHAP summary plot cho LightGBM.

    Args:
        lgb_model: Mô hình LightGBM đã huấn luyện.
        X_test:    Tập test.
        horizon:   Horizon hiện tại.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP không khả dụng — bỏ qua.")
        return

    logger.info(f"[SHAP H={horizon}] Đang tính SHAP values...")
    explainer   = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.title(f"SHAP Feature Importance — LightGBM H={horizon}", fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"shap_summary_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[SHAP H={horizon}] Đã lưu SHAP summary.")

    # Bar plot (mean |SHAP|)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Mean |Value| — LightGBM H={horizon}", fontsize=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"shap_bar_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════
# 9. LƯU MODEL
# ═══════════════════════════════════════════════════════
def save_model_artifacts(
    horizon: int,
    scaler: MinMaxScaler,
    rf_model: RandomForestRegressor,
    lgb_model: lgb.LGBMRegressor,
    feature_cols: list[str],
    best_lgb_params: dict[str, Any],
) -> None:
    """
    Lưu tất cả artifacts cần thiết cho một horizon.

    Args:
        horizon:         Horizon h.
        scaler:          MinMaxScaler đã fit.
        rf_model:        Random Forest đã huấn luyện.
        lgb_model:       LightGBM đã huấn luyện.
        feature_cols:    Danh sách tên đặc trưng.
        best_lgb_params: Siêu tham số tốt nhất.
    """
    prefix = MODELS_DIR / f"h{horizon}"

    joblib.dump(scaler,    f"{prefix}_scaler.joblib")
    joblib.dump(rf_model,  f"{prefix}_rf.joblib")
    joblib.dump(lgb_model, f"{prefix}_lgbm.joblib")

    metadata = {
        "horizon":         horizon,
        "train_end_date":  TRAIN_END,
        "feature_list":    feature_cols,
        "best_lgb_params": best_lgb_params,
        "target_col":      TARGET_COL,
        "exog_cols":       EXOG_COLS,
    }
    with open(f"{prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"[H={horizon}] Đã lưu artifacts vào models/")


# ═══════════════════════════════════════════════════════
# 10. PIPELINE TỔNG THỂ
# ═══════════════════════════════════════════════════════
def main() -> None:
    """Điểm nhập chính — chạy toàn bộ pipeline."""

    # ── Bước 1: Tải dữ liệu ──
    df_full = load_and_preprocess(DATA_PATH)

    # ── Bước 2: EDA ──
    adf_results = run_eda(df_full)

    # ── Bước 3–9: Lặp qua từng horizon ──
    all_metrics: list[dict] = []

    for horizon in HORIZONS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  HORIZON H = {horizon}")
        logger.info(f"{'='*60}")

        # -- Feature engineering --
        feat_df      = build_features(df_full, horizon)
        feature_cols = get_feature_columns(feat_df, horizon)

        # -- Train/test split --
        X_train, X_test, y_train_resid, y_test_resid = split_data(feat_df, horizon)

        # -- Scaling (fit only on train) --
        X_train_sc, X_test_sc, scaler = fit_scaler(X_train, X_test)

        # -- Chuỗi target thực cho tập test (để tính metrics) --
        y_test_true = df_full[df_full.index >= TEST_START][TARGET_COL]

        # ── Mô hình 1: Naive ──
        y_naive_true, y_naive_pred = naive_forecast(df_full, horizon)
        metrics_naive = compute_metrics(y_naive_true, y_naive_pred, f"Naive H={horizon}")
        all_metrics.append(metrics_naive)

        # ── Mô hình 2: Prophet (chỉ H=1 để tiết kiệm thời gian) ──
        if horizon == 1:
            try:
                prophet_model, prophet_fc = train_prophet(df_full)
                prophet_pred = (
                    prophet_fc.set_index("ds")["yhat"]
                    .rename_axis("date")
                    .pipe(lambda s: s[s.index >= TEST_START])
                    .reindex(y_test_true.index)
                )
                metrics_prophet = compute_metrics(y_test_true, prophet_pred, "Prophet H=1")
                all_metrics.append(metrics_prophet)
                # Lưu Prophet
                joblib.dump(prophet_model, MODELS_DIR / "h1_prophet.joblib")
            except Exception as exc:
                logger.warning(f"[Prophet] Thất bại: {exc}")

        # ── Mô hình 3: SARIMAX (chỉ H=1) ──
        if horizon == 1:
            try:
                sarimax_fit, sarimax_pred = train_sarimax(df_full)
                sarimax_pred = sarimax_pred.reindex(y_test_true.index)
                metrics_sarimax = compute_metrics(y_test_true, sarimax_pred, "SARIMAX H=1")
                all_metrics.append(metrics_sarimax)
                joblib.dump(sarimax_fit, MODELS_DIR / "h1_sarimax.joblib")
            except Exception as exc:
                logger.warning(f"[SARIMAX] Thất bại: {exc}")

        # ── Mô hình 4: Random Forest (Residual) ──
        rf_model  = train_random_forest(X_train_sc, y_train_resid)
        rf_pred   = predict_with_residual(rf_model, X_test_sc, df_full[TARGET_COL], horizon)
        rf_pred   = rf_pred.reindex(y_test_true.index)
        metrics_rf = compute_metrics(y_test_true, rf_pred, f"RandomForest H={horizon}")
        all_metrics.append(metrics_rf)

        # ── Mô hình 5: LightGBM + Optuna (Residual) ──
        best_params = tune_lightgbm(X_train_sc, y_train_resid, horizon)
        lgb_model   = train_lightgbm(X_train_sc, y_train_resid, best_params)
        lgb_pred    = predict_with_residual(lgb_model, X_test_sc, df_full[TARGET_COL], horizon)
        lgb_pred    = lgb_pred.reindex(y_test_true.index)
        metrics_lgb = compute_metrics(y_test_true, lgb_pred, f"LightGBM H={horizon}")
        all_metrics.append(metrics_lgb)

        # ── Biểu đồ dự báo ──
        preds_dict: dict[str, pd.Series] = {
            f"Naive H={horizon}":       y_naive_pred.reindex(y_test_true.index),
            f"RandomForest H={horizon}": rf_pred,
            f"LightGBM H={horizon}":    lgb_pred,
        }
        if horizon == 1 and "prophet_pred" in dir():
            preds_dict["Prophet H=1"] = prophet_pred
        plot_predictions(y_test_true, preds_dict, horizon)

        # ── SHAP ──
        plot_shap(lgb_model, X_test_sc, horizon)

        # ── Lưu artifacts ──
        save_model_artifacts(
            horizon, scaler, rf_model, lgb_model, feature_cols, best_params
        )

    # ── Bước 10: Bảng kết quả tổng hợp ──
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(IMG_DIR / "evaluation_metrics.csv", index=False)
    logger.info(f"\n=== KẾT QUẢ ĐÁNH GIÁ ===\n{metrics_df.to_string(index=False)}")

    # Biểu đồ so sánh MAE theo mô hình (H=1)
    h1_metrics = metrics_df[metrics_df["Model"].str.contains("H=1")]
    if not h1_metrics.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ["MAE", "RMSE"]):
            bars = ax.barh(h1_metrics["Model"], h1_metrics[metric],
                          color=["#E63946", "#457B9D", "#2A9D8F", "#F4A261"][:len(h1_metrics)])
            ax.set_xlabel(metric)
            ax.set_title(f"So sánh {metric} — Horizon H=1")
            ax.bar_label(bars, fmt="%.3f", padding=3)
            ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "model_comparison_h1.png", dpi=150)
        plt.close()

    logger.info("\n✅ Pipeline hoàn thành! Kiểm tra thư mục models/ và img/")


if __name__ == "__main__":
    main()
