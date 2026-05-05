# 🥇 Dự Báo Giá Vàng SJC

> **Anti-leakage Multi-horizon Time Series Forecasting Pipeline**  
> Dự báo giá vàng SJC (1 ngày / 3 ngày / 7 ngày) sử dụng Residual Learning + LightGBM + Optuna

---

## 📂 Cấu trúc dự án

```
sjc_gold/
├── data/
│   └── gold_dataset_v2.csv       # Dataset đầu vào
├── models/
│   ├── h1_scaler.joblib          # Scaler tập train (H=1)
│   ├── h1_rf.joblib              # Random Forest (H=1)
│   ├── h1_lgbm.joblib            # LightGBM (H=1)
│   ├── h1_metadata.json          # Metadata: feature list, params...
│   └── ...                       # Tương tự cho H=3, H=7
├── img/
│   ├── 01_sjc_price_trend.png
│   ├── 02_correlation_heatmap.png
│   ├── adf_test_results.csv
│   ├── predictions_h1.png
│   ├── shap_summary_h1.png
│   └── evaluation_metrics.csv
├── .github/workflows/
│   └── crawler.yml               # GitHub Actions cron job
├── crawler.py                    # Thu thập dữ liệu tự động
├── pipeline.py                   # Pipeline huấn luyện đầy đủ
├── app.py                        # Ứng dụng Streamlit
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Đặt dữ liệu
Đặt file `gold_dataset_v2.csv` vào thư mục `data/`.

### 3. Chạy pipeline huấn luyện
```bash
python pipeline.py
```
Sẽ tạo các file model trong `models/` và biểu đồ trong `img/`.

### 4. Chạy ứng dụng Streamlit
```bash
streamlit run app.py
```

### 5. Cập nhật dữ liệu thủ công
```bash
python crawler.py
```

---

## 🛡️ Anti-Leakage Checklist

| # | Biện pháp | Trạng thái |
|---|-----------|-----------|
| 1 | Shift tất cả biến ngoại sinh `.shift(1)` trước khi tạo lag/rolling | ✅ |
| 2 | Rolling window tính trên dữ liệu đã shift (không bao giờ chứa t) | ✅ |
| 3 | Scaler **chỉ fit trên train**, transform riêng test | ✅ |
| 4 | `TimeSeriesSplit` — **không shuffle** | ✅ |
| 5 | Multi-horizon target: y[t+h-1] với đặc trưng chỉ từ t-1 | ✅ |
| 6 | Residual learning: e(t) = y(t) − y(t-1) với X tại t-1 | ✅ |
| 7 | Không có thông tin test nào được dùng trong quá trình train | ✅ |

---

## 🤖 Mô hình & chiến lược

### Residual Learning
```
y_hat_final(t) = y_naive(t) + e_hat_ML(t)
y_naive(t)     = y(t - h)          ← Naive forecast
e_hat_ML(t)    = ML.predict(X_{t-1})  ← Học phần dư
```

### Direct Multi-step Forecasting
- Train **mô hình riêng** cho mỗi horizon (H=1, H=3, H=7)
- Target: `y[t + h - 1]`
- Không dùng dự báo đệ quy

### Cross-Validation
- `TimeSeriesSplit(n_splits=3)` — không shuffle
- Optuna tối ưu theo Negative MAE trên validation fold

---

## 📊 Các mô hình được so sánh

| Mô hình | Horizon | Ghi chú |
|---------|---------|---------|
| Naive Baseline | H=1,3,7 | y(t) = y(t-h) |
| Prophet | H=1 | Seasonal decomposition |
| SARIMAX(1,1,1)(1,1,1,7) | H=1 | Thống kê truyền thống |
| Random Forest (Residual) | H=1,3,7 | 200 cây, residual learning |
| LightGBM + Optuna (Residual) | H=1,3,7 | GBDT tối ưu, **khuyến nghị** |

---
