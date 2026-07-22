# Chương 3: Chuẩn bị Dữ liệu

## 3.1 Nguồn dữ liệu

- **Dataset**: Rainfall Timeseries Data
- **Nguồn**: [Kaggle - poojag718/rainfall-timeseries-data](https://www.kaggle.com/datasets/poojag718/rainfall-timeseries-data)
- **File**: `data/raw/weather.csv`

## 3.2 Mô tả Dataset

| Thuộc tính | Giá trị |
|---|---|
| Số dòng (samples) | 252 |
| Số cột (features) | 7 |
| Thời gian | 2000 – 2020, theo tháng |
| Khu vực | Mumbai, Ấn Độ |

### Các cột dữ liệu:

| Cột | Kiểu dữ liệu | Mô tả |
|---|---|---|
| Year | int64 | Năm |
| Month | int64 | Tháng |
| Day | int64 | Ngày (luôn = 1, dữ liệu theo tháng) |
| Specific Humidity | float64 | Độ ẩm đặc trưng (g/kg) |
| Relative Humidity | float64 | Độ ẩm tương đối (%) |
| Temperature | float64 | Nhiệt độ (°C) |
| Precipitation | float64 | Lượng mưa (mm) – **Biến mục tiêu** |

## 3.3 Các bước Tiền xử lý (Data Preprocessing)

**File**: `src/data/preprocessing.py`

1. **Load CSV**: Đọc file `data/raw/weather.csv` bằng `pandas.read_csv()`.
2. **Missing Values**: Kiểm tra giá trị thiếu → Không phát hiện giá trị thiếu (0 missing). Nếu có, sử dụng linear interpolation + ffill/bfill.
3. **Duplicate Removal**: Kiểm tra dòng trùng lặp → Không phát hiện dòng trùng (0 duplicates).
4. **Outlier Detection (IQR)**: Áp dụng phương pháp IQR cho các cột số:
   - `Specific Humidity`: 0 outliers
   - `Relative Humidity`: 0 outliers
   - `Temperature`: 0 outliers
   - `Precipitation`: **16 outliers** → Capped (làm mịn) tại giới hạn IQR
5. **Train/Test Split 80/20**: Chia dữ liệu theo thứ tự thời gian (chronological split):
   - Train: 201 samples
   - Test: 51 samples
6. **StandardScaler**: Chuẩn hóa các features bằng `StandardScaler`, fit trên tập train để tránh data leakage.
7. **Lưu kết quả**:
   - `data/processed/X_train.csv`
   - `data/processed/X_test.csv`
   - `data/processed/y_train.csv`
   - `data/processed/y_test.csv`

## 3.4 Các bước Feature Engineering

**File**: `src/features/feature_engineering.py`

1. **Correlation Matrix**: Tính ma trận tương quan Pearson giữa tất cả features với biến mục tiêu `Precipitation`.
2. **Feature Selection**: Loại bỏ features có |correlation| < 0.1 với biến mục tiêu.
   - `Temperature` bị loại (correlation = 0.016).
3. **Multicollinearity Check**: Kiểm tra đa cộng tuyến giữa các features được chọn (threshold = 0.95).
   - `Relative Humidity` và `Specific Humidity` có tương quan cao (0.916) nhưng dưới ngưỡng 0.95.
   - Giữ lại cả hai để đảm bảo mô hình hồi quy đa biến (Multiple Linear Regression).

## 3.5 Features cuối cùng được chọn

| Feature | Correlation với Precipitation | Vai trò |
|---|---|---|
| **Relative Humidity** | 0.7652 | Biến độc lập |
| **Specific Humidity** | 0.7431 | Biến độc lập |
| **Precipitation** | — | Biến mục tiêu (target) |

- `Temperature` bị loại do tương quan quá thấp (0.016).
- `Year`, `Month`, `Day` không được sử dụng làm features dự đoán (chỉ dùng để sắp xếp thời gian).

## 3.6 Hình ảnh EDA

- `outputs/figures/correlation_heatmap.png` – Ma trận tương quan
- `outputs/figures/histogram_features.png` – Biểu đồ phân phối
- `outputs/figures/boxplot_features.png` – Biểu đồ hộp phát hiện outliers
