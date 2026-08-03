# Chương 3: Chuẩn bị Dữ liệu

## 3.1 Nguồn dữ liệu

- **Dataset**: Vietnam Weather Dataset
- **Nguồn**: `data/raw/weather-vn-1.csv` ... `weather-vn-5.csv`
- **File**: `data/raw/weather-vn-*.csv`

## 3.2 Mô tả Dataset

| Thuộc tính | Giá trị |
|---|---|
| Số dòng (samples) | 3,815,246 |
| Số cột (features) | 20 |
| Khu vực | Các tỉnh / thành phố tại Việt Nam |

### Các cột dữ liệu:

| Cột | Kiểu dữ liệu | Mô tả |
|---|---|---|
| time | string | Thời gian quan sát |
| province | string | Tỉnh / Thành phố |
| city | string | Quận / Huyện |
| temperature | float64 | Nhiệt độ (°C) |
| temp_min | float64 | Nhiệt độ tối thiểu (°C) |
| temp_max | float64 | Nhiệt độ tối đa (°C) |
| humidity | float64 | Độ ẩm (%) |
| feels_like | float64 | Cảm giác nhiệt (°C) |
| visibility | float64 | Tầm nhìn (km) |
| precipitation | float64 | Lượng mưa (mm) – **Biến mục tiêu** |
| cloudcover | float64 | Độ che phủ mây (%) |
| wind_speed | float64 | Tốc độ gió (km/h) |
| wind_gust | float64 | Gió giật (km/h) |
| wind_direction | float64 | Hướng gió (độ) |
| pressure | float64 | Áp suất khí quyển (hPa) |
| is_day | int64 | Thời gian ban ngày (1/0) |
| weather_code | int64 | Mã thời tiết |
| weather_main | string | Nhóm thời tiết chính |
| weather_description | string | Mô tả chi tiết thời tiết |
| weather_icon | string | Biểu tượng thời tiết |

## 3.3 Các bước Tiền xử lý (Data Preprocessing)

**File**: `src/data/preprocessing.py`

1. **Load CSV**: Đọc và hợp nhất các file `data/raw/weather-vn-*.csv` bằng `pandas.read_csv()`.
2. **Missing Values**: Xử lý 12,103 giá trị thiếu ở cột `visibility` bằng nội suy tuyến tính (interpolation) kết hợp ffill/bfill.
3. **Duplicate Removal**: Xóa 4,333 dòng dữ liệu trùng lặp.
4. **Outlier Detection (IQR)**: Áp dụng clipping theo dải IQR cho các cột số: `temperature`, `humidity`, `pressure`, `visibility`, `cloudcover`, `wind_speed`.
5. **Train/Test Split 80/20**: Chia dữ liệu theo thứ tự thời gian (chronological split).
6. **StandardScaler**: Chuẩn hóa đặc trưng bằng `StandardScaler` fit trên tập train.
7. **Lưu kết quả**:
   - `data/processed/X_train.csv`
   - `data/processed/X_test.csv`
   - `data/processed/y_train.csv`
   - `data/processed/y_test.csv`

## 3.4 Các bước Feature Engineering

**File**: `src/features/feature_engineering.py`

1. **Correlation Matrix**: Tính tương quan giữa các đặc trưng và `precipitation`.
2. **Multicollinearity Check**: Lọc các cặp đặc trưng có tương quan cao ( threshold = 0.8) để tránh đa cộng tuyến.
3. **Feature Selection**: Chọn các đặc trưng khí tượng tối ưu nhất cho mô hình Hồi quy Tuyến tính.

## 3.5 Hình ảnh EDA

- `outputs/figures/correlation_heatmap.png` – Ma trận tương quan
- `outputs/figures/histogram_features.png` – Biểu đồ phân phối
- `outputs/figures/boxplot_features.png` – Biểu đồ hộp phát hiện outliers
