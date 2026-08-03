# Hướng Dẫn Làm Việc Nhóm Với Git & Quy Trình Dự Báo Lượng Mưa (Vietnam Weather)

Dự án Hồi quy Tuyến tính dự báo lượng mưa dựa trên bộ dữ liệu thời tiết Việt Nam (Vietnam Weather Dataset).

---

# Cấu Trúc Nhánh

```text
main
│
└── develop
     │
     ├── feature/preprocessing
     ├── feature/feature-engineering
     ├── feature/linear-regression
     ├── feature/evaluation
     └── feature/visualization
```

---

# Quy Trình Chạy Pipeline Dự Án

Thực hiện chạy toàn bộ pipeline dự án (Tiền xử lý -> Feature Selection -> Huấn luyện Mô hình -> Trực quan hóa):

```bash
python main.py
```

Các bước thực hiện chi tiết:
1. **Tiền xử lý dữ liệu**: `python src/data/preprocessing.py`
2. **Feature Engineering**: `python src/features/feature_engineering.py`
3. **Huấn luyện mô hình & Đánh giá**: `python src/models/train.py`
4. **Trực quan hóa kết quả**: `python src/visualization/plots.py`

---

# Cấu Trúc Dự Án

```text
RainForecast_LinearRegression/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/ (weather-vn-1.csv ... weather-vn-5.csv)
│   └── processed/ (X_train.csv, X_test.csv, y_train.csv, y_test.csv)
├── docs/
│   └── chapter3_summary.md
├── notebooks/
│   └── EDA.ipynb
├── outputs/
│   ├── figures/
│   └── models/
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── linear_regression.py
│   │   ├── sklearn_linear_regression.py
│   │   └── train.py
│   └── visualization/
│       ├── generate_figures.py
│       └── plots.py
└── tests/
    ├── test_linear_regression.py
    ├── test_metrics.py
    └── test_sklearn_linear_regression.py
```

---

# Thành Viên Dự Án

| Thành viên | Công việc |
| ------------ | --------------------------------------- |
| Thành viên 1 | Tiền xử lý dữ liệu, Feature Engineering |
| Thành viên 2 | Huấn luyện mô hình, Đánh giá mô hình |
| Cả nhóm | Kiểm thử, Báo cáo, Hoàn thiện README |
