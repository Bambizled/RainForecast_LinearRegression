# Hướng Dẫn Làm Việc Nhóm Với Git

Dự án sử dụng mô hình Git Flow đơn giản nhằm giúp các thành viên làm việc song song, hạn chế xung đột mã nguồn và đảm bảo chất lượng dự án.

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

## Ý Nghĩa Các Nhánh

| Nhánh     | Chức năng                                            |
| --------- | ---------------------------------------------------- |
| main      | Chứa phiên bản ổn định cuối cùng của dự án           |
| develop   | Nhánh tổng hợp để kiểm tra và tích hợp các chức năng |
| feature/* | Nhánh phát triển từng chức năng riêng biệt           |

⚠️ Không được code trực tiếp trên nhánh `main`.

---

# Hướng Dẫn Clone Dự Án

## Bước 1: Clone Repository

```bash
git clone https://github.com/Bambizled/RainForecast_LinearRegression.git
```

## Bước 2: Di chuyển vào thư mục dự án

```bash
cd RainForecast_LinearRegression
```

## Bước 3: Kiểm tra trạng thái Git

```bash
git status
```

Nếu xuất hiện:

```text
On branch main
nothing to commit, working tree clean
```

thì clone thành công.

---

# Xem Trạng Thái Dự Án

Kiểm tra các file đã sửa, đã xóa hoặc chưa được theo dõi:

```bash
git status
```

Ví dụ:

```text
modified: src/models/train.py
untracked: data/raw/weather.csv
```

---

# Xem Danh Sách Nhánh

Xem các nhánh trên máy:

```bash
git branch
```

Xem các nhánh trên GitHub:

```bash
git branch -r
```

Xem tất cả:

```bash
git branch -a
```

---

# Chuyển Nhánh

Chuyển sang nhánh develop:

```bash
git checkout develop
```

Chuyển sang nhánh chức năng:

```bash
git checkout feature/preprocessing
```

---

# Tạo Nhánh Chức Năng Mới

Luôn tạo nhánh mới từ develop.

## Cập nhật develop

```bash
git checkout develop
git pull origin develop
```

## Tạo nhánh mới

```bash
git checkout -b feature/linear-regression
```

## Đẩy nhánh lên GitHub

```bash
git push -u origin feature/linear-regression
```

---

# Thêm File Mới

Sau khi tạo hoặc thêm file:

```bash
git add .
```

Tạo commit:

```bash
git commit -m "Thêm chức năng mới"
```

Đẩy lên GitHub:

```bash
git push
```

---

# Sửa File Và Cập Nhật

Sau khi chỉnh sửa code:

```bash
git add .
```

Commit:

```bash
git commit -m "Cập nhật module xử lý dữ liệu"
```

Push:

```bash
git push
```

---

# Xóa File

Ví dụ xóa file test.txt:

```bash
git rm test.txt
```

Commit:

```bash
git commit -m "Xóa file test"
```

Push:

```bash
git push
```

---

# Cập Nhật Code Mới Nhất Từ GitHub

Trước khi bắt đầu làm việc mỗi ngày, luôn cập nhật code mới nhất.

## Cập nhật main

```bash
git checkout main
git pull origin main
```

## Cập nhật develop

```bash
git checkout develop
git pull origin develop
```

## Cập nhật nhánh chức năng

```bash
git checkout feature/preprocessing
git pull origin feature/preprocessing
```

---

# Quy Trình Làm Việc Hằng Ngày

## Bước 1: Cập nhật code mới nhất

```bash
git checkout develop
git pull origin develop
```

## Bước 2: Chuyển sang nhánh chức năng

```bash
git checkout feature/preprocessing
```

## Bước 3: Viết code

Thực hiện chức năng được phân công.

## Bước 4: Lưu thay đổi

```bash
git add .
git commit -m "Hoàn thành xử lý dữ liệu"
git push
```

## Bước 5: Tạo Pull Request

```text
feature/preprocessing
          ↓
       develop
```

Sau khi được kiểm tra và chấp nhận thì mới merge.

---

# Quy Trình Merge

## Merge từ Nhánh Chức Năng vào Develop

```text
feature/preprocessing
          ↓
       develop
```

Các chức năng phải được kiểm tra trước khi merge.

---

## Merge từ Develop vào Main

```text
develop
   ↓
 main
```

Chỉ merge khi:

* Hoàn thành chức năng
* Không còn lỗi
* Đã kiểm thử đầy đủ

---

# Các Lệnh Git Thường Dùng

## Kiểm tra trạng thái

```bash
git status
```

## Xem lịch sử commit

```bash
git log --oneline
```

## Xem remote

```bash
git remote -v
```

## Xem nhánh hiện tại

```bash
git branch
```

## Tải thông tin mới nhất từ GitHub

```bash
git fetch
```

## Hủy thay đổi chưa commit

```bash
git restore .
```

## Xóa nhánh trên máy

```bash
git branch -d feature/old-feature
```

## Xóa nhánh trên GitHub

```bash
git push origin --delete feature/old-feature
```

---

# Quy Tắc Làm Việc Nhóm

1. Không push trực tiếp lên main.
2. Luôn pull code mới nhất trước khi làm việc.
3. Mỗi chức năng phải có một nhánh riêng.
4. Viết commit message rõ ràng.
5. Luôn tạo Pull Request trước khi merge.
6. Kiểm tra lỗi trước khi merge vào develop.
7. Chỉ merge vào main khi đã kiểm thử hoàn chỉnh.
8. Không tự ý xóa file hoặc sửa phần việc của thành viên khác nếu chưa trao đổi.

---

# Thành Viên Dự Án

| Thành viên   | Công việc                               |
| ------------ | --------------------------------------- |
| Thành viên 1 | Tiền xử lý dữ liệu, Feature Engineering |
| Thành viên 2 | Huấn luyện mô hình, Đánh giá mô hình    |
| Cả nhóm      | Kiểm thử, Báo cáo, Hoàn thiện README    |
