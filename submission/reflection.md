# BÁO CÁO THỰC HÀNH MLOPS - LAB 21

**Sinh viên thực hiện:** Nguyễn Mậu Lân  
**MSSV:** 2A202600400  
**Ngày thực hiện:** 07/05/2026

---

## 1. Bộ siêu tham số và Lý do lựa chọn

| Tham số | Giá trị | 
|--- |--- | 
| **n_estimators** | 200 |
| **max_depth** | 10 | 
| **random_state** | 5 | 

Lý do: Có accuracy cao nhất

---

## 2. Khó khăn gặp phải và Cách giải quyết

Trong suốt quá trình thiết lập Pipeline CI/CD và triển khai lên máy ảo (VM), tôi đã gặp và xử lý các vấn đề sau:

### 2.1. Lỗi xác thực SSH (Handshake failed)
* **Mô tả:** GitHub Actions báo lỗi `unable to authenticate` khi cố gắng kết nối tới VM để deploy.
* **Cách giải quyết:** 1. Truy cập trực tiếp vào VM qua SSH Browser trên GCP Console.
    2. Kiểm tra và dán thủ công nội dung Public Key của máy cá nhân vào file `~/.ssh/authorized_keys`.
    3. Cập nhật lại chính xác Private Key vào GitHub Secret `VM_SSH_KEY`.

### 2.2. Lỗi môi trường (KeyError: 'GCS_BUCKET')
* **Mô tả:** Khi chạy server thủ công (`uvicorn`) trên máy local hoặc VM, ứng dụng bị sập do thiếu biến môi trường để kết nối Cloud Storage.
* **Cách giải quyết:** 1. Sử dụng lệnh `$env:GCS_BUCKET="tên_bucket"` trên PowerShell (Local).
    2. Cập nhật mã nguồn sử dụng `os.getenv("GCS_BUCKET", "default_value")` để đảm bảo ứng dụng không bị sập nếu thiếu biến.

### 2.3. Lỗi kết nối (No such host / Invalid URI)
* **Mô tả:** Lỗi xảy ra khi IP External của VM thay đổi sau khi khởi động lại, hoặc do cú pháp gọi lệnh `curl` trên PowerShell không đúng.
* **Cách giải quyết:** 1. Cập nhật IP mới nhất từ GCP Console vào GitHub Secrets (`VM_HOST`).
    2. Sử dụng cú pháp `Invoke-RestMethod` trong PowerShell thay cho `curl` truyền thống để đảm bảo tương thích trên Windows.

---

## 3. Kết quả đạt được

1. **Pipeline tự động:** Toàn bộ quy trình từ Train -> Evaluate -> Upload -> Deploy được kích hoạt tự động mỗi khi có thay đổi về code hoặc dữ liệu (`.dvc`).
2. **Khả năng mở rộng:** Dữ liệu đã được cập nhật thành công từ 2998 lên 5996 mẫu, mô hình đã được huấn luyện lại và tự động deploy bản mới nhất.
3. **API hoạt động:** Endpoint `/predict` phản hồi chính xác kết quả dự đoán chất lượng rượu dựa trên 12 đặc trưng đầu vào.

---