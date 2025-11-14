# Đồ án: Trình trích xuất Hóa đơn dùng Gemini

Đây là ứng dụng Streamlit cho phép tải lên ảnh hóa đơn và sử dụng API của Google Gemini (cùng các backend OCR) để trích xuất thông tin chi tiết.

## Cài đặt

1.  **Tạo môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình API Keys:**
    * Tạo một thư mục `.streamlit` ở thư mục gốc.
    * Tạo file `.streamlit/secrets.toml`.
    * Thêm API keys của bạn vào file:
        ```toml
        GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"
        GOOGLE_API_KEY = "YOUR_GOOGLE_VISION_KEY_HERE"
        ```

## Chạy ứng dụng

Chạy lệnh sau trong terminal:

```bash
streamlit run app.py
