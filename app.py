import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import pandas as pd
import base64
import requests
import numpy as np
from io import BytesIO
import re
import gc
import cv2 

# ================= Optional OCR Backends =================
try:
    from paddleocr import PaddleOCR
    if 'paddle_ocr' not in st.session_state:
        # Khởi tạo PaddleOCR (chỉ chạy 1 lần)
        st.session_state.paddle_ocr = PaddleOCR(use_textline_orientation=True, lang="vi")
        print("[INFO] PaddleOCR initialized successfully.")
    paddle_ocr = st.session_state.paddle_ocr
except Exception as e:
    paddle_ocr = None
    st.warning(f"PaddleOCR init error: {e}\nNếu chưa cài: pip install paddleocr paddlepaddle")
    print("[ERROR] PaddleOCR init failed:", e)

# ================= API Keys (Sử dụng Streamlit Secrets) =================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

except KeyError as e:
    missing_key = e.args[0]
    st.error(f"Lỗi: Không tìm thấy secret '{missing_key}'.")
    st.info(f"Vui lòng vào 'Manage app' -> 'Settings' -> 'Secrets' và thêm '{missing_key}' vào.")
    st.stop()
except FileNotFoundError: # Lỗi này thường không xảy ra trên Cloud
    st.error("Lỗi: Không tìm thấy file .streamlit/secrets.toml.")
    st.info("Vui lòng tạo file .streamlit/secrets.toml và thêm API keys vào đó.")
    st.stop()


# ================= Configure Gemini =================
if not GEMINI_API_KEY:
    st.error("Lỗi: Bạn chưa cung cấp Gemini API Key trong file secrets.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[INFO] Gemini API configured.")
except Exception as e:
    st.error(f"Lỗi khi cấu hình Gemini API: {e}")
    st.stop()

# ================= Gemini Prompt =================
def get_gemini_prompt():
    """Trả về prompt chuẩn cho Gemini."""
    return """
Bạn là một chuyên gia xử lý hóa đơn.
Nhiệm vụ của bạn là trích xuất thông tin chi tiết từ hình ảnh hóa đơn được cung cấp.

Nếu bạn chỉ được đưa TEXT, hãy phân tích TEXT đó.

Hãy trả về một đối tượng JSON:
{
  "items": [
    {
      "ten_hang": "...",
      "don_vi_tinh": "...",
      "so_luong": number/null,
      "don_gia": number/null,
      "thanh_tien": number/null
    }
  ],
  "tong_tien": number/null
}

QUAN TRỌNG:
- Chỉ trả lời JSON.
- Không thêm mô tả.
- Nếu không thấy trường nào -> trả null.
"""

# ================= OCR Backends =================
@st.cache_data(show_spinner=False)
def ocr_google_vision_api_key(image_bytes):
    """Sử dụng Google Vision API để OCR ảnh (đã được cache)."""
    # Sử dụng GOOGLE_API_KEY đã load từ secrets
    if not GOOGLE_API_KEY:
        st.error("Bạn chưa cung cấp Google Vision API Key trong file secrets.")
        return ""
        
    img_base64 = base64.b64encode(image_bytes).decode()
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    payload = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }
    response = requests.post(url, json=payload)
    res_json = response.json()
    text = ""
    try:
        resp0 = res_json.get("responses", [{}])[0]
        if "error" in resp0:
            print("Google Vision API Error:", resp0["error"])
            st.error(f"Google Vision API Error: {resp0['error'].get('message', 'Unknown error')}")
        elif "fullTextAnnotation" in resp0:
            text = resp0["fullTextAnnotation"]["text"]
    except Exception as e:
        print("Lỗi parse Google Vision response:", e)
    print(f"[OCR Google Vision] Kết quả OCR:\n{text}\n{'-'*40}")
    return text

@st.cache_data(show_spinner=False)
def ocr_paddle(image_array_bgr):
    """Sử dụng PaddleOCR để OCR ảnh (đã được cache).
    Lưu ý: PaddleOCR yêu cầu ảnh ở định dạng BGR (np.array).
    """
    if paddle_ocr is None:
        raise RuntimeError("PaddleOCR chưa được cài hoặc khởi tạo thất bại.")
    try:
        # PaddleOCR xử lý ảnh BGR
        result = paddle_ocr.predict(image_array_bgr, use_textline_orientation=True)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR predict lỗi: {e}")
    all_text = []
    if not result or result == [None]:
        print("[WARN] PaddleOCR returned no results.")
        return ""
    for page_result in result:
        # Kiểm tra định dạng trả về của PaddleOCR (có thể là list hoặc dict)
        current_texts = []
        if isinstance(page_result, dict):
            # Định dạng mới (dictionary)
            current_texts = page_result.get('rec_texts', [])
        elif isinstance(page_result, list):
             # Định dạng cũ (list of tuples)
            current_texts = [line[0] for line in page_result if isinstance(line, (list, tuple)) and len(line) > 0]

        all_text.extend(current_texts)

    text = "\n".join(all_text)
    print(f"[OCR PaddleOCR] Kết quả OCR:\n{text}\n{'-'*40}")
    return text

# ================= Gemini JSON extractor =================
@st.cache_data(show_spinner=False)
def extract_invoice_json(text_or_image_data):
    """Gửi text (str) hoặc image (bytes) cho Gemini để trích xuất JSON."""
    
    # Sử dụng model hỗ trợ JSON mode (ví dụ: gemini-2.5-flash)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = get_gemini_prompt()
    
    content_to_send = None
    if isinstance(text_or_image_data, str):
        content_to_send = text_or_image_data
    else:
        try:
            image_pil = Image.open(BytesIO(text_or_image_data))
            content_to_send = image_pil
        except Exception as e:
            print(f"[ERROR] Không thể chuyển bytes thành ảnh: {e}")
            return "{}"

    # Yêu cầu Gemini trả về JSON
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )
    
    try:
        response = model.generate_content(
            [prompt, content_to_send],
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini API: {e}")
        print(f"[ERROR] Lỗi gọi Gemini API: {e}")
        return "{}"

# --- THAY ĐỔI: HÀM TIỀN XỬ LÝ ẢNH ĐƯỢC TỐI ƯU HÓA ---
# ================= Image Pre-processing Helpers =================
@st.cache_data(show_spinner=False)
def correct_skew(image_array_rgb):
    """
    Xoay ảnh RGB (np.array) bị nghiêng.
    Phiên bản này được tối ưu hóa, dùng contours thay vì np.where, tốc độ nhanh hơn.
    """
    try:
        # Chuyển sang ảnh xám
        image_gray = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2GRAY)
        
        # Nhị phân hóa ảnh và đảo ngược (chữ trắng, nền đen)
        _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # --- TỐI ƯU HÓA: Dùng findContours ---
        # Tìm tất cả các đường viền (contours)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Nối tất cả các điểm từ các contours lại thành một mảng
        if not contours:
            print("[INFO] Không tìm thấy contours, bỏ qua xoay.")
            return image_array_rgb
            
        all_points = np.concatenate([cnt for cnt in contours])
        
        # Tìm hình chữ nhật nhỏ nhất bao quanh TẤT CẢ các điểm
        rect = cv2.minAreaRect(all_points)
        # --- KẾT THÚC TỐI ƯU HÓA ---
        
        angle = rect[-1]
        
        # Chuẩn hóa góc:
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Nếu góc quá nhỏ (ảnh đã thẳng), bỏ qua
        if abs(angle) < 0.1:
            print("[INFO] Ảnh đã thẳng, không xoay.")
            return image_array_rgb

        # Xoay ảnh
        (h, w) = image_array_rgb.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(image_array_rgb, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255)) # Fill viền trắng
        
        print(f"[INFO] Đã xoay ảnh một góc: {angle:.2f} độ")
        return rotated
    except Exception as e:
        print(f"[ERROR] Lỗi khi xoay ảnh: {e}")
        return image_array_rgb # Trả về ảnh gốc nếu lỗi

# --- HÀM improve_contrast ĐÃ BỊ XÓA ---

# ================= Streamlit UI =================
st.set_page_config(page_title="Trích xuất Hóa đơn", layout="wide")
st.title("🧾 Trình trích xuất Thông tin Hóa đơn")
st.write("Tải lên một hoặc nhiều hình ảnh hóa đơn")

col1, col2 = st.columns([2, 3]) 

OCR_METHODS = ["Vision", "Google Vision", "PaddleOCR"] 
selected_ocr = st.sidebar.selectbox("Chọn phương thức OCR/Vision", OCR_METHODS)

# --- ĐÃ XÓA: CÁC TÙY CHỌN TIỀN XỬ LÝ TRONG SIDEBAR ---


with col1:
    uploaded_files = st.file_uploader(
        "Chọn ảnh hóa đơn...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Đã chọn {len(uploaded_files)} tệp.")

        with st.expander(f"Xem {len(uploaded_files)} ảnh đã tải lên (preview)"):
            
            num_cols = 5 # Số cột để hiển thị ảnh
            
            for i, uploaded_file in enumerate(uploaded_files):
                if i % num_cols == 0:
                    cols = st.columns(num_cols)
                
                image = Image.open(uploaded_file)
                
                with cols[i % num_cols]:
                    st.image(
                        image, 
                        caption=f"HĐ {i+1}",
                        use_container_width=True 
                    )

    if st.button("Trích xuất thông tin", type="primary", disabled=not uploaded_files):
        with st.spinner("Đang xử lý..."):
            
            master_items_list = []
            master_raw_responses = []
            master_invoice_totals = [] 

            for i, uploaded_file in enumerate(uploaded_files):
                st.info(f"Đang xử lý Hóa đơn số {i+1} ({uploaded_file.name})...")
                
                try:
                    # --- THAY ĐỔI: LUỒNG XỬ LÝ ẢNH TỰ ĐỘNG ---
                    image_pil = Image.open(uploaded_file)
                    
                    # Chuyển sang np.array (RGB) để xử lý OpenCV
                    img_np_rgb = np.array(image_pil.convert("RGB"))

                    # BƯỚC 1: TIỀN XỬ LÝ (TỰ ĐỘNG VÀ NHANH)
                    # Chỉ tự động xoay ảnh, vì hàm này đã được tối ưu hóa
                    with st.spinner(f"File {uploaded_file.name}: Đang tự động làm thẳng ảnh..."):
                        img_np_rgb = correct_skew(img_np_rgb)
                    
                    # (Đã xóa bước tăng tương phản)
                    
                    # BƯỚC 2: CHUẨN BỊ DỮ LIỆU SAU KHI TIỀN XỬ LÝ
                    
                    # Chuyển ảnh ĐÃ XỬ LÝ về lại PIL
                    image_pil_processed = Image.fromarray(img_np_rgb)

                    # Chuẩn bị image bytes cho Vision và Google OCR (từ ảnh đã xử lý)
                    buffered = BytesIO()
                    save_format = "PNG" # PNG tốt hơn cho OCR sau khi xử lý
                    image_pil_processed.save(buffered, format=save_format)
                    image_bytes_for_ocr = buffered.getvalue()

                    # Chuẩn bị ảnh BGR cho PaddleOCR (từ ảnh đã xử lý)
                    # Chuyển RGB (img_np_rgb) -> BGR (PaddleOCR dùng BGR)
                    img_np_bgr_for_paddle = img_np_rgb[..., ::-1] 
                    # --- KẾT THÚC THAY ĐỔI LUỒNG XỬ LÝ ẢNH ---

                    ocr_text = None # Lưu kết quả OCR trung gian
                    raw_response = None
                    
                    if selected_ocr == "Vision":
                        # Gửi ảnh đã xử lý
                        raw_response = extract_invoice_json(image_bytes_for_ocr)
                    else:
                        if selected_ocr == "Google Vision":
                            # Gửi ảnh đã xử lý
                            ocr_text = ocr_google_vision_api_key(image_bytes_for_ocr)
                        elif selected_ocr == "PaddleOCR":
                            # Gửi ảnh BGR đã xử lý cho Paddle
                            ocr_text = ocr_paddle(img_np_bgr_for_paddle)
                        
                        if not ocr_text:
                            st.warning(f"File {uploaded_file.name}: OCR không trả về kết quả.")
                            raw_response = "{}" 
                        else:
                            raw_response = extract_invoice_json(ocr_text)

                    # Lưu cả kết quả OCR và JSON thô
                    master_raw_responses.append({
                        "hoa_don_so": i + 1,
                        "file": uploaded_file.name,
                        "ocr_text": ocr_text, # Sẽ là None nếu dùng 'Vision'
                        "response": raw_response
                    })

                    try:
                        # Làm sạch JSON (loại bỏ markdown ` ```json `)
                        clean = raw_response.strip()
                        
                        json_start = clean.find('{')
                        json_end = clean.rfind('}')
                        
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            clean = clean[json_start:json_end+1]
                        else:
                            raise json.JSONDecodeError("Không tìm thấy đối tượng JSON hợp lệ.", clean, 0)
                        
                        json_data = json.loads(clean)
                        
                        # --- KIỂM TRA SCHEMA ---
                        if "items" not in json_data or not isinstance(json_data.get("items"), list):
                            st.warning(f"File {uploaded_file.name}: JSON trả về không có 'items' hoặc 'items' không phải là list. Bỏ qua items.")
                            json_data["items"] = [] # Đặt mặc định để code không lỗi
                        
                        if "tong_tien" not in json_data:
                            json_data["tong_tien"] = None # Đặt mặc định
                        # --- HẾT KIỂM TRA SCHEMA ---
                        
                        if json_data:
                            items = json_data.get("items", [])
                            master_items_list.extend(items)
                            
                            tong_tien = json_data.get("tong_tien", None)
                            current_total_numeric = None
                            if tong_tien is not None:
                                try:
                                    # Chuẩn hóa số (loại bỏ dấu '.', thay ',' bằng '.')
                                    if isinstance(tong_tien, str):
                                        tong_tien = tong_tien.replace(".", "").replace(",", ".")
                                    
                                    current_total_numeric = float(tong_tien)
                                except (ValueError, TypeError):
                                    pass # Giữ là None nếu không thể convert
                            
                            master_invoice_totals.append({
                                "id": i + 1,
                                "total_value": current_total_numeric,
                                "file_name": uploaded_file.name
                            })

                    except Exception as e:
                        st.error(f"Lỗi parse JSON cho tệp {uploaded_file.name}: {e}")
                        print(f"[ERROR] JSON parse error: {e}\nRaw response:\n{raw_response}")

                except Exception as e:
                    st.error(f"Lỗi xử lý tệp {uploaded_file.name}: {e}")
                    print(f"[ERROR] Main processing error: {e}")
                
                gc.collect() # Dọn dẹp rác
            
            # Lưu kết quả vào session state để hiển thị ở col2
            st.session_state.aggregated_items = master_items_list
            st.session_state.invoice_totals = master_invoice_totals
            st.session_state.aggregated_raw = master_raw_responses
            
            st.success(f"Đã xử lý xong {len(uploaded_files)} tệp!")
            
            # Tự động refresh lại toàn bộ trang để hiển thị kết quả ở col2
            st.rerun()


# ================= Hiển thị kết quả ở Cột 2 =================
with col2:
    if "invoice_totals" in st.session_state:
        st.subheader("Tổng tiền theo Từng Hóa đơn")
        
        all_totals = st.session_state.invoice_totals
        grand_total = 0
        
        num_invoices = len(all_totals)
        
        if num_invoices > 0:
            
            for data in all_totals:
                total_val = data['total_value']
                
                value_str = "Không có"
                if total_val is not None:
                    # Định dạng tiền tệ Việt Nam
                    val_str_formatted = f"{total_val:,.0f}".replace(",", ".")
                    value_str = f"{val_str_formatted} VNĐ"
                
                st.metric(
                    label=f"Hóa đơn số {data['id']} ({data['file_name']})", 
                    value=value_str
                )
                
                if total_val is not None:
                    grand_total += total_val
        
        st.divider()
        val_str_formatted = f"{grand_total:,.0f}".replace(",", ".")
        st.metric("🎉 TỔNG CỘNG (Tất cả hóa đơn)", f"{val_str_formatted} VNĐ")
        st.divider()


    if "aggregated_items" in st.session_state:
        st.subheader("Chi tiết Mặt hàng (Tổng hợp)")
        items = st.session_state.aggregated_items
        
        if items: 
            df = pd.DataFrame(items)

            # Đảm bảo các cột luôn tồn tại
            cols_to_add = ["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]
            final_cols = [] 
            
            for c in cols_to_add:
                if c not in df.columns:
                    df[c] = None
                final_cols.append(c)
            
            df = df[final_cols] 
            df.columns = ["Tên Hàng", "ĐV Tính", "Số Lượng", "Đơn Giá", "Thành Tiền"]
            
            # --- DÙNG st.data_editor ---
            st.write("Bạn có thể nhấp đúp để sửa lỗi trích xuất:")
            df_edited = st.data_editor(
                df, 
                use_container_width=True,
                num_rows="dynamic", # Cho phép người dùng thêm/xóa hàng
                key="data_editor_results"
            )
            
            # Lưu lại df đã chỉnh sửa để export
            st.session_state.edited_items_df = df_edited
            
            # --- NÚT DOWNLOAD ---
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                # Chuyển DataFrame thành CSV, mã hóa utf-8
                return df_to_convert.to_csv(index=False).encode('utf-8-sig')

            csv_data = convert_df_to_csv(df_edited)
            
            st.download_button(
                label="📥 Tải về CSV (dữ liệu đã sửa)",
                data=csv_data,
                file_name="hoa_don_trich_xuat.csv",
                mime="text/csv",
            )

        else:
            st.warning("Không có mặt hàng nào được tìm thấy.")

    if "aggregated_raw" in st.session_state:
        # --- EXPANDER CHI TIẾT HƠN ---
        with st.expander("Xem dữ liệu thô (OCR text và JSON response)"):
            for raw_data in st.session_state.aggregated_raw:
                st.subheader(f"File: {raw_data['file']} (HĐ số {raw_data['hoa_don_so']})")
                
                # Hiển thị text OCR (nếu có)
                if raw_data['ocr_text']:
                    st.text("Kết quả OCR (đã gửi cho Gemini):")
                    st.text_area(f"OCR_{raw_data['hoa_don_so']}", raw_data['ocr_text'], height=150, disabled=True)
                else:
                    st.info("Chạy ở chế độ 'Vision' (không có bước OCR text trung gian).")

                # Hiển thị JSON response
                st.text("Kết quả JSON (Gemini trả về):")
                st.json(raw_data['response'])
                st.divider()
