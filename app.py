import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageOps
import json
import pandas as pd
import base64
import requests
import numpy as np
from io import BytesIO
import cv2
import gc
from pymongo import MongoClient
from datetime import datetime
import hashlib
import traceback
import difflib

# ================= DICTIONARY =================
KNOWN_UNITS = [
    "m3", "tấm", "kg", "m", "cái", "viên", "tuýp", "bao", "m2", "bộ",
    "ngày", "lít", "hộp", "thùng"
]

KNOWN_ITEM_NAMES = [
    "Đá 0x4",
    "Ván ép",
    "Keo dán đá granite",
    "Gạch terrazzo",
    "Ống nhựa PVC thoát nước",
    "Thép cuộn 6mm",
    "Công tắc 1 chiều",
    "Ống PVC cấp nước",
    "Thép cây Ø10",
    "Gạch đặc 220x105x65",
    "Keo silicone",
    "Gạch ốp tường 20x30",
    "Keo PU",
    "Đá 1x2",
    "Keo dán gạch",
    "Kính 8mm",
    "Vữa trát",
    "Cửa gỗ công nghiệp",
    "Ống đồng",
    "Ốc vít",
    "Cút nối PVC",
    "Đèn LED âm trần",
    "Vữa xây",
    "Giẻ lau",
    "Nẹp nhôm",
    "Tấm xi măng Cemboard",
    "Ống ruột gà",
    "Vòi lavabo",
    "Ngói lợp",
    "Giấy nhám",
    "Gạch men lát nền 30x30",
    "Men gạch",
    "Đinh 5cm",
    "Chổi quét sơn",
    "Gạch tuynel",
    "Lavabo",
    "Đai treo ống",
    "Chốt cửa, bản lề",
    "Bồn cầu",
    "Bóng đèn",
    "Gạch block",
    "Tôn lạnh",
    "Xi măng PCB40",
    "Bột tự san phẳng",
    "Màng chống thấm",
    "Sơn nội thất",
    "Rulo sơn",
    "Sika chống thấm",
    "Gỗ xẻ (tấm)",
    "Sản phẩm vệ sinh (xi phông)",
    "Sơn ngoại thất",
    "Bulong",
    "Cát vàng",
    "Tấm cách nhiệt",
    "Phụ gia bê tông",
    "Dây điện CVV 2x1.5",
    "Bột bả",
    "Tô vữa",
    "Tôn lạnh",
    "Van khóa 21",
    "Lưới thép hàn",
    "Gạch men lát nền 30x30",
    "Bột tự san phẳng",
    "Bột bả",
    "Chất làm dẻo bê tông",
    "Máy trộn cầm tay (thuê)",
    "Sơn lót",
    "Cửa nhôm",
    "Sika chống thấm",
    "Màng chống thấm",
    "Phụ gia bê tông"
]

# ================= VIETNAMESE OCR POSTPROCESSING HELPERS =================
def _normalize_basic_vi(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = " ".join(s.split())
    simple_repls = {
        "0xO": "0x0",
        "OXA": "0x4",
        "her": "kg",
        "ka": "kg",
        "lo": "kg",
        "jag": "kg",
        "kỳ": "cái",
        "ing": "kg",
        "sag": "kg",
        "tam": "tấm",
        "Ø": "d",
        "Kẹo": "Keo",
        "eaú": "cái",
        "kú": "cái",
        "urên": "viên",
        "ven": "viên",
        "Then": "Thép",
        "cas": "cái",
        "car": "cái",
        "ca": "cái",
        "có": "cái",
        "cro": "cái",
        "twyp": "tuýp",
        "túyp": "tuýp",
        "tuyn": "tuýp",
        "Xing": "Xẻng",
    }
    for k, v in simple_repls.items():
        s = s.replace(k, v)
    return s.strip()

def _best_fuzzy_match(token: str, candidates, cutoff=0.8):
    if not token:
        return token
    matches = difflib.get_close_matches(token, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else token

def postprocess_unit_vi(unit_text: str) -> str:
    if not unit_text:
        return unit_text
    s = _normalize_basic_vi(unit_text).lower()
    normalized_candidates = [u.lower() for u in KNOWN_UNITS]
    corrected = _best_fuzzy_match(s, normalized_candidates, cutoff=0.6)
    return corrected

def postprocess_item_name_vi(name_text: str) -> str:
    if not name_text:
        return name_text
    s = _normalize_basic_vi(name_text)
    corrected = _best_fuzzy_match(s, KNOWN_ITEM_NAMES, cutoff=0.5)
    return corrected

def postprocess_invoice_items_vi(items: list) -> list:
    if not isinstance(items, list):
        return items
    fixed = []
    for it in items:
        if not isinstance(it, dict):
            fixed.append(it)
            continue
        new_item = dict(it)
        new_item["ten_hang"] = postprocess_item_name_vi(it.get("ten_hang", ""))
        new_item["don_vi_tinh"] = postprocess_unit_vi(it.get("don_vi_tinh", ""))
        fixed.append(new_item)
    return fixed

# ================= API Keys =================
GEMINI_API_KEY = "AIzaSyDya5MCCXN6QgsYBWeW93ZaO_5CkuXogTk"
GOOGLE_VISION_API_KEY = "AIzaSyC5aageWFdZSuOQc21jU9YUcgGgV2V-qRA"

# ================= Initialize Gemini =================
if 'gemini_configured' not in st.session_state:
    if not GEMINI_API_KEY:
        st.error("Bạn chưa cung cấp Gemini API Key.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            st.session_state.gemini_configured = True
            print("[INFO] Gemini API configured.")
        except Exception as e:
            st.error(f"Lỗi khi cấu hình Gemini API: {e}")
            st.session_state.gemini_configured = False

# ================= Ensure session_state defaults =================
for key in ["aggregated_items", "invoice_totals", "aggregated_raw", "extraction_done", "edited_data_for_db"]:
    if key not in st.session_state:
        if key == "extraction_done":
            st.session_state[key] = False
        elif key == "edited_data_for_db":
            st.session_state[key] = {}
        else:
            st.session_state[key] = []

for key in ["main_items", "main_totals", "main_raw", "main_done", "main_uploaded_files_bytes", "main_db_edited"]:
    if key not in st.session_state:
        if key == "main_done":
            st.session_state[key] = False
        elif key == "main_uploaded_files_bytes":
            st.session_state[key] = {}
        elif key == "main_db_edited":
            st.session_state[key] = {}
        else:
            st.session_state[key] = []

for key in ["paddle_ocr", "vietocr_detector", "easyocr_reader"]:
    if key not in st.session_state:
        st.session_state[key] = None

def hash_image_data(image_data):
    if isinstance(image_data, bytes):
        return hashlib.md5(image_data).hexdigest()
    elif isinstance(image_data, str):
        return hashlib.md5(image_data.encode()).hexdigest()
    else:
        return hashlib.md5(str(image_data).encode()).hexdigest()

# ================= Gemini Prompt =================
def get_gemini_prompt():
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
- Không thêm thông tin không xuất hiện trong text.
- Nếu không thấy trường nào -> trả null.
- Nếu là toàn là số, không thể là đơn vị.

Return only valid JSON — no extra text before/after.
"""

# ================= Vision (Google) helpers =================
@st.cache_resource
def get_vision_client():
    return GOOGLE_VISION_API_KEY

@st.cache_data(show_spinner=False)
def ocr_google_vision_api_key(image_bytes):
    api_key = get_vision_client()
    if not api_key:
        return "", []

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    body = {
        "requests": [
            {
                "image": {
                    "content": base64.b64encode(image_bytes).decode("utf-8")
                },
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {
                    "languageHints": ["vi", "en"]
                }
            }
        ]
    }

    try:
        response = requests.post(url, json=body, timeout=60)
        data = response.json()

        if "error" in data:
            return "", []

        resp = data.get("responses", [{}])[0]
        fta = resp.get("fullTextAnnotation", {})

        extracted_text = fta.get("text", "") if fta else ""

        boxes = []
        text_annotations = resp.get("textAnnotations", [])
        for t in text_annotations[1:]:
            poly = t.get("boundingPoly", {}).get("vertices", [])
            if len(poly) == 4:
                box = [
                    (poly[0].get("x", 0), poly[0].get("y", 0)),
                    (poly[1].get("x", 0), poly[1].get("y", 0)),
                    (poly[2].get("x", 0), poly[2].get("y", 0)),
                    (poly[3].get("x", 0), poly[3].get("y", 0))
                ]
                boxes.append(box)

        return extracted_text, boxes

    except Exception:
        return "", []
    
    except requests.exceptions.Timeout:
        st.error("Google Vision API timeout sau 60 giây")
        return ""
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối Google Vision API: {e}")
        return ""
    except Exception as e:
        st.error(f"Lỗi khi gọi Google Vision API: {e}")
        traceback.print_exc()
        return ""

# ================= MongoDB helper =================
MONGO_URI = "mongodb+srv://kiet410pham_db_user:kiet04102003@cluster0.xcuzaq0.mongodb.net/?appName=Cluster0"

@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"Không thể kết nối MongoDB: {e}")
        return None

def get_collection_for_method(method_name: str, db_name="hoa_don_db"):
    mapping = {
        "Vision": "vision",
        "Google Vision": "google_vision",
        "Paddle": "paddle_vietocr",
        "EasyOCR": "easy_vietocr"
    }
    
    client = get_mongo_client()
    if client is None:
        return None
    
    coll_name = mapping.get(method_name, "other_ocr")
    db = client[db_name]
    return db[coll_name]

def save_extraction_batch(selected_method: str):
    coll = get_collection_for_method(selected_method)
    if coll is None:
        st.error("MongoDB client chưa được cấu hình. Kiểm tra cấu hình kết nối.")
        return False
    
    saved_count = 0
    edited_data = st.session_state.get("edited_data_for_db", {})
    
    for raw in st.session_state.get("aggregated_raw", []):
        try:
            hoa_don_so = raw.get("hoa_don_so")
            file_name = raw.get("file")
            ocr_text = raw.get("ocr_text")
            raw_model_output = raw.get("model_output", "{}")
            
            parsed_items = []
            parsed_total = None
            cleaned = raw.get("response", "{}")
            
            try:
                if isinstance(cleaned, str):
                    clean = cleaned.strip()
                    start = clean.find('{')
                    end = clean.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        clean = clean[start:end+1]
                    parsed = json.loads(clean)
                else:
                    parsed = cleaned
                
                if file_name in edited_data:
                    parsed_items = edited_data[file_name].get("items", [])
                    parsed_total = edited_data[file_name].get("tong_tien", None)
                else:
                    parsed_items = parsed.get("items", []) if isinstance(parsed, dict) else []
                    parsed_total = parsed.get("tong_tien", None) if isinstance(parsed, dict) else None
                
                parsed_items = postprocess_invoice_items_vi(parsed_items)
                
                if parsed_total is not None:
                    if isinstance(parsed_total, str):
                        parsed_total = parsed_total.replace(".", "").replace(",", ".")
                        try:
                            parsed_total = float(parsed_total)
                        except Exception:
                            parsed_total = None
            except Exception:
                parsed_items = []
            
            doc = {
                "hoa_don_so": hoa_don_so,
                "file": file_name,
                "ocr_text": ocr_text,
                "model_output": raw_model_output,
                "json_response": cleaned,
                "items": parsed_items,
                "tong_tien": parsed_total,
                "ocr_method": selected_method,
                "saved_at": datetime.utcnow()
            }
            
            coll.update_one(
                {"file": file_name, "ocr_method": selected_method},
                {"$set": doc},
                upsert=True
            )
            saved_count += 1
        
        except Exception as e:
            print(f"Lỗi lưu 1 hóa đơn: {e}")
            traceback.print_exc()
            continue
    
    st.success(f"Đã lưu {saved_count} bản ghi vào collection '{coll.name}'.")
    return True

def list_documents_for_method(selected_method: str, limit=200):
    coll = get_collection_for_method(selected_method)
    if coll is None:
        return []
    
    docs = list(coll.find().sort("hoa_don_so", 1).limit(limit))
    for d in docs:
        d["_id"] = str(d.get("_id"))
        if isinstance(d.get("saved_at"), datetime):
            d["saved_at"] = d["saved_at"].isoformat()
    
    return docs

# ================= OCR Backend (Paddle / VietOCR / EasyOCR) =================
def ensure_paddle_loaded():
    if 'paddle_ocr' in st.session_state and st.session_state.get('paddle_ocr') is not None:
        return
    
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        st.session_state['paddle_ocr'] = None
        if st.session_state.get('DEBUG', False):
            print("[PADDLE] Import error:", e)
            traceback.print_exc()
        return
    
    constructors = [
        {"use_textline_orientation": True, "lang": "vi"},
        {"lang": "vi", "use_angle_cls": True},
        {"lang": "vi"},
        {}
    ]
    
    inst = None
    for params in constructors:
        try:
            inst = PaddleOCR(**params)
            st.session_state['paddle_ocr'] = inst
            if st.session_state.get('DEBUG', False):
                print(f"[PADDLE] Initialized with params: {params}")
            return
        except Exception as e:
            if st.session_state.get('DEBUG', False):
                print(f"[PADDLE] Init attempt failed with params {params}: {e}")
                traceback.print_exc()
            inst = None
            continue
    
    st.session_state['paddle_ocr'] = None

def ensure_vietocr_loaded():
    if 'vietocr_detector' in st.session_state and st.session_state.get('vietocr_detector') is not None:
        return
    
    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
    except Exception as e:
        st.session_state['vietocr_detector'] = None
        st.warning(f"[VIETOCR] Import error: {e}")
        st.text(traceback.format_exc())
        return
    
    try:
        cfg = Cfg.load_config_from_name('vgg_transformer')
        cfg['device'] = 'cpu'
        cfg['predictor']['beamsearch'] = False
        st.session_state.vietocr_detector = Predictor(cfg)
        print("[INFO] VietOCR predictor initialized successfully.")
    except Exception as e:
        st.session_state.vietocr_detector = None
        st.warning(f"[VIETOCR] VietOCR init error: {e}")
        st.text(traceback.format_exc())

def ensure_paddle_and_viet_loaded():
    ensure_paddle_loaded()
    ensure_vietocr_loaded()

def ensure_easyocr_loaded():
    if 'easyocr_reader' in st.session_state and st.session_state.get('easyocr_reader') is not None:
        return
    
    try:
        import easyocr
        st.session_state.easyocr_reader = easyocr.Reader(
            ['vi'],
            gpu=False,
            verbose=False
        )
        print("[INFO] EasyOCR initialized successfully.")
    except Exception:
        st.session_state.easyocr_reader = None

def ocr_paddle(image_array_bgr):
    ensure_paddle_loaded()
    paddle = st.session_state.get('paddle_ocr')
    if paddle is None:
        return ""
    
    try:
        result = paddle.predict(image_array_bgr, use_textline_orientation=True)
    except Exception:
        return ""
    
    all_text = []
    if not result or result == [None]:
        return ""
    
    for page_result in result:
        if isinstance(page_result, dict):
            all_text.extend(page_result.get('rec_texts', []))
        elif isinstance(page_result, list):
            all_text.extend([line[0] for line in page_result if isinstance(line, (list, tuple)) and len(line) > 0])
    
    return "\n".join(all_text)

def ocr_paddle_vietocr(image_array_rgb):
    ensure_paddle_and_viet_loaded()
    paddle = st.session_state.get('paddle_ocr')
    viet = st.session_state.get('vietocr_detector')
    
    if paddle is None or viet is None:
        return ""
    
    img_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    
    try:
        det_result = paddle.predict(img_bgr, use_textline_orientation=True)
    except Exception:
        return ""
    
    if not det_result or det_result == [None]:
        return ""
    
    raw = det_result[0] if isinstance(det_result, list) else det_result
    img_np_for_crop = image_array_rgb
    
    polys = []
    if isinstance(raw, dict):
        polys = raw.get('dt_polys', [])
        if not polys and 'dt_boxes' in raw:
            boxes = raw.get('dt_boxes', [])
            poly_list = []
            for b in boxes:
                try:
                    x1, y1, x2, y2 = map(int, b[:4])
                    poly_list.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
                except Exception:
                    continue
            polys = poly_list
    else:
        parsed = []
        try:
            for entry in det_result:
                if isinstance(entry, list) and len(entry) > 0:
                    first = entry[0]
                    arr = np.array(first)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        parsed.append(arr)
            if parsed:
                polys = parsed
        except Exception:
            polys = []
    
    if not polys:
        if isinstance(raw, dict):
            recs = raw.get('rec_texts', [])
            return "\n".join(recs) if recs else ""
        return ""
    
    all_texts = []
    h, w = img_np_for_crop.shape[:2]
    
    for poly in polys:
        try:
            poly = np.array(poly)
            if poly.size == 0:
                continue
            
            xs = poly[:, 0]
            ys = poly[:, 1]
            x_min, x_max = max(0, int(xs.min())), min(w, int(xs.max()))
            y_min, y_max = max(0, int(ys.min())), min(h, int(ys.max()))
            
            pad_x = max(1, int((x_max - x_min) * 0.05))
            pad_y = max(1, int((y_max - y_min) * 0.05))
            
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)
            
            crop_np = img_np_for_crop[y_min:y_max, x_min:x_max]
            if crop_np.size == 0:
                continue
            
            crop_pil = Image.fromarray(crop_np)
            try:
                text = viet.predict(crop_pil)
            except Exception:
                continue
            
            if text:
                all_texts.append(text)
        except Exception:
            continue
    
    return "\n".join(all_texts)

def ocr_easyocr(image_array_rgb):
    ensure_easyocr_loaded()
    reader = st.session_state.get('easyocr_reader')
    if reader is None:
        return ""
    
    try:
        res = reader.readtext(image_array_rgb, detail=1, paragraph=False)
        texts = []
        for detection in res:
            if len(detection) >= 2:
                text = detection[1]
                confidence = detection[2] if len(detection) > 2 else 0
                if confidence > 0.3:
                    texts.append(str(text))
        return "\n".join(texts)
    except Exception:
        return ""

def ocr_easyocr_vietocr(image_array_rgb):
    ensure_easyocr_loaded()
    ensure_vietocr_loaded()
    
    reader = st.session_state.get('easyocr_reader')
    viet = st.session_state.get('vietocr_detector')
    
    if reader is None or viet is None:
        return ""
    
    try:
        res = reader.readtext(image_array_rgb, detail=1, paragraph=False)
    except Exception:
        return ""
    
    if not res:
        return ""
    
    all_texts = []
    h, w = image_array_rgb.shape[:2]
    
    for entry in res:
        try:
            if len(entry) < 2:
                continue
            
            bbox = entry[0]
            bbox_arr = np.array(bbox)
            if bbox_arr.ndim != 2 or bbox_arr.shape[1] != 2:
                continue
            
            xs = bbox_arr[:, 0]
            ys = bbox_arr[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            
            pad_x = max(2, int((x_max - x_min) * 0.05))
            pad_y = max(2, int((y_max - y_min) * 0.05))
            
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)
            
            crop_np = image_array_rgb[y_min:y_max, x_min:x_max]
            if crop_np.size == 0 or crop_np.shape[0] < 5 or crop_np.shape[1] < 5:
                continue
            
            crop_pil = Image.fromarray(crop_np)
            try:
                text = viet.predict(crop_pil)
                if text and text.strip():
                    all_texts.append(text.strip())
            except Exception:
                easy_text = entry[1] if len(entry) > 1 else ""
                if easy_text and str(easy_text).strip():
                    all_texts.append(str(easy_text).strip())
                continue
        except Exception:
            continue
    
    return "\n".join(all_texts)

# ================= Gemini JSON extractor =================
@st.cache_data(show_spinner=False, ttl=3600)
def extract_invoice_json(_image_hash, text_or_image_data):
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    prompt = get_gemini_prompt()
    
    content_to_send = text_or_image_data
    if not isinstance(content_to_send, str):
        try:
            content_to_send = Image.open(BytesIO(text_or_image_data))
        except Exception:
            return json.dumps({"error": "Lỗi đọc ảnh"})
    
    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
        
        response = model.generate_content(
            [prompt, content_to_send],
            generation_config=generation_config
        )
        
        if not response.parts:
            feedback = getattr(response, 'prompt_feedback', None)
            error_info = {
                "error": "Model trả về rỗng",
                "feedback": str(feedback) if feedback else "No feedback available"
            }
            return json.dumps(error_info, ensure_ascii=False, indent=2)
        
        return response.text
    
    except Exception:
        traceback.print_exc()
        return json.dumps({"error": "Gemini API call failed"})

def get_invoice_response(text_or_image_data):
    img_hash = hash_image_data(text_or_image_data)
    raw = extract_invoice_json(img_hash, text_or_image_data)
    raw_str = raw if isinstance(raw, str) else str(raw)
    
    cleaned = "{}"
    try:
        s = raw_str.strip()
        a = s.find('{')
        b = s.rfind('}')
        if a != -1 and b != -1 and b > a:
            cleaned = s[a:b+1]
            _ = json.loads(cleaned)
        else:
            cleaned = "{}"
    except Exception:
        cleaned = "{}"
    
    return raw_str, cleaned

# ================= Accuracy helpers =================
def normalize_val(val):
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s.replace(',', '').replace('.', '').replace('-', '').isdigit():
        return s.replace(',', '').replace('.', '')
    return s.rstrip('.')

def normalize_dataframe_types(df):
    df_copy = df.copy()
    numeric_cols = ["so_luong", "don_gia", "thanh_tien", "Tổng Tiền HĐ", "Số Lượng", "Đơn Giá", "Thành Tiền"]
    
    for col in numeric_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: str(x).replace(".", "").replace(",", "") if pd.notna(x) and x not in [None, ''] else None
            )
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    return df_copy

def fetch_all_docs(method_name):
    coll = get_collection_for_method(method_name)
    if coll is None:
        return []
    return list(coll.find({}))

def calculate_accuracy_stats(ground_truth_docs, target_method_name):
    target_docs = fetch_all_docs(target_method_name)
    target_map = {d.get('file'): d for d in target_docs if d.get('file')}
    
    item_fields = ["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]
    stats = {
        "tong_tien": {"correct": 0, "total": 0},
        "items_found": {"correct": 0, "total": 0}
    }
    for f in item_fields:
        stats[f] = {"correct": 0, "total": 0}
    
    files_checked = 0
    
    for gt_doc in ground_truth_docs:
        fname = gt_doc.get('file')
        if not fname:
            continue
        
        pred_doc = target_map.get(fname)
        if not pred_doc:
            if gt_doc.get("tong_tien") is not None:
                stats["tong_tien"]["total"] += 1
            for item in gt_doc.get("items", []):
                for f in item_fields:
                    if item.get(f) is not None:
                        stats[f]["total"] += 1
            continue
        
        files_checked += 1
        
        gt_total = gt_doc.get("tong_tien")
        if gt_total is not None:
            stats["tong_tien"]["total"] += 1
            pred_total = pred_doc.get("tong_tien")
            if normalize_val(gt_total) == normalize_val(pred_total):
                stats["tong_tien"]["correct"] += 1
            elif isinstance(gt_total, (int, float)) and isinstance(pred_total, (int, float)):
                if abs(gt_total - pred_total) < 100:
                    stats["tong_tien"]["correct"] += 1
        
        gt_items = gt_doc.get("items", [])
        pred_items = pred_doc.get("items", [])
        
        for i, gt_item in enumerate(gt_items):
            pred_item = pred_items[i] if i < len(pred_items) else {}
            
            for field in item_fields:
                gt_val = gt_item.get(field)
                if gt_val is not None and str(gt_val).strip() != "":
                    stats[field]["total"] += 1
                    pred_val = pred_item.get(field)
                    if normalize_val(gt_val) == normalize_val(pred_val):
                        stats[field]["correct"] += 1
    
    return stats, files_checked

def draw_boxes_on_image(img_bytes, boxes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(img_np, [pts], True, (255, 0, 0), 2)

    return img_np


# ================= Streamlit UI =================
st.set_page_config(page_title="Trích xuất Hóa đơn", layout="wide")
st.title("🧾 Trình trích xuất Thông tin Hóa đơn")

tab_main, tab_advanced = st.tabs(["Main UI", "Advanced / Database"])

OCR_METHODS = ["Vision", "Google Vision", "Paddle", "EasyOCR"]

# ========== MAIN UI ==========
with tab_main:
    uploaded_files_main = st.file_uploader(
        "Chọn ảnh hóa đơn...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="main_uploader"
    )
    
    if uploaded_files_main:
        st.write(f"Đã chọn {len(uploaded_files_main)} tệp.")
        
        with st.expander("Xem ảnh đã tải lên"):
            num_cols = 4
            for i, uploaded_file in enumerate(uploaded_files_main):
                if i % num_cols == 0:
                    cols = st.columns(num_cols)
                
                uploaded_file.seek(0)
                img = Image.open(uploaded_file)
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                
                w, h = img.size
                if w > h * 1.2:
                    img = img.rotate(90, expand=True)
                
                with cols[i % num_cols]:
                    st.image(img, width='stretch', caption=f"HĐ {i+1} - {uploaded_file.name}")
                
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.session_state.main_uploaded_files_bytes[uploaded_file.name] = buf.getvalue()
    
    extract_clicked_main = st.button("Trích xuất thông tin", type="primary", disabled=not uploaded_files_main)
    
    if extract_clicked_main and uploaded_files_main:
        st.session_state.main_done = False
        main_items_list = []
        main_raw_responses = []
        main_invoice_totals = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files_main):
            status_text.info(f"Đang xử lý Hóa đơn số {i+1}/{len(uploaded_files_main)}: {uploaded_file.name}")
            progress_bar.progress((i) / len(uploaded_files_main))
            
            image_pil = None
            img_np_rgb = None
            image_bytes_for_ocr = None
            
            try:
                uploaded_file.seek(0)
                original_bytes = uploaded_file.read()
                image_pil = Image.open(BytesIO(original_bytes))
                
                try:
                    image_pil = ImageOps.exif_transpose(image_pil)
                except Exception:
                    pass
                
                w, h = image_pil.size
                if w > h * 1.2:
                    image_pil = image_pil.rotate(90, expand=True)
                
                if image_pil.mode == 'RGBA':
                    background = Image.new('RGB', image_pil.size, (255, 255, 255))
                    background.paste(image_pil, mask=image_pil.split()[3])
                    img_np_rgb = np.array(background)
                elif image_pil.mode != 'RGB':
                    img_np_rgb = np.array(image_pil.convert("RGB"))
                else:
                    img_np_rgb = np.array(image_pil)
                
                image_bytes_for_ocr = BytesIO()
                Image.fromarray(img_np_rgb).save(image_bytes_for_ocr, format="PNG")
                image_bytes_for_ocr = image_bytes_for_ocr.getvalue()
                
                ocr_text, ocr_boxes = ocr_google_vision_api_key(image_bytes_for_ocr)
                
                if not ocr_text or not ocr_text.strip():
                    st.warning(f"⚠️ Không trích xuất được text từ {uploaded_file.name}")
                    model_raw_output = json.dumps({"error": "No OCR text extracted"})
                    model_clean_json = "{}"
                else:
                    model_raw_output, model_clean_json = get_invoice_response(ocr_text)
                
                main_raw_responses.append({
                    "hoa_don_so": i+1,
                    "file": uploaded_file.name,
                    "ocr_text": ocr_text,
                    "bbox": ocr_boxes,
                    "model_output": model_raw_output,
                    "response": model_clean_json
                })

                
                clean = model_clean_json.strip() if isinstance(model_clean_json, str) else "{}"
                json_start, json_end = clean.find('{'), clean.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    clean = clean[json_start:json_end+1]
                else:
                    clean = "{}"
                
                try:
                    json_data = json.loads(clean)
                except json.JSONDecodeError:
                    json_data = {"items": [], "tong_tien": None}
                
                if "items" not in json_data or not isinstance(json_data.get("items"), list):
                    json_data["items"] = []
                if "tong_tien" not in json_data:
                    json_data["tong_tien"] = None
                
                items = json_data.get("items", [])
                items = postprocess_invoice_items_vi(items)
                
                for it in items:
                    it["file_name"] = uploaded_file.name
                    it["hoa_don_id"] = i + 1
                
                main_items_list.extend(items)
                
                tong_tien = json_data.get("tong_tien", None)
                current_total_numeric = None
                if tong_tien is not None:
                    try:
                        if isinstance(tong_tien, str):
                            tong_tien = tong_tien.replace(".", "")
                        current_total_numeric = float(tong_tien)
                    except Exception:
                        pass
                
                main_invoice_totals.append({
                    "id": i+1,
                    "total_value": current_total_numeric,
                    "file_name": uploaded_file.name
                })
            
            except Exception:
                traceback.print_exc()
            
            finally:
                del image_pil, img_np_rgb, image_bytes_for_ocr
                gc.collect()
        
        progress_bar.progress(1.0)
        status_text.success(f"✅ Đã xử lý xong {len(uploaded_files_main)} tệp!")
        
        st.session_state.main_items = main_items_list
        st.session_state.main_totals = main_invoice_totals
        st.session_state.main_raw = main_raw_responses
        st.session_state.main_done = True
    
    if st.session_state.get("main_done", False) and st.session_state.main_totals:
        st.divider()
        if st.session_state.main_totals:
            grand_total_main = sum(d["total_value"] for d in st.session_state.main_totals if d["total_value"] is not None)
            col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
            with col_center2:
                st.markdown("<h2 style='text-align: center;'>Tổng cộng (tất cả hóa đơn)</h2>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; color: #00c853;'>{grand_total_main:,.0f} VNĐ</h1>".replace(",", "."), unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### Tổng tiền từng hóa đơn")
        
        col_left, col_right = st.columns([3, 4])
        
        with col_left:
            invoice_options_main = [
                f"Hóa đơn {d['id']} – {d['file_name']}" for d in st.session_state.main_totals
            ]
            
            selected_invoice_main = st.selectbox(
                "Chọn hóa đơn để xem",
                invoice_options_main,
                key="main_invoice_select"
            )
            
            inv_id_main = None
            file_name_main = None
            inv_data_main = None
            
            if selected_invoice_main:
                inv_id_main = int(selected_invoice_main.split(" ")[2])
                inv_data_main = next(
                    (d for d in st.session_state.main_totals if d["id"] == inv_id_main),
                    None
                )
                
                if inv_data_main:
                    file_name_main = inv_data_main["file_name"]
                    img_bytes = st.session_state.main_uploaded_files_bytes.get(file_name_main)
                    
                    if img_bytes:
                        raw_entry = next(
                            (r for r in st.session_state.main_raw if r["file"] == file_name_main),
                            None
                        )

                        bbox = []
                        if raw_entry and "bbox" in raw_entry:
                            bbox = raw_entry["bbox"]

                        show_bbox = st.checkbox(
                            "Hiện bounding box",
                            key=f"bbox_toggle_{inv_id_main}"
                        )

                        if show_bbox and bbox:
                            img_with_bbox = draw_boxes_on_image(img_bytes, bbox)
                            st.image(
                                img_with_bbox,
                                width='stretch',
                                caption=f"Hóa đơn {inv_id_main} – {file_name_main} (BBox)"
                            )
                        else:
                            st.image(
                                img_bytes,
                                width='stretch',
                                caption=f"Hóa đơn {inv_id_main} – {file_name_main}"
                            )

        
        with col_right:
            if st.session_state.main_items and inv_id_main is not None:
                items_for_invoice = [
                    it for it in st.session_state.main_items
                    if it.get("hoa_don_id") == inv_id_main
                ]
                
                st.subheader("Chi tiết mặt hàng")
                
                if items_for_invoice:
                    df_inv = pd.DataFrame(items_for_invoice)
                    
                    for c in ["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]:
                        if c not in df_inv.columns:
                            df_inv[c] = None
                    
                    df_inv = df_inv[["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]]
                    df_inv.columns = ["Tên Hàng", "ĐV Tính", "Số Lượng", "Đơn Giá", "Thành Tiền"]
                    df_inv = normalize_dataframe_types(df_inv)
                    
                    edited_df_inv = st.data_editor(
                        df_inv,
                        width='stretch',
                        num_rows="dynamic",
                        key=f"main_editor_{inv_id_main}"
                    )
                    
                    if file_name_main:
                        items_edited = []
                        for _, row in edited_df_inv.iterrows():
                            items_edited.append({
                                "ten_hang": row["Tên Hàng"],
                                "don_vi_tinh": row["ĐV Tính"],
                                "so_luong": row["Số Lượng"],
                                "don_gia": row["Đơn Giá"],
                                "thanh_tien": row["Thành Tiền"]
                            })
                        
                        st.session_state.main_db_edited[file_name_main] = {
                            "items": items_edited,
                            "tong_tien": inv_data_main["total_value"]
                        }
                    
                    st.divider()
                    total_val = inv_data_main["total_value"]
                    if total_val is not None:
                        value_str = f"{total_val:,.0f}".replace(",", ".") + " VNĐ"
                    else:
                        value_str = "Không có"
                    st.markdown(f"**Tổng tiền hóa đơn này:** {value_str}")

                    col_save_left, col_save_right = st.columns([3, 1])
                    with col_save_right:
                        def save_main_to_db():
                            coll = get_collection_for_method("Google Vision")
                            if coll is None:
                                st.error("MongoDB client chưa được cấu hình.")
                                return False
                            
                            saved_count = 0
                            for raw in st.session_state.main_raw:
                                try:
                                    hoa_don_so = raw.get("hoa_don_so")
                                    file_name = raw.get("file")
                                    ocr_text = raw.get("ocr_text")
                                    raw_model_output = raw.get("model_output", "{}")
                                    cleaned = raw.get("response", "{}")
                                    
                                    parsed_items = []
                                    parsed_total = None
                                    
                                    if file_name in st.session_state.main_db_edited:
                                        parsed_items = st.session_state.main_db_edited[file_name].get("items", [])
                                        parsed_total = st.session_state.main_db_edited[file_name].get("tong_tien", None)
                                    else:
                                        inv_data = next((d for d in st.session_state.main_totals if d["file_name"] == file_name), None)
                                        if inv_data:
                                            parsed_total = inv_data.get("total_value")
                                        
                                        file_items = [it for it in st.session_state.main_items if it.get("file_name") == file_name]
                                        for it in file_items:
                                            parsed_items.append({
                                                "ten_hang": it.get("ten_hang"),
                                                "don_vi_tinh": it.get("don_vi_tinh"),
                                                "so_luong": it.get("so_luong"),
                                                "don_gia": it.get("don_gia"),
                                                "thanh_tien": it.get("thanh_tien")
                                            })
                                    
                                    doc = {
                                        "hoa_don_so": hoa_don_so,
                                        "file": file_name,
                                        "ocr_text": ocr_text,
                                        "model_output": raw_model_output,
                                        "json_response": cleaned,
                                        "items": parsed_items,
                                        "tong_tien": parsed_total,
                                        "ocr_method": "Google Vision",
                                        "saved_at": datetime.utcnow()
                                    }
                                    
                                    coll.update_one(
                                        {"file": file_name, "ocr_method": "Google Vision"},
                                        {"$set": doc},
                                        upsert=True
                                    )
                                    saved_count += 1
                                
                                except Exception as e:
                                    print(f"Lỗi lưu 1 hóa đơn (Main UI): {e}")
                                    traceback.print_exc()
                                    continue
                            
                            st.success(f"Đã lưu {saved_count} bản ghi vào collection 'google_vision'.")
                            return True
                        
                        mongo_client_available_main = get_mongo_client() is not None
                        
                        if mongo_client_available_main:
                            st.button(
                                "💾 Lưu vào DB",
                                type="primary",
                                disabled=not st.session_state.get("main_done", False),
                                on_click=save_main_to_db,
                                key="save_main_btn"
                            )
                        else:
                            st.button("💾 Lưu vào DB", disabled=True)
                
                else:
                    st.info("Hóa đơn này chưa có mặt hàng nào được trích xuất.")

# ========== ADVANCED / DATABASE ==========
with tab_advanced:
    tab_extract, tab_db = st.tabs(["Trích xuất (Advanced)", "Database & Evaluation"])
    
    with tab_extract:
        st.write("Tải lên một hoặc nhiều hình ảnh hóa đơn")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            selected_ocr = st.selectbox("Chọn phương thức OCR/Vision", OCR_METHODS)
            
            uploaded_files = st.file_uploader(
                "Chọn ảnh hóa đơn...",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="advanced_uploader"
            )
            
            if uploaded_files:
                st.write(f"Đã chọn {len(uploaded_files)} tệp.")
                
                with st.expander(f"Xem {len(uploaded_files)} ảnh đã tải lên (preview)"):
                    num_cols = 5
                    for i, uploaded_file in enumerate(uploaded_files):
                        if i % num_cols == 0:
                            cols = st.columns(num_cols)
                        
                        uploaded_file.seek(0)
                        img = Image.open(uploaded_file)
                        try:
                            img = ImageOps.exif_transpose(img)
                        except Exception:
                            pass
                        
                        w, h = img.size
                        if w > h * 1.2:
                            img = img.rotate(90, expand=True)
                        
                        with cols[i % num_cols]:
                            st.image(img, width='stretch', caption=f"HĐ {i+1}")
            
            if st.button("Trích xuất thông tin (Advanced)", type="primary", disabled=not uploaded_files):
                st.session_state.extraction_done = False
                master_items_list = []
                master_raw_responses = []
                master_invoice_totals = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.info(f"Đang xử lý Hóa đơn số {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    progress_bar.progress((i) / len(uploaded_files))
                    
                    image_pil = None
                    img_np_rgb = None
                    image_bytes_for_ocr = None
                    
                    try:
                        uploaded_file.seek(0)
                        original_bytes = uploaded_file.read()
                        image_pil = Image.open(BytesIO(original_bytes))
                        
                        try:
                            image_pil = ImageOps.exif_transpose(image_pil)
                        except Exception:
                            pass
                        
                        w, h = image_pil.size
                        if w > h * 1.2:
                            image_pil = image_pil.rotate(90, expand=True)
                        
                        if image_pil.mode == 'RGBA':
                            background = Image.new('RGB', image_pil.size, (255, 255, 255))
                            background.paste(image_pil, mask=image_pil.split()[3])
                            img_np_rgb = np.array(background)
                        elif image_pil.mode != 'RGB':
                            img_np_rgb = np.array(image_pil.convert("RGB"))
                        else:
                            img_np_rgb = np.array(image_pil)
                        
                        ocr_text = None
                        model_raw_output = None
                        model_clean_json = "{}"
                        
                        if selected_ocr == "Vision":
                            image_bytes_for_ocr = BytesIO()
                            Image.fromarray(img_np_rgb).save(image_bytes_for_ocr, format="PNG")
                            image_bytes_for_ocr = image_bytes_for_ocr.getvalue()
                            model_raw_output, model_clean_json = get_invoice_response(image_bytes_for_ocr)
                        else:
                            if selected_ocr == "Google Vision":
                                image_bytes_for_ocr = BytesIO()
                                Image.fromarray(img_np_rgb).save(image_bytes_for_ocr, format="PNG")
                                image_bytes_for_ocr = image_bytes_for_ocr.getvalue()
                                ocr_text = ocr_google_vision_api_key(image_bytes_for_ocr)
                            elif selected_ocr == "Paddle":
                                ocr_text = ocr_paddle_vietocr(img_np_rgb)
                            elif selected_ocr == "EasyOCR":
                                ocr_text = ocr_easyocr_vietocr(img_np_rgb)
                            
                            if not ocr_text or not ocr_text.strip():
                                st.warning(f"⚠️ Không trích xuất được text từ {uploaded_file.name}")
                                model_raw_output = json.dumps({"error": "No OCR text extracted"})
                                model_clean_json = "{}"
                            else:
                                model_raw_output, model_clean_json = get_invoice_response(ocr_text)
                        
                        main_raw_responses.append({
                            "hoa_don_so": i+1,
                            "file": uploaded_file.name,
                            "ocr_text": ocr_text,
                            "bbox": ocr_boxes,
                            "model_output": model_raw_output,
                            "response": model_clean_json
                        })

                        
                        clean = model_clean_json.strip() if isinstance(model_clean_json, str) else "{}"
                        json_start, json_end = clean.find('{'), clean.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            clean = clean[json_start:json_end+1]
                        else:
                            clean = "{}"
                        
                        try:
                            json_data = json.loads(clean)
                        except json.JSONDecodeError:
                            json_data = {"items": [], "tong_tien": None}
                        
                        if "items" not in json_data or not isinstance(json_data.get("items"), list):
                            json_data["items"] = []
                        if "tong_tien" not in json_data:
                            json_data["tong_tien"] = None
                        
                        items = json_data.get("items", [])
                        items = postprocess_invoice_items_vi(items)
                        
                        for it in items:
                            it["file_name"] = uploaded_file.name
                            it["hoa_don_id"] = i + 1
                        
                        master_items_list.extend(items)
                        
                        tong_tien = json_data.get("tong_tien", None)
                        current_total_numeric = None
                        if tong_tien is not None:
                            try:
                                if isinstance(tong_tien, str):
                                    tong_tien = tong_tien.replace(".", "")
                                current_total_numeric = float(tong_tien)
                            except Exception:
                                pass
                        
                        master_invoice_totals.append({
                            "id": i+1,
                            "total_value": current_total_numeric,
                            "file_name": uploaded_file.name
                        })
                    
                    except Exception:
                        traceback.print_exc()
                    
                    finally:
                        del image_pil, img_np_rgb, image_bytes_for_ocr
                        gc.collect()
                
                progress_bar.progress(1.0)
                status_text.success(f"✅ Đã xử lý xong {len(uploaded_files)} tệp!")
                
                st.session_state.aggregated_items = master_items_list
                st.session_state.invoice_totals = master_invoice_totals
                st.session_state.aggregated_raw = master_raw_responses
                st.session_state.extraction_done = True
        
        with col2:
            if st.session_state.invoice_totals:
                st.subheader("Tổng tiền theo từng hóa đơn")
                
                grand_total = 0
                for data in st.session_state.invoice_totals:
                    total_val = data['total_value']
                    value_str = "Không có" if total_val is None else f"{total_val:,.0f}".replace(",", ".") + " VNĐ"
                    st.metric(label=f"Hóa đơn số {data['id']} ({data['file_name']})", value=value_str)
                    
                    if total_val is not None:
                        grand_total += total_val
                
                st.divider()
                st.metric("Tổng cộng (tất cả hóa đơn)", f"{grand_total:,.0f}".replace(",", ".") + " VNĐ")
        
        st.divider()
        
        if st.session_state.aggregated_items:
            st.subheader("Chi tiết mặt hàng (tổng hợp)")
            items_list = st.session_state.aggregated_items
            df = pd.DataFrame(items_list)
            
            for c in ["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien", "file_name", "hoa_don_id"]:
                if c not in df.columns:
                    df[c] = None
            
            df["Ảnh nguồn"] = df["file_name"]
            df = df[["Ảnh nguồn", "hoa_don_id", "ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]]
            df.columns = ["Ảnh nguồn", "HĐ số", "Tên Hàng", "ĐV Tính", "Số Lượng", "Đơn Giá", "Thành Tiền"]
            df = normalize_dataframe_types(df)
            
            st.write("Bạn có thể nhấp đúp để sửa lỗi trích xuất:")
            df_edited = st.data_editor(
                df,
                width='stretch',
                num_rows="dynamic",
                key="data_editor_results"
            )
            
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="Tải về CSV (dữ liệu đã sửa)",
                data=convert_df_to_csv(df_edited),
                file_name="hoa_don_trich_xuat.csv",
                mime="text/csv"
            )
        
        if selected_ocr == "Vision" and st.session_state.aggregated_items:
            st.divider()
            st.subheader("Chỉnh sửa dữ liệu từng hóa đơn trước khi lưu DB")
            
            invoice_options = [f"HĐ #{d['id']}: {d['file_name']}" for d in st.session_state.invoice_totals]
            selected_invoice = st.selectbox("Chọn hóa đơn cần sửa", invoice_options, key="select_invoice_to_edit")
            
            if selected_invoice:
                inv_id = int(selected_invoice.split("#")[1].split(":")[0])
                inv_data = next((d for d in st.session_state.invoice_totals if d['id'] == inv_id), None)
                
                if inv_data:
                    file_name = inv_data['file_name']
                    raw_entry = next((r for r in st.session_state.aggregated_raw if r['file'] == file_name), None)
                    
                    if not raw_entry:
                        st.warning("Không tìm thấy dữ liệu")
                    else:
                        try:
                            clean = raw_entry.get("response", "{}")
                            if isinstance(clean, str):
                                s = clean.strip()
                                a = s.find('{')
                                b = s.rfind('}')
                                if a != -1 and b != -1 and b > a:
                                    clean = s[a:b+1]
                                parsed = json.loads(clean)
                            else:
                                parsed = clean
                        except Exception:
                            parsed = {"items": [], "tong_tien": None}
                        
                        if file_name in st.session_state.edited_data_for_db:
                            current_items = st.session_state.edited_data_for_db[file_name].get("items", [])
                            current_total = st.session_state.edited_data_for_db[file_name].get("tong_tien", None)
                        else:
                            current_items = parsed.get("items", [])
                            current_total = parsed.get("tong_tien", None)
                        
                        col_edit1, col_edit2 = st.columns([3, 1])
                        
                        with col_edit1:
                            st.write("Danh sách items:")
                            df_inv = pd.DataFrame(current_items)
                            
                            for c in ["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]:
                                if c not in df_inv.columns:
                                    df_inv[c] = None
                            
                            df_inv = df_inv[["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]]
                            
                            with st.form(key=f"form_edit_{file_name}"):
                                edited_inv = st.data_editor(
                                    df_inv,
                                    num_rows="dynamic",
                                    key=f"editor_{file_name}",
                                    width='stretch'
                                )
                                
                                new_total = st.number_input(
                                    "Tổng tiền hóa đơn",
                                    value=float(current_total) if current_total is not None else 0.0,
                                    key=f"total_{file_name}"
                                )
                                
                                submitted = st.form_submit_button("Lưu thay đổi", type="primary")
                                
                                if submitted:
                                    items_list = edited_inv.to_dict('records')
                                    st.session_state.edited_data_for_db[file_name] = {
                                        "items": items_list,
                                        "tong_tien": new_total if new_total != 0 else None
                                    }
                                    st.success(f"Đã lưu thay đổi cho {file_name}")
                                    st.rerun()
                        
                        with col_edit2:
                            edited_count = len(st.session_state.edited_data_for_db)
                            total_count = len(st.session_state.invoice_totals)
                            st.metric("Đã sửa", f"{edited_count}/{total_count}")
                            
                            if file_name in st.session_state.edited_data_for_db:
                                st.success("Đã sửa")
                            else:
                                st.info("Chưa sửa")
        
        elif st.session_state.extraction_done and not st.session_state.aggregated_items:
            st.warning("Không có mặt hàng nào được tìm thấy.")
        
        if st.session_state.get("extraction_done", False):
            st.divider()
            st.subheader("JSON Raw")
            
            raw_list = st.session_state.get("aggregated_raw", [])
            if raw_list:
                for entry in raw_list:
                    with st.expander(f"HĐ {entry.get('hoa_don_so')} — {entry.get('file')}"):
                        if entry.get("ocr_text"):
                            st.text_area(
                                "OCR Text",
                                entry.get("ocr_text", ""),
                                height=120,
                                key=f"ocr_{entry.get('hoa_don_so')}"
                            )
                        
                        st.subheader("Model output (raw):")
                        st.code(entry.get("model_output", "{}"), language="json")
                        
                        st.subheader("Cleaned JSON (parsed):")
                        st.code(entry.get("response", "{}"), language="json")
            else:
                st.info("Không có dữ liệu JSON raw.")
        
        st.divider()
        st.subheader("Lưu kết quả vào Database")
        
        mongo_client_available_adv = get_mongo_client() is not None
        
        if not mongo_client_available_adv:
            st.info("MongoDB chưa được cấu hình.")
            st.button("Lưu vào DB", disabled=True)
        else:
            if st.button("Lưu tất cả vào DB", type="primary", disabled=not st.session_state.extraction_done):
                if not st.session_state.extraction_done:
                    st.warning("Chạy trích xuất trước khi lưu.")
                else:
                    with st.spinner("Đang lưu vào database..."):
                        if selected_ocr == "Vision" and st.session_state.edited_data_for_db:
                            st.info(f"Lưu với {len(st.session_state.edited_data_for_db)} hóa đơn đã chỉnh sửa")
                        
                        success = save_extraction_batch(selected_ocr)
                        
                        if success and selected_ocr == "Vision":
                            st.session_state.edited_data_for_db = {}
    
    with tab_db:
        st.subheader("Dữ liệu MongoDB & đánh giá độ chính xác")
        
        with st.expander("Xem dữ liệu chi tiết trong collection", expanded=True):
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                db_method = st.selectbox("Chọn collection (theo phương thức OCR)", OCR_METHODS)
            
            with col_b:
                if st.button("Làm mới dữ liệu"):
                    st.rerun()
            
            docs = list_documents_for_method(db_method)
            
            if docs is None:
                st.error("Không thể kết nối tới MongoDB.")
            elif len(docs) == 0:
                st.info(f"Collection cho '{db_method}' hiện chưa có tài liệu.")
            else:
                tbl = []
                for d in docs:
                    items = d.get("items", [])
                    if items:
                        for item in items:
                            row = item.copy()
                            row["HĐ số"] = d.get("hoa_don_so")
                            row["File"] = d.get("file")
                            row["Tổng Tiền HĐ"] = d.get("tong_tien")
                            row["Saved At"] = d.get("saved_at")
                            tbl.append(row)
                    else:
                        tbl.append({
                            "HĐ số": d.get("hoa_don_so"),
                            "File": d.get("file"),
                            "Tổng Tiền HĐ": d.get("tong_tien"),
                            "Saved At": d.get("saved_at")
                        })
                
                if tbl:
                    df_tbl = pd.DataFrame(tbl)
                    cols_order = [
                        "HĐ số", "File", "ten_hang", "so_luong", "don_vi_tinh",
                        "don_gia", "thanh_tien", "Tổng Tiền HĐ", "Saved At"
                    ]
                    final_cols = [c for c in cols_order if c in df_tbl.columns]
                    df_tbl = normalize_dataframe_types(df_tbl)
                    
                    st.dataframe(df_tbl[final_cols], width='stretch')
                    st.caption(f"Tổng số bản ghi: {len(docs)}")
        
        st.divider()
        st.subheader("So sánh độ chính xác")
        
        comparison_mode = st.radio(
            "Chọn phương thức so sánh:",
            ["So sánh theo trường (Field-level)", "So sánh theo hóa đơn (Invoice-level)"],
            horizontal=True
        )
        
        if st.button("Bắt đầu tính toán độ chính xác", type="primary"):
            with st.spinner("Đang tải dữ liệu và tính toán..."):
                gt_docs = fetch_all_docs("Vision")
                
                if not gt_docs:
                    st.warning("Chưa có dữ liệu trong collection 'Vision' để làm chuẩn.")
                else:
                    st.success(f"Đã tải {len(gt_docs)} hóa đơn mẫu từ Vision (ground truth)")
                    
                    methods_to_compare = [m for m in OCR_METHODS if m != "Vision"]
                    results_container = st.container()
                    
                    if comparison_mode == "So sánh theo trường (Field-level)":
                        chart_data = []
                        
                        for method in methods_to_compare:
                            stats, files_matched = calculate_accuracy_stats(gt_docs, method)
                            
                            with results_container:
                                st.markdown(f"### Phương thức: {method}")
                                st.caption(f"Đã so sánh trên {files_matched}/{len(gt_docs)} hóa đơn khớp tên file.")
                                
                                rows = []
                                field_map_vi = {
                                    "tong_tien": "Tổng tiền HĐ",
                                    "ten_hang": "Tên hàng",
                                    "don_vi_tinh": "Đơn vị tính",
                                    "so_luong": "Số lượng",
                                    "don_gia": "Đơn giá",
                                    "thanh_tien": "Thành tiền (Item)"
                                }
                                
                                for field, count_data in stats.items():
                                    if field == "items_found":
                                        continue
                                    
                                    correct = count_data["correct"]
                                    total = count_data["total"]
                                    acc = (correct / total * 100) if total > 0 else 0.0
                                    
                                    rows.append({
                                        "Trường thông tin": field_map_vi.get(field, field),
                                        "Chính xác": correct,
                                        "Tổng mẫu (Vision)": total,
                                        "Tỷ lệ (%)": f"{acc:.2f}%"
                                    })
                                    
                                    chart_data.append({
                                        "Method": method,
                                        "Field": field_map_vi.get(field, field),
                                        "Accuracy": acc
                                    })
                                
                                st.table(pd.DataFrame(rows))
                        
                        if chart_data:
                            st.divider()
                            st.subheader("Biểu đồ so sánh")
                            df_chart = pd.DataFrame(chart_data)
                            st.bar_chart(
                                df_chart,
                                x="Field",
                                y="Accuracy",
                                color="Method",
                                stack=False
                            )
                    
                    else:
                        chart_data_invoice = []
                        
                        for method in methods_to_compare:
                            target_docs = fetch_all_docs(method)
                            target_map = {d.get('file'): d for d in target_docs if d.get('file')}
                            
                            total_invoices = 0
                            correct_invoices = 0
                            invoice_details = []
                            
                            for gt_doc in gt_docs:
                                fname = gt_doc.get('file')
                                if not fname:
                                    continue
                                
                                pred_doc = target_map.get(fname)
                                if not pred_doc:
                                    invoice_details.append({
                                        "File": fname,
                                        "Kết quả": "Thiếu dữ liệu"
                                    })
                                    total_invoices += 1
                                    continue
                                
                                total_invoices += 1
                                is_correct = True
                                errors = []
                                
                                gt_total = gt_doc.get("tong_tien")
                                pred_total = pred_doc.get("tong_tien")
                                
                                if gt_total is not None:
                                    if normalize_val(gt_total) != normalize_val(pred_total):
                                        if not (
                                            isinstance(gt_total, (int, float))
                                            and isinstance(pred_total, (int, float))
                                            and abs(gt_total - pred_total) < 100
                                        ):
                                            is_correct = False
                                            errors.append("Tổng tiền sai")
                                
                                gt_items = gt_doc.get("items", [])
                                pred_items = pred_doc.get("items", [])
                                
                                if len(gt_items) != len(pred_items):
                                    is_correct = False
                                    errors.append(f"Số lượng items khác nhau ({len(pred_items)}/{len(gt_items)})")
                                
                                for i, gt_item in enumerate(gt_items):
                                    pred_item = pred_items[i] if i < len(pred_items) else {}
                                    
                                    for field in ["ten_hang", "don_vi_tinh", "so_luong", "don_gia", "thanh_tien"]:
                                        gt_val = gt_item.get(field)
                                        if gt_val is not None and str(gt_val).strip() != "":
                                            pred_val = pred_item.get(field)
                                            if normalize_val(gt_val) != normalize_val(pred_val):
                                                is_correct = False
                                                errors.append(f"Item {i+1}: {field}")
                                
                                if is_correct:
                                    correct_invoices += 1
                                    invoice_details.append({
                                        "File": fname,
                                        "Kết quả": "Đúng 100%"
                                    })
                                else:
                                    invoice_details.append({
                                        "File": fname,
                                        "Kết quả": f"Sai: {', '.join(errors[:3])}"
                                    })
                            
                            accuracy = (correct_invoices / total_invoices * 100) if total_invoices > 0 else 0.0
                            
                            with results_container:
                                st.markdown(f"### Phương thức: {method}")
                                st.metric("Độ chính xác (theo hóa đơn)", f"{accuracy:.2f}%")
                                st.caption(f"{correct_invoices}/{total_invoices} hóa đơn đúng hoàn toàn")
                                
                                with st.expander(f"Chi tiết từng hóa đơn ({method})"):
                                    df_details = pd.DataFrame(invoice_details)
                                    st.dataframe(df_details, width='stretch', height=400)
                                
                                st.divider()
                            
                            chart_data_invoice.append({
                                "Method": method,
                                "Accuracy": accuracy,
                                "Correct": correct_invoices,
                                "Total": total_invoices
                            })
                        
                        if chart_data_invoice:
                            st.divider()
                            st.subheader("Biểu đồ độ chính xác theo hóa đơn")
                            df_chart_inv = pd.DataFrame(chart_data_invoice)
                            st.bar_chart(
                                df_chart_inv,
                                x="Method",
                                y="Accuracy",
                                width='stretch'
                            )
                            st.table(df_chart_inv[["Method", "Correct", "Total", "Accuracy"]])
