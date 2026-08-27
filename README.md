# ClearMark — AI xóa watermark (MVP self-host)

Công cụ web tự host: tải ảnh → AI phát hiện và xóa watermark/logo/chữ → tải kết quả.
Bạn dùng **miễn phí, không giới hạn** trên máy của mình. Chưa có đăng nhập / thanh toán / quảng cáo.

## Yêu cầu

- Docker + Docker Compose (khuyến nghị), **hoặc**
- Node.js 20+ và Python 3.11 (API dùng PyTorch — tránh Python 3.14)
- RAM ≥ 8GB
- GPU NVIDIA (tuỳ chọn, nhanh hơn nhiều): đặt `LAMA_DEVICE=cuda`

## Chạy nhanh (không cần Docker) — QUAN TRỌNG

Cách dễ nhất: nhấp đúp 2 file trong thư mục `scripts\`:

1. `scripts\start-api.bat`  → chạy API (tự chọn `.venv311` nếu có = LaMa AI)
2. `scripts\start-web.bat`  → chạy web

Mở http://localhost:3000

> ⚠️ **Phải dùng `.venv311`, KHÔNG dùng `.venv`.**
> `.venv` chỉ có OpenCV → xóa bị **mờ/vỡ** và **không có EasyOCR** nên tự động phát hiện rất kém.
> `.venv311` có **LaMa (AI xóa sạch, không nhòe) + EasyOCR + model big-lama.pt**. Đây là môi trường bắt buộc để có chất lượng như mong muốn.

Chạy thủ công bằng PowerShell (nếu không dùng .bat):

```powershell
# Terminal 1 — API (dùng .venv311!)
cd "C:\Users\MY_LOVE\Downloads\Water AI\apps\api"
.\.venv311\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Web
cd "C:\Users\MY_LOVE\Downloads\Water AI\apps\web"
npm run dev
```

Kiểm tra đang chạy đúng AI: mở http://localhost:8000/health — phải thấy `"inpaint":"lama"`.
Nếu thấy `"inpaint":"opencv"` nghĩa là đang chạy sai môi trường (kết quả sẽ bị mờ).

```bash
cd "Water AI"
docker compose up --build
```

- Web: http://localhost:3000  
- API: http://localhost:8000/health  

Lần đầu sẽ tải model LaMa + EasyOCR (vài trăm MB). Cache nằm trong volume `lama-cache`.

### Bật GPU (NVIDIA Container Toolkit)

Sửa `docker-compose.yml`: đặt `LAMA_DEVICE=cuda` và bỏ comment khối `deploy.resources...devices`.

```bash
LAMA_DEVICE=cuda docker compose up --build
```

## Chạy local (không Docker)

### 1. API

Dùng **Python 3.11** (khuyến nghị). Python 3.14 thường chưa tương thích PyTorch.

```bash
cd apps/api
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Cài torch CPU trước, rồi các gói còn lại
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Web

```bash
cd apps/web
npm install
npm run dev
```

Mở http://localhost:3000 — Next.js proxy `/api/*` sang `http://127.0.0.1:8000`.

## Batch (nhiều ảnh)

- Chọn nhiều ảnh cùng lúc, hoặc tải 1 file **ZIP** chứa ảnh.
- Bấm **Xử lý** — hệ thống xử lý lần lượt (tối đa 30 ảnh, 10MB/ảnh).
- Khi xong, bấm **Tải ZIP kết quả** để tải về tất cả ảnh đã làm sạch trong 1 file ZIP.
- Chế độ **Thủ công** chỉ dùng khi còn 1 ảnh.

1. Chọn **Tự động** hoặc **Thủ công**.
2. Tải / kéo-thả / Ctrl+V ảnh (tối đa 10MB; png, jpg, webp, avif).
3. **Tự động:** CRAFT (neural text detection) phát hiện chữ mờ → tinh chỉnh mức nét chữ → LaMa inpaint. Chữ in thật trong ảnh (logo trên gối, nhãn hiệu...), môi, chi tiết màu đậm được bảo vệ tự động — chỉ watermark bán trong suốt bị xóa.  
   **Thủ công:** tô vùng cần xóa → Xử lý.
4. So sánh trước/sau và **Tải kết quả**.

> **Chất lượng:** ảnh kết quả giữ nguyên độ phân giải gốc. AI chỉ xử lý ở vùng watermark; mọi pixel khác giữ 100% ảnh gốc.
> **Thời gian:** chạy CPU mất ~1–2 phút/ảnh lớn (GPU nhanh hơn nhiều lần — đặt `LAMA_DEVICE=cuda`).

Nếu chế độ tự động báo không phát hiện watermark, hãy dùng **Thủ công**.

## Cấu trúc

```
Water AI/
  apps/web/     # Next.js (UI tiếng Việt)
  apps/api/     # FastAPI + EasyOCR + LaMa
  docker-compose.yml
```

## Nâng cấp chất lượng (v1.1)

Ba lỗi kiến trúc cũ đã được sửa — không cần GPU, chạy ngay trên CPU:

1. **Inpaint full-res theo tile** (`services/tiler.py`). Trước đây cả ảnh bị resize
   về ≤2048px rồi phóng lại → toàn ảnh bị mềm. Giờ chỉ **crop vùng quanh mask ở độ
   phân giải gốc → chia tile 512 có overlap → LaMa từng tile → ghép bằng cửa sổ
   Hann (feather)**. Mọi pixel ngoài mask **giữ nguyên bit-exact**.
2. **`remove_text` opt-in.** Mặc định chỉ xóa logo/watermark màu; bật checkbox
   *"Xóa cả chữ"* mới chạy OCR. Hết cảnh xóa nhầm chữ in thật trên áo/bao bì/biển hiệu.
3. **Session + mask cộng dồn** (`services/sessions.py`). Server giữ ảnh gốc full-res;
   mỗi lần "xóa thêm" luôn inpaint lại từ ảnh gốc với mask cộng dồn, **không bao giờ
   inpaint đè lên ảnh đã xử lý** → không mờ lũy tiến. Ảnh gốc tự xóa sau 1 giờ.

Phần AI nặng (Florence-2 + SAM2 detect, SDXL cho ca khó, Real-ESRGAN, video ProPainter)
chạy trên **GPU serverless** — xem `worker/` và [`docs/UPGRADE.md`](docs/UPGRADE.md).
Khi chưa cấu hình GPU, API tự fallback về LaMa CPU.

## API

| Endpoint | Mô tả |
|----------|--------|
| `POST /api/remove` | Auto remove (`multipart`: `image`; `remove_text=0/1`, mặc định 0) |
| `POST /api/inpaint` | Manual (`image` + `mask`, trắng = xóa) — full-res |
| `POST /api/detect-mask` | Trả mask tự động (`remove_text=0/1`) để preview |
| `POST /api/session` | Mở phiên tinh chỉnh, giữ ảnh gốc full-res server-side |
| `POST /api/session/{id}/erase` | Cộng dồn mask (auto và/hoặc brush) → inpaint từ ảnh gốc |
| `GET /api/session/{id}/mask` | Mask cộng dồn hiện tại (PNG, trắng = sẽ xóa) |
| `POST /api/session/{id}/reset-mask` | Xóa mask cộng dồn, làm lại từ đầu |
| `GET /health` | Health check |

## Pháp lý

Chỉ dùng với ảnh bạn sở hữu, tự tạo, hoặc được ủy quyền chỉnh sửa. Bạn chịu trách nhiệm tuân thủ luật bản quyền khi xóa watermark.

Use this tool only with images you own, created yourself, or are authorized to edit.

## Phase sau (chưa có trong MVP)

- Đăng nhập, giới hạn khách miễn phí, gói PRO
- Quảng cáo (AdSense)
- Batch nhiều file / API public
