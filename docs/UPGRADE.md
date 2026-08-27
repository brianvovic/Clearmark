# ClearMark — nâng cấp lên chất lượng "Pro" (Phase 2)

Tài liệu này map thẳng đề xuất trong bản đánh giá (đối chiếu dewatermark.ai) vào
code đã có trong repo, và hướng dẫn bật phần GPU serverless bằng tài khoản của bạn.

## Đã làm sẵn (chạy ngay, CPU, $0) — Phase 1

| Bệnh cũ | Sửa ở đâu |
|--------|-----------|
| Ảnh bị mờ / mất chi tiết (resize toàn ảnh) | `apps/api/services/tiler.py` — inpaint **full-res theo tile**, ngoài mask giữ bit-exact |
| Xóa nhầm chữ in thật | `remove_text` opt-in — checkbox *"Xóa cả chữ"*, mặc định **tắt** |
| Mờ lũy tiến khi xóa nhiều lần | `apps/api/services/sessions.py` — giữ ảnh gốc, **mask cộng dồn**, luôn inpaint từ gốc |
| Không thấy AI bắt đúng chỗ | `/api/detect-mask` + `/api/session/{id}/mask` trả mask ra UI |

Không cần cấu hình gì. `GET /health` → `"engine":"local-cpu"`.

## Bật GPU serverless — Phase 2

Kiến trúc: **API (FastAPI) là não điều phối, GPU worker (Modal) là cơ bắp.**
`apps/api/services/engine.py` là điểm nối duy nhất — có `GPU_WORKER_URL` thì
detect/erase/video chạy trên worker; không thì tự fallback về LaMa CPU.

```
Browser ──▶ Next.js ──▶ FastAPI ──(nếu có GPU_WORKER_URL)──▶ Modal GPU worker
                          │                                    Florence-2 + SAM2 (detect)
                          │                                    LaMa / SDXL (erase) + Real-ESRGAN
                          └─ video job queue ─────────────────▶ ProPainter (video)
                          Storage: temp 1h  ── hoặc ──▶ Cloudflare R2 (TTL 1h)
```

### 1. Deploy worker lên Modal (rẻ nhất, pay-per-second, scale-to-zero)

```bash
pip install modal
modal token new
# đặt secret token (API và worker phải trùng)
modal secret create clearmark-worker-token TOKEN="$(openssl rand -hex 24)"
modal deploy worker/modal_app.py
```

Modal in ra URL dạng `https://<bạn>--clearmark-worker-web.modal.run`.
GPU mặc định `L4` (đủ cho SDXL, ~1¢/ảnh). Đổi bằng biến `CLEARMARK_GPU` (A10G/A100).

### 2. Trỏ API sang worker

`apps/api/.env` (xem `.env.example`):

```
GPU_WORKER_URL=https://<bạn>--clearmark-worker-web.modal.run
GPU_WORKER_TOKEN=<đúng TOKEN đã tạo ở trên>
GPU_PREDICT_MODE=3.0        # 3.0 = LaMa nhanh; 4.0 = SDXL cho ca khó
```

Khởi động lại API → `GET /health` phải thấy `"engine":"gpu-worker"`.
Ảnh giờ dùng Florence-2 + SAM2 để bắt watermark/logo mờ, và có nhánh SDXL cho nền phức tạp.

### 3. (Tùy chọn) Lưu video trên Cloudflare R2

Video dùng luồng bất đồng bộ **upload → task → poll** (`/api/video/*`). Mặc định
lưu ở temp 1 giờ; để chạy thật, bật R2 (miễn phí 10GB, 0đ egress):

```
R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
R2_BUCKET=clearmark
R2_ACCESS_KEY=...
R2_SECRET_KEY=...
```

`pip install boto3`. Thêm lifecycle rule xóa object sau 1 ngày để chốt chi phí.

## Định tuyến model theo giá (giống dewatermark v5/v2)

- **3.0** — LaMa full-res tiling. Nhanh (~vài giây GPU), đủ cho ~80% ca. Mặc định.
- **4.0** — SDXL inpainting (`services/engine.py` gửi `predict_mode=4.0`). Cho nền
  phức tạp / watermark lớn. Đắt hơn ~3× compute → tính 3× credit nếu bạn thu phí.
- Cả hai đều chạy thêm **Real-ESRGAN** chỉ trên vùng mask để che độ mềm sau inpaint.

## Video

- Client nên dùng **ffmpeg.wasm** để cắt/preview trước khi upload (giảm tải server).
- Worker giả định **watermark tĩnh**: detect 1 lần (hoặc nhận 1 mask brush) → ProPainter
  inpaint nhất quán thời gian → mux lại audio gốc. Không có ProPainter thì fallback
  LaMa từng frame (vẫn dùng chung 1 mask tĩnh nên ít nhấp nháy).
- Giới hạn khuyến nghị: MP4/H.264, ≤ 200MB, ≤ vài phút.

## Còn lại phải tự làm (cần dữ liệu/GPU của bạn)

- **Dataset watermark tự sinh** để fine-tune detector + model xóa (thứ tạo nên "v5").
  Script sinh cặp (ảnh-có-wm, ảnh-gốc, mask) là bước tiếp theo đáng làm nhất.
- **Auth + credit + thanh toán** (Firebase/Clerk + Stripe) nếu thương mại hóa.

## Pháp lý

Chỉ xử lý ảnh/video bạn sở hữu hoặc được ủy quyền. Xóa watermark của người khác để
tái sử dụng là vi phạm bản quyền ở hầu hết các nước.
