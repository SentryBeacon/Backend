<div align="center">

  <img src="https://raw.githubusercontent.com/SentryBeacon/.github/main/profile/assets/video.gif" width="100%" alt="SentryBeacon Demo">

  <br/>

  <img src="https://raw.githubusercontent.com/SentryBeacon/.github/main/profile/assets/banner.svg" width="100%" alt="SentryBeacon Banner">

  <h1>🗼 SentryBeacon — Traffic Vision System</h1>
  <p><i>"Smart Vision for Safer Roads – Drive with care, someone is waiting for you."</i> 🛡️</p>

  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Model-YOLOv8-orange.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Focus-Computer%20Vision-blueviolet.svg" alt="CV">

</div>

---
## 📝 Giới thiệu

**SentryBeacon** là hệ thống giám sát giao thông thông minh được xây dựng cho bài toán **nhận diện biển số xe và phân loại phương tiện giao thông**, đồng thời phát hiện tự động các hành vi vi phạm như **vượt đèn đỏ** và **đi sai làn đường**.

> *Đề tài: Nhận diện biển số xe và phân loại phương tiện giao thông, phát hiện các hành vi vi phạm như vượt đèn đỏ và đi sai làn.*

---

## 🚀 Tính năng chính

| # | Tính năng | Mô tả |
|---|---|---|
| 🚗 | **Vehicle Detection** | Nhận diện và phân loại phương tiện (ô tô, xe máy, xe tải, xe buýt) theo thời gian thực bằng YOLOv8 |
| 🔤 | **License Plate Recognition (LPR)** | Trích xuất vùng biển số và OCR ký tự độ chính xác cao bằng PaddleOCR / EasyOCR |
| 🚦 | **Red Light Violation Detection** | Xác định trạng thái đèn đỏ, phát hiện xe vượt vạch dừng bằng Virtual Line và Point-in-Polygon |
| 🛣️ | **Wrong Lane Detection** | Đối chiếu vị trí xe với định nghĩa làn đường, phát hiện xe đi vào làn cấm bằng Polygon Mapping |
| 📦 | **Evidence Storage** | Tự động chụp ảnh toàn cảnh (Scene) và cận cảnh (Crop) khi phát hiện vi phạm, lưu kèm Timestamp |

---

## 🧠 Kiến trúc tổng thể hệ thống

Hệ thống gồm **4 tầng xử lý** kết nối tuần tự:

| Tầng (Layer) | Thành phần xử lý | Chức năng cụ thể |
|---|---|---|
| 1. Input Layer | Camera / Video Stream | Tiếp nhận luồng dữ liệu video từ camera giám sát tại ngã tư |
| 2. Perception Layer | YOLOv8 + SORT Tracker | YOLOv8 phát hiện xe (Bbox) và phân loại (Class); SORT Tracker gán ID duy nhất để theo dõi quỹ đạo (Trajectory) |
| 3. Logic Layer | Geometry Analysis | Module 2: kiểm tra vạch dừng ảo khi đèn đỏ bật; Module 3: kiểm tra tâm xe trong vùng Polygon làn đường |
| 4. Output Layer | OCR & Database | Kích hoạt OCR nhận diện biển số khi `Vi phạm = True`; lưu bằng chứng vào SQL Server |

---

## 🔧 Backend — 3 Module xử lý

| Đặc điểm | Module 1: Nhận diện & Phân loại (`detection_ocr.py`) | Module 2: Vi phạm đèn đỏ (`red_light_logic.py`) | Module 3: Đi sai làn đường (`wrong_lane_detector.py`) |
|---|---|---|---|
| **Chức năng chính** | Phát hiện phương tiện; Phân loại (ô tô, xe máy...); Trích xuất biển số xe | Xác định trạng thái đèn; Phát hiện xe vượt vạch dừng | Đối chiếu vị trí xe với làn; Phát hiện xe đi vào làn cấm |
| **Công nghệ** | YOLOv8 · PaddleOCR / EasyOCR · OpenCV | Point-in-Polygon · Virtual Line (Vạch ảo) · Logic Timer/AI | ROI / Polygon Mapping · Logic Type-check · Spatial Analysis |
| **Đầu vào** | Frame video (giám sát); Ảnh phương tiện (từ Mod 2/3) | Tọa độ xe; Tọa độ vạch dừng (Web config); Trạng thái đèn | Tọa độ + Loại xe; Danh sách tọa độ làn đường; Quy tắc làn (Web config) |
| **Đầu ra** | Loại xe, tọa độ Bbox; Chuỗi ký tự biển số (String); Video/Ảnh kết quả | Trạng thái vi phạm (True/False); Loại lỗi: "Vượt đèn đỏ"; Video bằng chứng | Trạng thái vi phạm (True/False); Loại lỗi: "Sai làn đường"; Video bằng chứng |
| **Lưu trữ** | Bảng `Vehicles`, `Plates`; Thư mục `crop_plates/` | Bảng `Violations`; Thư mục `evidence_red/` | Bảng `Violations`; Thư mục `evidence_lane/` |
| **Nhận nhiệm vụ** | Hoàng Thị Hoạt | Đỗ Công Trí | Nguyễn Đức Mạnh |

---

## ⚙️ Pipeline Module 3 — Đi sai làn đường

| Giai đoạn | Thành phần chính | Chi tiết kỹ thuật & Logic xử lý |
|---|---|---|
| 1. Nhập liệu | Video Stream | Đọc luồng video qua VideoReader đa luồng để không làm tắc nghẽn hàng đợi xử lý |
| 2. Tiền xử lý | Frame Scaling | Resize khung hình về 960px (INFER_W) để cân bằng giữa độ chính xác và tốc độ FPS |
| 3. Phát hiện | YOLOv8m AI | Nhận diện phương tiện (Car, Motor, Bus, Truck) và lọc qua NMS với ngưỡng IoU 0.45 |
| 4. Theo dõi | SORT Tracker | Bộ lọc Kalman dự đoán quỹ đạo và gán ID duy nhất cho từng phương tiện |
| 5. Phân tích | State Machine | Quản lý vòng đời xe qua 4 trạng thái: `UNSEEN → ENTERING → TRACKING → VIOLATED` |
| 6. Xác thực | Line Crossing | Kiểm tra va chạm đa điểm (phần dưới xe, tâm xe và 4 góc bbox) với vạch vi phạm |
| 7. Lưu trữ | Violation Saver | Chụp ảnh toàn cảnh (Scene) và cận cảnh (Crop) khi phát hiện vi phạm, kèm Timestamp |

---

## 🧩 Khái niệm cốt lõi

**1. Object Detection** — Mô hình **YOLOv8m** xác định vị trí bounding box và phân loại phương tiện trong từng frame từ camera giám sát.

**2. SORT Tracker** — Sử dụng **bộ lọc Kalman** dự đoán quỹ đạo và gán ID duy nhất cho mỗi xe qua các frame liên tiếp.

**3. Virtual Line & Polygon Mapping** — Vạch dừng ảo (Module 2) và vùng Polygon làn đường (Module 3) được cấu hình qua Web config, dùng thuật toán **Point-in-Polygon** để xác định vi phạm.

**4. State Machine** — Vòng đời của mỗi phương tiện được quản lý qua 4 trạng thái: `UNSEEN → ENTERING → TRACKING → VIOLATED`, đảm bảo chỉ ghi nhận vi phạm đúng thời điểm.

**5. IoU & NMS** — Chỉ số **Intersection over Union** kết hợp **Non-Maximum Suppression** (ngưỡng 0.45) loại bỏ các vùng nhận diện trùng lặp, giữ lại kết quả tin cậy nhất.

**6. OCR Pipeline** — **PaddleOCR / EasyOCR** chỉ được kích hoạt khi `Vi phạm = True`, chuyển ảnh crop biển số thành chuỗi ký tự và lưu vào SQL Server.

---

## 🛠️ Tech Stack

```
Language   :  Python
Vision     :  OpenCV · YOLOv8m
Tracking   :  SORT (Kalman Filter)
OCR        :  PaddleOCR · EasyOCR
ML Backend :  PyTorch
Database   :  SQL Server
Backend    :  Flask (Dashboard / Web config)
Tools      :  Jupyter Notebook · Git LFS
```

---

## 📁 Cấu trúc thư mục

```
SentryBeacon/
├── assets/
│   └── banner.svg              ← banner động README
├── models/                     ← YOLO weights (Git LFS)
├── src/
│   ├── detection_ocr.py        ← Module 1: nhận diện & phân loại + OCR biển số
│   ├── red_light_logic.py      ← Module 2: phát hiện vượt đèn đỏ
│   ├── wrong_lane_detector.py  ← Module 3: phát hiện đi sai làn đường
│   └── dashboard/              ← Flask web app (cấu hình vạch, làn đường)
├── evidence/
│   ├── evidence_red/           ← ảnh/video bằng chứng vượt đèn đỏ
│   ├── evidence_lane/          ← ảnh/video bằng chứng sai làn
│   └── crop_plates/            ← ảnh biển số đã crop
├── notebooks/                  ← phân tích dữ liệu (EDA, thực nghiệm)
└── README.md
```

---

## ⚡ Cài đặt nhanh

```bash
git clone https://github.com/ducmanh-jr/SentryBeacon.git
cd SentryBeacon
pip install -r requirements.txt

# Chạy nhận diện từ webcam
python src/detection_ocr.py --source 0

# Chạy phát hiện vi phạm đèn đỏ từ video
python src/red_light_logic.py --source video.mp4

# Chạy phát hiện sai làn đường
python src/wrong_lane_detector.py --source video.mp4
```

---

## 🔗 Tài nguyên

> 🌐 **[Tài liệu kỹ thuật chi tiết](https://your-link-here.com)**
> 📊 **[Dataset & Notebook phân tích](https://your-link-here.com)**

---

<div align="center">

*Chúng tôi tin rằng công nghệ có thể làm cho con đường về nhà của mỗi người trở nên an toàn hơn.*  
*Hãy giữ vững tay lái và tuân thủ luật lệ giao thông.* ❤️

<br/>

© 2026 **SentryBeacon Team** · Developed by [Nguyen Duc Manh](https://github.com/ducmanh-jr)

</div>
