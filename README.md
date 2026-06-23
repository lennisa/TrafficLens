# TrafficLens 🚦

### AI-Assisted Traffic Violation & Road Safety Monitoring System

TrafficLens is an AI powered smart traffic monitoring platform designed to analyze traffic images, videos, and live camera streams. It detects road users, traffic signs, violations, and license plates, then generates structured and visual evidence for monitoring and enforcement workflows.

Built for **Gridlock Hackathon 2.0**.

---

## Problem Statement

Manual monitoring of thousands of CCTV feeds is slow, expensive, error prone, and difficult to scale.

TrafficLens helps automate traffic surveillance by:

- Detecting vehicles, pedestrians, riders, animals, and autorickshaws
- Identifying traffic lights, traffic signs, and violation related context
- Detecting license plates and extracting registration numbers using OCR
- Generating annotated images/videos with bounding boxes and confidence scores
- Saving analysis sessions and detection records for later review

---

## Core AI Modules

TrafficLens combines three computer vision models:

| Module | Full Form                               | Purpose                                                                             |
| ------ | --------------------------------------- | ----------------------------------------------------------------------------------- |
| VPD    | Vehicle & Pedestrian Detection          | Detects road users and Indian road specific objects                                 |
| VID    | Traffic Violation & Road Sign Detection | Detects traffic signs, traffic lights, vehicles, pedestrians, and violation context |
| LPR    | License Plate Recognition               | Detects license plates and extracts plate text using OCR                            |

---

## 1. VPD — Vehicle & Pedestrian Detection

VPD is trained for Indian road conditions, including mixed traffic, autorickshaws, riders, pedestrians, animals, and traffic infrastructure.

### Detected Classes

```text
Person
Car
Truck
Bus
Motorcycle
Bicycle
Animal
Autorickshaw
Rider
Traffic Light
Traffic Sign
```

### Model Details

| Property     | Value                        |
| ------------ | ---------------------------- |
| Architecture | YOLO11m                      |
| Input Size   | 640 × 640 RGB                |
| Dataset      | Indian Driving Dataset (IDD) |
| Training     | 70 epochs, batch size 32     |
| Optimizer    | AdamW                        |
| Hardware     | 2 × NVIDIA T4 GPUs           |
| Deployment   | ONNX (`vpd.onnx`)            |

### Performance Metrics

| Metric    | Score |
| --------- | ----: |
| Precision | 0.778 |
| Recall    | 0.539 |
| mAP@50    | 0.602 |
| mAP@50-95 | 0.390 |

---

## 2. VID — Traffic Violation & Road-Sign Detection

VID detects vehicles, pedestrians, traffic lights, traffic signs, speed limits, and road sign context needed for traffic violation analysis.

### Example Detection Categories

```text
Vehicles and Pedestrians
Red Light / Green Light
Stop Sign
No Entry
No Overtaking
No U-Turn
No Left Turn
No Right Turn
Speed Limit Signs (20–120 km/h)
```

### Model Details

| Property        | Value                               |
| --------------- | ----------------------------------- |
| Architecture    | YOLO11m                             |
| Parameters      | 20.05M                              |
| Layers          | 126                                 |
| Compute         | 67.7 GFLOPs @ 640 × 640             |
| Dataset         | Traffic Violation Detection Dataset |
| Validation Data | 1,470 images / 1,592 objects        |
| Training        | 150 epochs, batch size 32           |
| Hardware        | 2 × NVIDIA T4 GPUs                  |
| Deployment      | ONNX (`best.onnx`)                  |

### Performance Metrics

| Metric    | Score |
| --------- | ----: |
| Precision | 0.928 |
| Recall    | 0.877 |
| mAP@50    | 0.927 |
| mAP@50-95 | 0.810 |

---

## 3. LPR — License Plate Recognition

LPR detects license plates from traffic frames and uses OCR to extract vehicle registration numbers.

### Pipeline

```text
Camera Frame
      ↓
License Plate Detector
      ↓
Plate Crop
      ↓
OCR Engine
      ↓
Registration Number
      ↓
Violation Evidence
```

### Model Details

| Property     | Value                    |
| ------------ | ------------------------ |
| Architecture | YOLOv8n (Nano)           |
| Parameters   | 3.01M                    |
| Layers       | 73                       |
| Compute      | 8.1 GFLOPs @ 640 × 640   |
| Dataset      | License Plate Dataset    |
| Training     | 50 epochs, batch size 32 |
| Hardware     | 1 × NVIDIA T4 GPU        |
| Deployment   | ONNX (`best.onnx`)       |
| OCR          | FastPlateOCR             |

### Performance Metrics

| Metric    | Score |
| --------- | ----: |
| Recall    | 0.877 |
| mAP@50    | 0.992 |
| mAP@50-95 | 0.862 |

---

## Key Features

- Three AI modules integrated into one dashboard
- Image upload support
- Video upload support
- Demo image and demo video modes
- Camera capture support
- Live stream inference
- Bounding boxes, confidence scores, and class counts
- Annotated output images and videos
- License plate OCR with valid and partial plate counts
- Session tracking with unique session IDs
- Detection records with frame/time context
- Bounding-box coordinates: `[x1, y1, x2, y2]`
- Metrics dashboard for all three models

---

## Proposed End-to-End Pipeline

```text
Camera / CCTV Feed / Uploaded Video / Captured Image
                        ↓
                Image Preprocessing
                        ↓
        Vehicle & Pedestrian Detection (VPD)
                        ↓
 Traffic Violation & Road Sign Detection (VID)
                        ↓
        License Plate Recognition + OCR (LPR)
                        ↓
          Violation Rule Engine (Proposed)
                        ↓
     Evidence Generation + Session Tracking
                        ↓
      Dashboard, Records, Analytics & Reports
```

---

## Project Structure

```text
TrafficLens/
│
├── backend/          # FastAPI backend and inference APIs
├── frontend/         # React frontend dashboard
├── demo/             # Demo images and videos
├── models/           # Model configuration / model loading files
├── outputs/          # Generated annotated outputs
├── preprocess/       # Image and video preprocessing utilities
├── tasks/            # Task-specific inference modules
├── train/            # Training notebooks and scripts
├── .env.example      # Example environment variables
└── .gitignore
```

> Model weights such as `.pt`, `.onnx`, `.engine`, and generated outputs are intentionally excluded from this repository using `.gitignore`.

---

## Tech Stack

### Frontend

- React
- JavaScript
- CSS

### Backend

- FastAPI
- Python
- Uvicorn

### AI / Computer Vision

- Ultralytics YOLO
- YOLO11m
- YOLOv8n
- ONNX Runtime
- OpenCV
- FastPlateOCR

---

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/lennisa/TrafficLens.git
cd TrafficLens
```

### 2. Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
cd ..
pip install -r requirements.txt
```

Run the backend:

```bash

 python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

 python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

```

### 3. Frontend Setup

Open a new terminal:

```bash
cd TrafficLens (if you are not in the folder)
cd frontend

npm install/npm i

npm install/npm i

npm start
```

## Team

**traffic-bread**

- Diptyajit Das
- Angelica Das

---

## License

This project is built upon datasets that are not openly licensed and are not
publicly available. As a result, this project and any outputs, models, or
artifacts derived from it are themselves considered derived works and cannot
be treated as open data or open-source material. No open license is granted
over the underlying data or any components directly derived from it.
