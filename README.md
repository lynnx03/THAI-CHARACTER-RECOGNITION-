# 🔤 Thai Character Recognition — Transfer Learning

โปรเจกต์นี้เป็นระบบจำแนกอักษรไทยและเลขไทยจากภาพ โดยใช้เทคนิค Transfer Learning เปรียบเทียบประสิทธิภาพของ 3 โมเดล ได้แก่ **ResNet50**, **EfficientNet-B3** และ **MobileNetV3-Large**

---

## 📋 สารบัญ

- [ภาพรวมโปรเจกต์](#ภาพรวมโปรเจกต์)
- [โครงสร้างโฟลเดอร์](#โครงสร้างโฟลเดอร์)
- [ความต้องการของระบบ](#ความต้องการของระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [การเตรียมข้อมูล](#การเตรียมข้อมูล)
- [วิธีการใช้งาน](#วิธีการใช้งาน)
- [รายละเอียดโมเดล](#รายละเอียดโมเดล)
- [Data Augmentation](#data-augmentation)
- [Output ที่ได้](#output-ที่ได้)

---

## ภาพรวมโปรเจกต์

ระบบนี้สามารถจำแนกตัวอักษรไทยและเลขไทยจากภาพ โดยมีความสามารถหลัก ดังนี้

- **Triple Data Augmentation** — แต่ละภาพถูกเพิ่มเป็น 3 เวอร์ชัน (Gentle / Mild / Strong) เพื่อขยายขนาด dataset เป็น 3 เท่า
- **Transfer Learning** — ใช้ pretrained weights จาก ImageNet พร้อมปรับ input layer รองรับภาพ Grayscale (1 channel)
- **เปรียบเทียบ 3 โมเดล** — ResNet50, EfficientNet-B3, MobileNetV3-Large
- **Evaluation ครบถ้วน** — Confusion Matrix, Classification Report, Training History, Overfitting Analysis
- **ทดสอบภาพเดี่ยว** — นำภาพใหม่มาทดสอบกับโมเดลที่ฝึกไว้แล้ว

---

## โครงสร้างโฟลเดอร์

```
├── Samlong.ipynb               # Notebook หลัก
├── Data_text/                  # Dataset (โฟลเดอร์ย่อยตามชื่ออักษร)
│   ├── ก/
│   ├── ข/
│   ├── ๐/
│   └── ...
├── outputs_resnet50/           # ผลลัพธ์ของ ResNet50
│   ├── resnet50_best.pt
│   ├── training_history.json
│   ├── class_to_idx.json
│   └── misclassified/
├── outputs_efficientnet_b3/    # ผลลัพธ์ของ EfficientNet-B3
├── outputs_mobilenet_v3/       # ผลลัพธ์ของ MobileNetV3-Large
├── training_comparison.png     # กราฟเปรียบเทียบโมเดลทั้ง 3
└── training_data_distribution.png
```

---

## ความต้องการของระบบ

- Python 3.8+
- CUDA (แนะนำ แต่ไม่บังคับ — รองรับ CPU)

### Python Packages

```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
tqdm
Pillow
```

---

## การติดตั้ง

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm Pillow
```

> 💡 สำหรับ GPU ให้ติดตั้ง PyTorch พร้อม CUDA ตาม [pytorch.org](https://pytorch.org/get-started/locally/)

---

## การเตรียมข้อมูล

จัดโครงสร้างโฟลเดอร์ `Data_text/` ให้แต่ละโฟลเดอร์ย่อยตั้งชื่อตามอักษรที่ต้องการจำแนก เช่น

```
Data_text/
├── ก/
│   ├── image_001.png
│   └── image_002.png
├── ข/
└── ๑/
```

รองรับไฟล์ภาพ `.png`, `.jpg`, `.jpeg`

---

## วิธีการใช้งาน

เปิดไฟล์ `Samlong.ipynb` แล้วรันเซลล์ตามลำดับ

### 1. ตั้งค่าพารามิเตอร์

แก้ไขค่าในเซลล์ **Configuration & Settings**

```python
DATA_PATH = "Data_text"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 224
```

> 💡 สำหรับทดสอบเบื้องต้น ลด `NUM_EPOCHS` เป็น 5–10 เพื่อประหยัดเวลา

### 2. ฝึกโมเดล

**ฝึกโมเดลเดียว** (เร็วกว่า เหมาะสำหรับทดสอบ)

```python
TRAIN_SINGLE_MODEL = True
model_to_train = 'efficientnet_b3'  # เลือก: 'resnet50', 'efficientnet_b3', 'mobilenet_v3'
```

**ฝึกทั้ง 3 โมเดลพร้อมกัน** (รันเซลล์ Train All Models)

### 3. ทดสอบโมเดล

```python
TEST_INDIVIDUAL = True
test_model_name = 'efficientnet_b3'
```

### 4. ทดสอบภาพเดี่ยว

รันเซลล์สุดท้าย แล้วเลือกไฟล์ภาพผ่าน File Dialog ระบบจะแสดงผลการทำนายพร้อม Confidence Score

---

## รายละเอียดโมเดล

| โมเดล | Input Layer | Output Layer | หมายเหตุ |
|---|---|---|---|
| ResNet50 | Conv2d(1, 64, 7×7) | Linear(2048, num_classes) | ปรับ conv1 รับ 1 channel |
| EfficientNet-B3 | Conv2d(1, 40, 3×3) | Linear(1536, num_classes) | ปรับ features[0][0] |
| MobileNetV3-Large | Conv2d(1, 16, 3×3) | Linear(1280, num_classes) | ปรับ features[0][0] |

โมเดลทุกตัวรับ **Grayscale image ขนาด 224×224** และใช้ **Adam optimizer** กับ **CrossEntropyLoss**

---

## Data Augmentation

ใช้ **Triple Augmentation** — แต่ละภาพต้นฉบับจะถูก augment เป็น 3 เวอร์ชัน ทำให้ dataset training ขยายเป็น 3 เท่าโดยอัตโนมัติ

| ระดับ | Rotation | Affine | Perspective | GaussianBlur |
|---|---|---|---|---|
| 🟢 Gentle | ±5° | — | — | σ 0.1–0.3 |
| 🟡 Mild | ±10° | translate 5%, scale 90–110% | — | σ 0.3–0.7 |
| 🔴 Strong | ±20° | translate 10%, scale 85–115%, shear 5° | distortion 0.1 | σ 0.7–1.0 |

**Validation / Test** ไม่มี augmentation — ใช้เฉพาะ resize, pad, normalize

---

## Output ที่ได้

หลังจากฝึกและทดสอบโมเดล ไฟล์ต่อไปนี้จะถูกสร้างในโฟลเดอร์ `outputs_<model_name>/`

| ไฟล์ | คำอธิบาย |
|---|---|
| `<model>_best.pt` | weights ของโมเดลที่ดีที่สุด |
| `training_history.json` | ค่า loss/accuracy ในแต่ละ epoch |
| `class_to_idx.json` | mapping ระหว่างชื่ออักษรกับ index |
| `confusion_matrix_<model>_test.png` | Confusion Matrix บน Test Set |
| `classification_report_<model>.json` | Precision / Recall / F1 รายคลาส |
| `prediction_samples_<model>.png` | ตัวอย่างภาพพร้อมผลการทำนาย |
| `training_history_<model>.png` | กราฟ Loss / Accuracy |

ไฟล์ระดับโปรเจกต์

| ไฟล์ | คำอธิบาย |
|---|---|
| `training_comparison.png` | เปรียบเทียบ training curve ทั้ง 3 โมเดล |
| `training_data_distribution.png` | การกระจายข้อมูลในแต่ละคลาส |
