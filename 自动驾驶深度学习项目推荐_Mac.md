# Mac M5 深度学习自动驾驶实战项目

> 适合 Mac Apple Silicon (M1/M2/M3/M4/M5) 原生运行，无需虚拟机

---

## 目录
- [环境配置](#环境配置)
- [推荐项目](#推荐项目)
- [项目对比](#项目对比)
- [快速开始](#快速开始)

---

## 环境配置

### 1. 安装 Miniforge (推荐替代 Anaconda)
```bash
brew install miniforge
```

### 2. 创建 ARM64 环境
```bash
conda create -n auto_drive python=3.10
conda activate auto_drive
```

### 3. 安装 PyTorch MPS 版本
```bash
pip install torch torchvision torchaudio
```

### 4. 验证 MPS 加速可用
```bash
python -c "import torch; print(f'MPS可用: {torch.backends.mps.is_available()}')"
```

---

## 推荐项目

### 1. YOLOv8 车辆检测与跟踪 ⭐推荐入门

**GitHub**: https://github.com/ultralytics/ultralytics

**架构**: YOLOv8 (2023年最新)

**特点**:
- 官方支持 Apple Silicon MPS 加速
- 开箱即用，文档完善
- 支持目标检测、实例分割、姿态估计
- 预训练模型丰富

**安装**:
```bash
pip install ultralytics
```

**快速测试**:
```bash
# 使用 MPS 加速推理
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' device=mps

# 查看检测结果
yolo predict model=yolov8s.pt source=path/to/your/video.mp4 device=mps show=True
```

**训练自定义数据**:
```bash
# 使用 MPS 加速训练
yolo detect train model=yolov8s.pt data=your_data.yaml epochs=100 device=mps imgsz=640

# 导出模型
yolo export model=yolov8s.pt format=onnx
```

**自动驾驶应用**:
- 车辆检测
- 行人检测
- 交通标志识别
- 交通灯状态识别

---

### 2. CLRNet 车道线检测

**GitHub**: https://github.com/Turoad/CLRNet

**架构**: Transformer + CNN 混合 (CVPR 2024)

**特点**:
- 最新车道线检测架构
- 支持 TuSimple、CULane 数据集
- 高精度实时检测

**安装**:
```bash
git clone https://github.com/Turoad/CLRNet.git
cd CLRNet
pip install -r requirements.txt
```

**训练**:
```bash
# TuSimple 数据集
python train.py --config configs/clrnet_tusimple.py --device mps

# CULane 数据集
python train.py --config configs/clrnet_culane.py --device mps
```

**推理**:
```bash
python demo.py --config configs/clrnet_tusimple.py \
    --load_from work_dirs/clrnet/best.pth \
    --img_path your_test_image.jpg \
    --device mps
```

---

### 3. YOLOv8 + DeepSORT 目标跟踪

**GitHub**: https://github.com/mikel-brostrom/YOLO_tracking

**架构**: YOLOv8 + DeepSORT + ByteTrack

**特点**:
- 实时目标检测 + 跟踪
- 支持多种跟踪算法
- 可视化轨迹绘制

**安装**:
```bash
pip install ultralytics
git clone https://github.com/mikel-brostrom/YOLO_tracking.git
cd YOLO_tracking
pip install -r requirements.txt
```

**运行跟踪**:
```bash
# 基础跟踪
python tracking/track.py --source video.mp4 --device mps --yolo-model yolov8s.pt

# 使用 ByteTrack
python tracking/track.py --source video.mp4 --device mps --tracking-method bytetrack

# 保存结果
python tracking/track.py --source video.mp4 --device mps --save
```

**应用场景**:
- 车辆计数
- 轨迹分析
- 异常行为检测

---

### 4. LaneNet 车道线检测

**GitHub**: https://github.com/MaybeShewill-CV/lanenet-lane-detection

**架构**: ENet + 聚类后处理

**特点**:
- 经典车道线检测架构
- 易于理解和修改
- 适合学习基础概念

**安装**:
```bash
git clone https://github.com/MaybeShewill-CV/lanenet-lane-detection.git
cd lanenet-lane-detection
pip install -r requirements.txt
```

**训练**:
```bash
python tools/train_lanenet.py \
    --dataset_dir /path/to/tusimple \
    --net_type bisenetv2 \
    --device mps
```

---

### 5. Traffic Sign Detection 交通标志识别

**数据集**: GTSDB (German Traffic Sign Detection Benchmark)

**模型**: YOLOv8

**准备数据**:
```yaml
# gtsdb.yaml
path: /path/to/gtsdb
train: images/train
val: images/val

nc: 43  # 类别数量
names: ['speed_limit_20', 'speed_limit_30', ...]  # 交通标志类别
```

**训练**:
```bash
yolo detect train model=yolov8s.pt data=gtsdb.yaml epochs=50 device=mps imgsz=640
```

**推理**:
```bash
yolo predict model=best.pt source=test_images/ device=mps
```

---

## 项目对比

| 项目 | 架构年份 | Mac原生 | 难度 | 实用性 | 数据集 |
|------|---------|---------|------|--------|--------|
| YOLOv8 | 2023 | ✅ | ⭐⭐ | ⭐⭐⭐⭐⭐ | COCO, 自定义 |
| CLRNet | 2024 | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐ | TuSimple, CULane |
| YOLO+DeepSORT | 2023 | ✅ | ⭐⭐ | ⭐⭐⭐⭐ | 任意视频 |
| LaneNet | 2018 | ✅ | ⭐⭐ | ⭐⭐⭐ | TuSimple |

---

## 快速开始

### 最简单的入门方案 (5分钟)

```bash
# 1. 创建环境
conda create -n auto_drive python=3.10 -y
conda activate auto_drive

# 2. 安装 YOLOv8
pip install ultralytics

# 3. 测试 MPS 加速
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' device=mps

# 4. 下载示例视频测试 (或使用自己的视频)
yolo predict model=yolov8s.pt source=your_video.mp4 device=mps show=True
```

### 完整学习路径

**第1周: 目标检测基础**
- 学习 YOLOv8 基本使用
- 在自定义数据集上训练
- 模型评估与优化

**第2周: 目标跟踪**
- 学习 DeepSORT 原理
- 实现车辆跟踪
- 轨迹分析与可视化

**第3周: 车道线检测**
- 学习 CLRNet 架构
- 在 TuSimple 数据集训练
- 实时车道线检测

**第4周: 综合项目**
- 多任务融合
- 构建完整感知系统
- 性能优化

---

## 常用数据集

### 目标检测
| 数据集 | 描述 | 大小 | 下载 |
|--------|------|------|------|
| COCO | 通用目标检测 | 118GB | 自动下载 |
| KITTI | 自动驾驶场景 | 12GB | http://www.cvlibs.net/datasets/kitti/ |
| BDD100K | 多样化驾驶场景 | 166GB | https://bdd-data.berkeley.edu/ |

### 车道线检测
| 数据集 | 描述 | 大小 | 下载 |
|--------|------|------|------|
| TuSimple | 高速公路车道线 | 10GB | https://github.com/TuSimple/tusimple-benchmark |
| CULane | 复杂场景车道线 | 25GB | https://xingangpan.github.io/projects/CULane.html |

---

## 性能优化建议

### MPS 加速配置
```python
import torch

# 检查 MPS 可用性
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS 加速")
else:
    device = torch.device("cpu")
    print("使用 CPU")

# 设置环境变量优化
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### 内存优化
```bash
# 减小 batch size
yolo train model=yolov8s.pt data=data.yaml batch=8 device=mps

# 减小图像尺寸
yolo train model=yolov8s.pt data=data.yaml imgsz=416 device=mps

# 使用混合精度训练
yolo train model=yolov8s.pt data=data.yaml amp=True device=mps
```

---

## 参考资料

- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [CLRNet 论文](https://arxiv.org/abs/2203.10350)
- [DeepSORT 论文](https://arxiv.org/abs/1703.07402)

---

## 故障排除

### MPS 不可用
```bash
# 检查 PyTorch 版本 (需要 2.0+)
python -c "import torch; print(torch.__version__)"

# 更新 PyTorch
pip install --upgrade torch torchvision torchaudio
```

### 内存不足
```bash
# 减小模型大小
yolov8n.pt  # nano - 最小
yolov8s.pt  # small
yolov8m.pt  # medium
yolov8l.pt  # large
yolov8x.pt  # xlarge - 最大
```

### 推理速度慢
```bash
# 使用更小的模型
yolo predict model=yolov8n.pt device=mps

# 减小输入尺寸
yolo predict model=yolov8s.pt imgsz=320 device=mps
```

---

*更新时间: 2025年5月*