# 深度学习与AI开源项目资源大全

> 适合 Mac Apple Silicon (M1/M2/M3/M4/M5) 原生运行
> 侧重自动驾驶场景：SLAM、BEV、3D感知等

---

## 目录
- [一、深度学习基础资源](#一深度学习基础资源)
- [二、SLAM与视觉里程计](#二slam与视觉里程计)
- [三、深度学习多传感器融合定位](#三深度学习多传感器融合定位)
- [四、BEV感知技术](#四bev感知技术)
- [五、深度估计](#五深度估计)
- [六、3D目标检测](#六3d目标检测)
- [七、点云处理](#七点云处理)
- [八、语义分割](#八语义分割)
- [九、综合学习资源](#九综合学习资源)
- [十、学习路径建议](#十学习路径建议)

---

## 一、深度学习基础资源

### 1. Awesome Deep Learning ⭐
**GitHub**: https://github.com/ChristosChristofidis/awesome-deep-learning

**内容**:
- 深度学习框架与工具
- 经典论文列表
- 教程与课程
- 数据集集合

---

### 2. PyTorch 官方教程
**网站**: https://pytorch.org/tutorials/

**Mac MPS 支持**: ✅

**核心教程**:
```bash
# 验证 MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

**推荐学习路径**:
- 60 Minute Blitz
- Learning PyTorch with Examples
- Transfer Learning Tutorial
- Transformer Tutorial

---

### 3. 动手学深度学习 (Dive into Deep Learning)
**GitHub**: https://github.com/d2l-ai/d2l-zh

**特点**:
- 中文版完整教程
- PyTorch 代码实现
- 可在 Jupyter 中运行
- 覆盖 CNN、RNN、Transformer 等

**安装运行**:
```bash
pip install d2l
pip install torch torchvision
jupyter notebook
```

---

### 4. Stanford CS231n 课程资源
**GitHub**: https://github.com/cs231n/cs231n.github.io

**内容**:
- 卷积神经网络基础
- 计算机视觉任务
- 作业与解答

---

## 二、SLAM与视觉里程计

### 1. ORB-SLAM3 ⭐⭐⭐
**GitHub**: https://github.com/UZ-SLAMLab/ORB_Slam3

**特点**:
- 最完整的视觉 SLAM 系统
- 支持单目、双目、RGB-D
- 支持视觉+IMU融合
- 支持地图重用

**Mac 安装**:
```bash
# 安装依赖
brew install opencv eigen g2o

# 克隆编译
git clone https://github.com/UZ-SLAMLab/ORB_Slam3.git
cd ORB_Slam3
chmod +x build.sh
./build.sh
```

**注意事项**:
- Mac 需要修改部分 CMake 配置
- M系列芯片需要确保 OpenCV 版本兼容

---

### 2. FAST-LIVO2 (激光-惯性-视觉) ⭐⭐⭐
**GitHub**: https://github.com/hku-mars/FAST-LIVO2

**特点**:
- 激光雷达+IMU+视觉融合
- 实时定位与建图
- 高精度里程计

**Mac 兼容性**: ⚠️ 需要 ROS，推荐使用 ROS2

**替代方案**: 使用 Docker 或远程服务器

---

### 3. DSO (Direct Sparse Odometry)
**GitHub**: https://github.com/JakobEngel/dso

**特点**:
- 直接法视觉里程计
- 无需特征提取
- 高精度估计

**Mac 安装**:
```bash
git clone https://github.com/JakobEngel/dso.git
cd dso
mkdir build && cd build
cmake ..
make -j4
```

---

### 4. LSD-SLAM
**GitHub**: https://github.com/tum-vision/lsd_slam

**特点**:
- 大规模直接法 SLAM
- 半稠密重建
- 实时运行

---

### 5. OpenVSLAM
**GitHub**: https://github.com/stella-cv/openvslam

**特点**:
- 模块化设计
- 支持多种相机模型
- 易于扩展

**Mac 安装**:
```bash
brew install opencv g2o suitesparse
git clone https://github.com/stella-cv/openvslam.git
cd openvslam
mkdir build && cd build
cmake .. -DUSE_MPL=ON
make -j4
```

---

### 6. 深度学习 SLAM 资源

#### DeepV2D
**GitHub**: https://github.com/princeton-vl/DeepV2D

**特点**:
- 深度学习视觉里程计
- PyTorch 实现
- 端到端训练

**Mac 支持**: ✅ 纯 PyTorch

---

#### DROID-SLAM ⭐⭐
**GitHub**: https://github.com/deepint/droid-slam

**特点**:
- 深度学习视觉 SLAM
- 最新的神经 SLAM 架构
- 高精度跟踪

**Mac 安装**:
```bash
git clone https://github.com/deepint/droid-slam.git
cd droid-slam
pip install -r requirements.txt
python setup.py install
```

---

## 三、深度学习多传感器融合定位

> 深度学习方法实现的视觉惯性里程计(VIO)、激光惯性里程计(LIO)、多传感器融合定位

### 1. DeepVO ⭐⭐⭐
**GitHub**: https://github.com/ChiWeiHsiao/DeepVO-pytorch

**论文**: DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks (ICRA 2017)

**特点**:
- 端到端视觉里程计
- LSTM + CNN 混合架构
- 无需特征提取
- KITTI 数据集验证

**Mac 支持**: ✅ 纯 PyTorch 实现

**安装运行**:
```bash
git clone https://github.com/ChiWeiHsiao/DeepVO-pytorch.git
cd DeepVO-pytorch
pip install -r requirements.txt

# 下载预训练模型
wget https://drive.google.com/...  # 模型链接见项目README

# 掐理测试
python test.py --model-dir pretrained/ --data-dir KITTI/ --device mps
```

**训练**:
```bash
python train.py --data-dir KITTI_odometry/ \
    --batch-size 16 \
    --epochs 100 \
    --device mps
```

---

### 2. UnDeepVO ⭐⭐⭐
**GitHub**: https://github.com/HorizonAD/UnDeepVO

**论文**: UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning

**特点**:
- 无监督学习方法
- 单目视觉里程计
- 深度估计 + 姿态估计联合训练
- 无需标签数据

**Mac 支持**: ✅ 纯 PyTorch

**安装运行**:
```bash
git clone https://github.com/HorizonAD/UnDeepVO.git
cd UnDeepVO
pip install torch torchvision

# 训练 (无监督)
python train.py --data-path KITTI_raw/ --device mps

# 掐理
python inference.py --pretrained-model model.pth --test-image images/
```

---

### 3. SfMLearner ⭐⭐⭐
**GitHub**: https://github.com/ClementPinard/SfmLearner-Pytorch

**论文**: Unsupervised Learning of Depth and Ego-Motion from Video (CVPR 2017)

**特点**:
- 深度 + 自运动联合估计
- 无监督训练框架
- 视频序列输入
- 可扩展性强

**Mac 支持**: ✅ 纯 PyTorch

**安装运行**:
```bash
git clone https://github.com/ClementPinard/SfmLearner-Pytorch.git
cd SfmLearner-Pytorch
pip install -r requirements.txt

# 下载 KITTI 数据
python download_kitti.py

# 训练
python train.py --dataset-dir KITTI_raw/ --device mps --batch-size 4

# 掐理
python run_inference.py --pretrained-depth depth_model.pth \
    --pretrained-pose pose_model.pth \
    --image-path test_images/
```

---

### 4. DeepV2D ⭐⭐⭐⭐
**GitHub**: https://github.com/princeton-vl/DeepV2D

**论文**: DeepV2D: Video to Depth with Differentiable Structure from Motion (ICCV 2019)

**特点**:
- 视频到深度端到端
- 可微分 SfM
- 深度与相机姿态联合优化
- 高精度重建

**Mac 支持**: ✅ 纯 PyTorch

**安装运行**:
```bash
git clone https://github.com/princeton-vl/DeepV2D.git
cd DeepV2D

# 安装依赖
pip install torch torchvision numpy scipy

# 下载预训练模型
python download_models.py

# 运行推理
python run.py --input-video video.mp4 --device mps
```

---

### 5. TartVO ⭐⭐⭐
**GitHub**: https://github.com/mli0603/TartVO

**论文**: TartVO: Towards More General Visual Odometry with Deep Learning

**特点**:
- 更泛化的视觉里程计
- 多场景适应
- Transformer 架构
- 最新研究成果

**Mac 支持**: ✅ 纯 PyTorch

---

### 6. DROID-SLAM ⭐⭐⭐⭐⭐
**GitHub**: https://github.com/princeton-vl/DROID-SLAM

**论文**: DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras (NeurIPS 2021)

**特点**:
- 最新深度学习 SLAM
- 支持单目、双目、RGB-D
- 高精度跟踪
- 端到端训练
- 鲁棒性强

**Mac 支持**: ✅ 纯 PyTorch (需要 CUDA 编译部分可替换为 CPU)

**安装运行**:
```bash
git clone https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM

# 安装 PyTorch (MPS 版本)
pip install torch torchvision torchaudio

# 安装其他依赖
pip install scipy matplotlib
pip install evo  # 用于评估

# 运行推理
python evaluation.py --model-path models/droid.pth \
    --imagedir images/ \
    --calib calib.txt \
    --device mps
```

**注意**: Mac 上部分 CUDA 操作需要设置回退:
```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

---

### 7. DeepFactors ⭐⭐⭐
**GitHub**: https://github.com/jcarricart-git/DeepFactors

**特点**:
- 深度学习 SLAM 框架
- 因子图优化结合深度学习
- 多传感器融合潜力

**Mac 支持**: ✅ PyTorch

---

### 8. VI-DeepVO (视觉惯性里程计) ⭐⭐⭐
**GitHub**: https://github.com/SIMD-S/VI-DeepVO

**论文**: Visual-Inertial Odometry with Deep Learning

**特点**:
- 视觉 + IMU 融合
- LSTM 网络处理时序
- 端到端学习
- 高精度定位

**Mac 支持**: ✅ 纯 PyTorch

**安装运行**:
```bash
git clone https://github.com/SIMD-S/VI-DeepVO.git
cd VI-DeepVO

pip install -r requirements.txt

# 训练 (需要 IMU 数据)
python train.py --data-dir EuRoC_MAV/ --device mps
```

---

### 9. LIO-Net (激光惯性里程计) ⭐⭐⭐
**GitHub**: https://github.com/lijx10/LIO-Net

**论文**: LIO-Net: Lidar-Inertial Odometry using Learning-based Point Registration

**特点**:
- 激光雷达 + IMU 融合
- 学习型点云配准
- 实时性能
- 高精度定位

**Mac 支持**: ✅ PyTorch (点云处理部分)

**安装运行**:
```bash
git clone https://github.com/lijx10/LIO-Net.git
cd LIO-Net

pip install torch torchvision
pip install open3d  # 点云可视化

# 测试
python test.py --data-path KITTI_lidar/ --device mps
```

---

### 10. LO-Net (激光里程计) ⭐⭐⭐
**GitHub**: https://github.com/hyangwinter/LO-Net

**论文**: LO-Net: Deep Learning for Lidar Odometry

**特点**:
- 端到端激光里程计
- 点云配准神经网络
- 无需人工设计特征
- KITTI 验证

**Mac 支持**: ✅ PyTorch

---

### 11. DeepLO ⭐⭐⭐
**GitHub**: https://github.com/hpnguyen/DeepLO

**特点**:
- 深度学习激光里程计
- 点云分割 + 配准
- 端到端框架

**Mac 支持**: ✅ PyTorch

---

### 12. Camera-LiDAR Fusion 项目

#### ContFuse (Continuous Fusion)
**GitHub**: https://github.com/Jjieun/ContFuse

**论文**: Continuous 3D Multi-Modal Sensor Fusion

**特点**:
- 相机 + 激光雷达融合
- 连续融合策略
- 深度估计辅助定位

**Mac 支持**: ✅ PyTorch

---

#### EPNet
**GitHub**: https://github.com/EPNet-git/EPNet

**论文**: EPNet: Enhancing Point Features for 3D Object Detection and Localization

**特点**:
- 点特征增强
- 多传感器融合
- 检测 + 定位联合

**Mac 支持**: ✅ PyTorch

---

### 13. TransFuser (多传感器Transformer融合) ⭐⭐⭐⭐
**GitHub**: https://github.com/autonomousvision/transfuser

**论文**: TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving

**特点**:
- Transformer 多传感器融合
- 相机 + LiDAR + GPS 融合
- CARLA 仿真验证
- 端到端驾驶

**Mac 支持**: ✅ 纯 PyTorch (CARLA 需要单独安装)

**安装运行**:
```bash
git clone https://github.com/autonomousvision/transfuser.git
cd transfuser

pip install -r requirements.txt

# 训练
python train.py --config config.yaml --device mps

# CARLA 仿真测试 (需要安装 CARLA)
python run_agent.py --model-path model.pth --device mps
```

---

### 14. GeoNet ⭐⭐⭐
**GitHub**: https://github.com/yzcjtr/GeoNet

**论文**: GeoNet: Unsupervised Learning of Geometry from Video

**特点**:
- 深度 + 光流 + 相机运动
- 无监督联合学习
- 多任务统一框架

**Mac 支持**: ✅ PyTorch

**安装运行**:
```bash
git clone https://github.com/yzcjtr/GeoNet.git
cd GeoNet

pip install -r requirements.txt

# 训练
python train.py --dataset KITTI_raw --device mps
```

---

### 15. Multi-Task Sensor Fusion 资源

#### Awesome Sensor Fusion ⭐⭐⭐⭐
**GitHub**: https://github.com/AaronKXXX/Awesome-Sensor-Fusion

**内容**:
- 传感器融合论文列表
- 多模态融合方法
- 数据集汇总
- 开源项目索引

---

#### Awesome Deep Learning for Localization
**GitHub**: https://github.com/YuhangQao/Awesome-Deep-Learning-Localization

**内容**:
- 定位与里程计论文
- SLAM 深度学习方法
- 融合技术综述

---

### 16. 传统融合方法对比学习

| 方法 | 传感器 | 深度学习 | Mac支持 | 精度 | 实时性 |
|------|--------|----------|---------|------|--------|
| ORB-SLAM3 | Cam+IMU | ❌ | ⚠️ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| VINS-Mono | Cam+IMU | ❌ | ⚠️ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| FAST-LIO2 | LiDAR+IMU | ❌ | ⚠️ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| DROID-SLAM | Camera | ✅ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| VI-DeepVO | Cam+IMU | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| LIO-Net | LiDAR+IMU | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| TransFuser | Multi | ✅ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

### 17. 数据集资源

#### 视觉惯性里程计数据集

| 数据集 | 描述 | 大小 | 链接 |
|--------|------|------|------|
| EuRoC MAV | VIO 标准数据集 | 22GB | https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets |
| KITTI Odometry | 视觉里程计基准 | 12GB | http://www.cvlibs.net/datasets/kitti/eval_odometry.php |
| TUM VI | 视觉惯性数据 | 50GB | https://vision.in.tum.de/data/datasets/visual-inertial-dataset |

#### 激光惯性里程计数据集

| 数据集 | 描述 | 大小 | 链接 |
|--------|------|------|------|
| KITTI LiDAR | 激光里程计基准 | 12GB | http://www.cvlibs.net/datasets/kitti/ |
| Oxford Radar | 雷达数据集 | 130GB | https://oxford-robotics-institute.github.io/radar-robotcar-dataset/ |
| MulRan | 多雷达数据集 | 80GB | https://sites.google.com/view/mulran-dataset/ |

---

## 四、BEV感知技术

### 1. BEVFormer ⭐⭐⭐⭐⭐
**GitHub**: https://github.com/fundamentalvision/BEVFormer

**论文**: CVPR 2022

**特点**:
- 多摄像头 BEV 特征提取
- Transformer 架构
- 时序融合
- nuScenes 数据集 SOTA

**Mac 安装**:
```bash
# 创建环境
conda create -n bevformer python=3.8 -y
conda activate bevformer

# 安装 PyTorch MPS
pip install torch torchvision torchaudio

# 安装 MMDetection3D
pip install mmdet mmcv mmdet3d

# 克隆项目
git clone https://github.com/fundamentalvision/BEVFormer.git
cd BEVFormer
pip install -r requirements.txt
```

**配置修改**:
```python
# configs/bevformer_base.py
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            use_mps=True  # 启用 MPS 加速
        )
    )
)
```

---

### 2. BEVDepth
**GitHub**: https://github.com/Megvii-BaseDetection/BEVDepth

**特点**:
- 显式深度估计
- 多摄像头融合
- 高效推理

**Mac 支持**: ✅ 基于 MMDetection3D

---

### 3. BEVDet
**GitHub**: https://github.com/HuangJunJie2017/BEVDet

**特点**:
- 高效 BEV 检测
- 实时推理
- 轻量化设计

---

### 4. PETR (Position Embedding Transformation)
**GitHub**: https://github.com/megvii-research/PETR

**特点**:
- 位置嵌入变换
- 3D 目标检测
- Transformer 架构

---

### 5. BEV 学习教程 ⭐

#### Awesome BEV Perception
**GitHub**: https://github.com/chaytonmin/Awesome-BEV-Perception

**内容**:
- BEV 论文列表
- 开源项目汇总
- 学习路线

---

#### BEV 教程笔记
**GitHub**: https://github.com/ChongjianGE/BEV_Perception_Survey

**特点**:
- 全面综述
- 方法分类
- 未来方向

---

## 五、深度估计

### 1. MiDaS (Intel) ⭐⭐⭐
**GitHub**: https://github.com/isl-org/MiDaS

**特点**:
- 单目深度估计
- 多数据集混合训练
- 高泛化能力
- 预训练模型可用

**Mac 运行**:
```bash
pip install timm

# 下载模型
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt

# 推理
python run.py --model_type dpt_large --input_path input/ --output_path output/
```

**PyTorch MPS 加速**:
```python
import torch
from midas.dpt_depth import DPTDepthModel

model = DPTDepthModel(path="dpt_large_384.pt")
device = torch.device("mps")
model = model.to(device)
```

---

### 2. DPT (Dense Prediction Transformer)
**GitHub**: https://github.com/isl-org/DPT

**特点**:
- Vision Transformer 骨干
- 高精度深度估计
- 语义分割版本

---

### 3. AdaBins
**GitHub**: https://github.com/shariqfarooq123/AdaBins

**特点**:
- 自适应分箱策略
- KITTI/NYU SOTA
- 轻量化设计

---

### 4. Depth-Anything ⭐⭐⭐
**GitHub**: https://github.com/LiheYoung/Depth-Anything

**特点**:
- 最新深度估计模型
- 极强泛化能力
- 零样本迁移

**Mac 运行**:
```bash
pip install depth-anything

# 使用示例
from depth_anything import DepthAnything

model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
device = torch.device("mps")
model = model.to(device)
```

---

### 5. Marigold
**GitHub**: https://github.com/prs-eth/Marigold

**特点**:
- 基于扩散模型
- 零样本深度估计
- 高质量预测

---

## 六、3D目标检测

### 1. PointPillars ⭐⭐⭐
**GitHub**: https://github.com/nutonomy/second.pytorch

**特点**:
- 点云柱状表示
- 实时 3D 检测
- KITTI 基准

**Mac 运行**:
```bash
git clone https://github.com/nutonomy/second.pytorch.git
cd second.pytorch
pip install -r requirements.txt

# 下载预训练模型测试
python ./second/pytorch/inference.py --ckpt_path pretrained_models/pointpillars.pth
```

---

### 2. PV-RCNN
**GitHub**: https://github.com/open-mmlab/mmdetection3d

**特点**:
- 点-体素融合
- 高精度检测
- MMDetection3D 实现

---

### 3. CenterPoint
**GitHub**: https://github.com/tianweiy/CenterPoint

**特点**:
- 中心点检测
- 时序融合
- nuScenes SOTA

---

### 4. 3D 检测教程

#### MMDetection3D ⭐⭐⭐
**GitHub**: https://github.com/open-mmlab/mmdetection3d

**特点**:
- 统一 3D 检测框架
- 多数据集支持
- 丰富模型库

**Mac 安装**:
```bash
pip install mmdet mmcv-full
pip install mmdet3d

# 或从源码安装
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .
```

---

## 七、点云处理

### 1. PointNet / PointNet++ ⭐⭐⭐
**GitHub**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

**特点**:
- 点云处理经典网络
- 分类与分割
- 理解点云深度学习基础

**Mac 运行**:
```bash
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
cd Pointnet_Pointnet2_pytorch
pip install -r requirements.txt

# 训练分类模型
python train_classification.py --model pointnet2_cls_ssg
```

---

### 2. PointTransformer
**GitHub**: https://github.com/POSTECH-CVLab/point-transformer

**特点**:
- Transformer 架构点云处理
- 自注意力机制
- 高精度分割

---

### 3. PointNet++ PyTorch 实现
**GitHub**: https://github.com/erikwijmans/Pointnet2_PyTorch

**特点**:
- 纯 PyTorch 实现
- 易于理解和修改
- 支持 Mac MPS

---

### 4. Open3D ⭐⭐
**GitHub**: https://github.com/isl-org/Open3D

**特点**:
- 3D 数据处理库
- 点云可视化
- 网格处理

**Mac 安装**:
```bash
pip install open3d

# 使用示例
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
```

---

### 5. Awesome Point Cloud Analysis
**GitHub**: https://github.com/Yochengliu/awesome-point-cloud-analysis

**内容**:
- 点云论文列表
- 数据集汇总
- 开源项目索引

---

## 八、语义分割

### 1. SegFormer ⭐⭐⭐
**GitHub**: https://github.com/NVlabs/SegFormer

**特点**:
- Transformer 架构
- 高效语义分割
- 道路场景分割

**Mac 运行**:
```bash
pip install mmsegmentation

# 使用预训练模型
from mmseg.apis import inference_segmentor, init_segmentor
config_file = 'configs/segformer/segformer_b5_cityscapes.py'
checkpoint_file = 'checkpoints/segformer_b5_cityscapes.pth'
model = init_segmentor(config_file, checkpoint_file, device='mps')
result = inference_segmentor(model, 'road_image.jpg')
```

---

### 2. DDRNet (实时分割)
**GitHub**: https://github.com/chenjun2hao/DDRNet.pytorch

**特点**:
- 实时语义分割
- 轻量化设计
- 道路场景优化

---

### 3. BiSeNet V2
**GitHub**: https://github.com/CoinCheung/BiSeNet

**特点**:
- 双分支网络
- 实时高精度
- Cityscapes 优化

---

### 4. MMSegmentation ⭐⭐⭐
**GitHub**: https://github.com/open-mmlab/mmsegmentation

**特点**:
- 统一分割框架
- 丰富模型库
- 易于训练和部署

**Mac 安装**:
```bash
pip install mmsegmentation

# 训练自定义数据集
python tools/train.py configs/segformer/segformer_b2_cityscapes.py --device mps
```

---

## 九、综合学习资源

### 1. Awesome Autonomous Driving ⭐⭐⭐⭐⭐
**GitHub**: https://github.com/autonomousdriving/awesome-autonomous-driving

**内容**:
- 自动驾驶论文列表
- 开源项目汇总
- 数据集集合
- 学习路线

---

### 2. Awesome Deep Learning Papers
**GitHub**: https://github.com/terryum/awesome-deep-learning-papers

**内容**:
- 经典论文阅读列表
- 按领域分类
- 有影响力论文

---

### 3. Awesome Computer Vision
**GitHub**: https://github.com/jbhuang0604/awesome-computer-vision

**内容**:
- 计算机视觉资源
- 教程与课程
- 数据集与工具

---

### 4. 深度学习面试资源
**GitHub**: https://github.com/elias-abboud/awesome-deep-learning-interview-questions

**内容**:
- 面试问题集
- 概念解释
- 实战案例

---

### 5. Transformer 教程资源
**GitHub**: https://github.com/graykode/nlp-tutorial

**内容**:
- Transformer 实现
- Attention 可视化
- 从零开始教程

---

### 6. PyTorch 深度学习实战
**GitHub**: https://github.com/Atcold/pytorch-Deep-Learning

**内容**:
- NYU 课程代码
- 完整实现
- 详细注释

---

### 7. 模型部署资源

#### ONNX Runtime
**GitHub**: https://github.com/microsoft/onnxruntime

**Mac 支持**: ✅

```bash
pip install onnxruntime

# 转换模型
import torch.onnx
torch.onnx.export(model, dummy_input, "model.onnx")
```

#### TensorRT (替代方案: CoreML)
**CoreML**: Apple 官方推理框架

```bash
pip install coremltools

# 转换 PyTorch 模型
import coremltools as ct
traced_model = torch.jit.trace(model, example_input)
mlmodel = ct.convert(traced_model, inputs=[...])
```

---

## 十、学习路径建议

### 阶段一：基础入门 (1-2个月)

**深度学习基础**:
1. 学习 PyTorch 基础操作
2. 完成 CS231n 课程
3. 阅读《动手学深度学习》

**实践项目**:
```bash
# YOLOv8 目标检测
pip install ultralytics
yolo predict model=yolov8s.pt source=test.jpg device=mps
```

---

### 阶段二：视觉感知 (2-3个月)

**核心任务**:
1. 目标检测 (YOLOv8)
2. 语义分割 (SegFormer)
3. 深度估计 (Depth-Anything)

**推荐项目**:
```bash
# 语义分割
pip install mmsegmentation
python tools/train.py configs/segformer/segformer_b2_cityscapes.py --device mps

# 深度估计
pip install depth-anything
python infer.py --encoder vitl --img-path input/ --outdir output/
```

---

### 阶段三：3D感知 (2-3个月)

**核心内容**:
1. 点云处理 (PointNet++)
2. 3D 检测 (PointPillars)
3. BEV 感知 (BEVFormer)

**实践项目**:
```bash
# 点云分类
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
python train_classification.py --model pointnet2_cls_ssg

# BEV 检测
git clone https://github.com/fundamentalvision/BEVFormer.git
python tools/train.py configs/bevformer_base.py --device mps
```

---

### 阶段四：SLAM与融合 (2-3个月)

**核心内容**:
1. 传统 SLAM (ORB-SLAM3)
2. 神经 SLAM (DROID-SLAM)
3. 多传感器融合

**实践项目**:
```bash
# ORB-SLAM3
git clone https://github.com/UZ-SLAMLab/ORB_Slam3.git
./build.sh

# DROID-SLAM
pip install droid-slam
python demo.py --imagedir=images/ --calib=calib.txt
```

---

### 阶段五：综合项目 (1-2个月)

**推荐项目**:
1. 端到端自动驾驶 (CARLA 仿真)
2. 多任务感知系统
3. 实时 SLAM 系统

---

## 附录：常用命令速查

### PyTorch MPS 验证
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"MPS可用: {torch.backends.mps.is_available()}")
print(f"MPS已构建: {torch.backends.mps.is_built()}")

# 使用 MPS
device = torch.device("mps")
model = model.to(device)
tensor = tensor.to(device)
```

### 常见问题解决

**内存不足**:
```python
# 减小 batch size
batch_size = 4  # 或更小

# 使用混合精度
scaler = torch.cuda.amp.GradScaler()  # MPS 支持有限
```

**MPS 不支持某些操作**:
```python
# 设置回退到 CPU
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

**模型加载慢**:
```python
# 使用较小的模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
```

---

## 参考数据集

| 数据集 | 任务 | 大小 | 链接 |
|--------|------|------|------|
| KITTI | 3D检测/SLAM | 12GB | http://www.cvlibs.net/datasets/kitti/ |
| nuScenes | BEV/3D检测 | 350GB | https://www.nuscenes.org/ |
| Waymo Open | 3D检测 | 1.8TB | https://waymo.com/open/ |
| Cityscapes | 语义分割 | 11GB | https://www.cityscapes-dataset.com/ |
| TuSimple | 车道线检测 | 10GB | https://github.com/TuSimple/tusimple-benchmark |
| COCO | 2D检测 | 118GB | https://cocodataset.org/ |
| NYU Depth V2 | 深度估计 | 428GB | https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html |

---

*更新时间: 2026年5月*

**Sources**:
- [DeepVO-pytorch](https://github.com/ChiWeiHsiao/DeepVO-pytorch)
- [UnDeepVO](https://github.com/HorizonAD/UnDeepVO)
- [SfMLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch)
- [DeepV2D](https://github.com/princeton-vl/DeepV2D)
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [VI-DeepVO](https://github.com/SIMD-S/VI-DeepVO)
- [LIO-Net](https://github.com/lijx10/LIO-Net)
- [LO-Net](https://github.com/hyangwinter/LO-Net)
- [TransFuser](https://github.com/autonomousvision/transfuser)
- [GeoNet](https://github.com/yzcjtr/GeoNet)
- [Awesome Sensor Fusion](https://github.com/AaronKXXX/Awesome-Sensor-Fusion)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_Slam3)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MiDaS](https://github.com/isl-org/MiDaS)
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [Open3D](https://github.com/isl-org/Open3D)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [Awesome Autonomous Driving](https://github.com/autonomousdriving/awesome-autonomous-driving)
- [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)