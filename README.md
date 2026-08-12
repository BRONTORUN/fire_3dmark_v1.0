# Fire 3DMark v1.0

固定单目视角下的尺度无关参数化三维火焰包络拟合标注工具。

本版本不使用真实相机内参，也不恢复米制三维坐标。加载一张火灾图片后，图片视角保持固定；标注员仅调整参数化三维几何体的形状比例、等效三维姿态和图像对齐参数，使几何体的弱透视投影尽量贴合火焰实例分割 Mask。

## V1 技术定义

- **固定观测视角**：背景图片/相机不允许旋转。
- **弱透视投影**：不需要焦距、主点或相机距离。
- **尺度无关形状**：圆锥固定 `h=1`，标 `r/h`；长方体固定 `H=1`，标 `L/H`、`W/H`。
- **等效姿态**：标注的是规范投影空间下可解释二维火焰轮廓的 equivalent/canonical pose，不宣称是真实物理姿态。
- **对齐参数与形状参数分离**：`u_norm/v_norm/scale_norm` 仅用于把归一化模型摆到图片上，不作为物理尺度。

## 已实现

- 打开火灾 RGB 图片。
- 加载二值火焰 Mask；支持自动寻找同级 `masks/` 目录。
- Cone / Cuboid / Invalid 三类标注。
- 圆锥 `r/h`、长方体 `L/H`、`W/H` 比例调节。
- 固定图片视角下的等效 3D 姿态调节；圆锥禁用绕自身主轴的冗余旋转。
- 鼠标拖拽平移形状；鼠标滚轮整体缩放；旋转模式下拖拽改变姿态。
- 根部两点标注，并自动将几何体基准位置吸附到根部中心。
- 3D primitive 的二维 silhouette / wireframe 叠加显示。
- 与火焰 Mask 实时计算 Projection IoU。
- JSON 保存 `Shape + Ratio + Equivalent Pose + Alignment + Root + Fit Quality`。
- 上一张 / 下一张浏览，并自动读取已保存标注。

## 安装

建议 Python 3.10+：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

开发测试依赖：

```bash
pip install -e .[dev]
pytest -q
```

## 运行

```bash
python run.py
```

## 快捷键 / 鼠标

| 操作 | 方式 |
|---|---|
| 打开图片 | `Ctrl+O` |
| 保存标注 | `Ctrl+S` |
| 平移模式 | `W` + 左键拖动 |
| 旋转模式 | `E` + 左键拖动；水平拖动→Rot Z，垂直拖动→Rot X |
| 根部标注 | `Q`，依次点击根部两个端点 |
| 整体投影缩放 | 鼠标滚轮 |
| 上一张 / 下一张 | `A` / `D` |

## 推荐数据目录

```text
dataset/
├─ images/
│  ├─ fire_0001.jpg
│  └─ fire_0002.jpg
├─ masks/
│  ├─ fire_0001.png
│  └─ fire_0002.png
└─ annotations/
   ├─ fire_0001.json
   └─ fire_0002.json
```

当图片位于 `images/` 目录时，软件会优先查找同级 `masks/<stem>.png`，保存标注到同级 `annotations/<stem>.json`。

## JSON 关键字段

```json
{
  "projection": {
    "type": "weak_perspective",
    "camera_intrinsics_required": false
  },
  "shape": {"class": "cone", "valid": true},
  "shape_parameters": {"height_norm": 1.0, "r_h": 0.32},
  "pose": {
    "representation": "canonical_equivalent_pose",
    "euler_xyz_deg": [12.0, 0.0, -8.0],
    "rotation_6d": [1, 0, 0, 0, 1, 0]
  },
  "image_alignment": {
    "u_norm": 0.51,
    "v_norm": 0.74,
    "scale_norm": 0.36
  }
}
```

## 与后续网络的对应关系

- `shape.class` → Shape Classification 标签。
- `shape_parameters` → Shape-specific Parameter Head 的尺寸比例标签。
- `pose.rotation_6d` / `pose.euler_xyz_deg` → 等效姿态标签。
- `root` → 可选 Root Head 标签。
- `image_alignment` → 图像对齐/复核参数，与后续物理尺度因子分离。

后续物理尺寸由独立尺度估计模块提供尺度因子 `s`，再将归一化形状转换为物理尺寸与等效体积。

## 当前边界

V1 故意不做真实相机标定、透视相机反演或真实三维重建。它的目的，是高效、统一地生产“尺度无关参数化火焰包络”训练标签。
