from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSplitter, QToolBar, QVBoxLayout, QWidget,
)

from .canvas import AnnotationCanvas
from .state import AnnotationState


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fire 3DMark v1.0 — 固定视角三维火焰包络标注")
        self.resize(1280, 820)
        self.canvas = AnnotationCanvas(self)
        self.state = self.canvas.state
        self._image_files: list[Path] = []
        self._image_index = -1
        self._build_toolbar()
        self._build_ui()
        self.canvas.stateEdited.connect(self._on_canvas_state_edited)
        self.canvas.rootEdited.connect(self._update_root_label)
        self.statusBar().showMessage("打开一张火灾图片开始")

    def _build_toolbar(self) -> None:
        tb = QToolBar("主工具栏", self); tb.setMovable(False); self.addToolBar(tb)
        for text, shortcut, callback in [
            ("打开图片", QKeySequence.Open, self.open_image),
            ("加载Mask", None, self.open_mask),
            ("保存标注", QKeySequence.Save, self.save_annotation),
        ]:
            a = QAction(text, self)
            if shortcut: a.setShortcut(shortcut)
            a.triggered.connect(callback); tb.addAction(a)
        tb.addSeparator()
        prev_a = QAction("上一张", self); prev_a.setShortcut("A"); prev_a.triggered.connect(lambda: self.navigate_image(-1)); tb.addAction(prev_a)
        next_a = QAction("下一张", self); next_a.setShortcut("D"); next_a.triggered.connect(lambda: self.navigate_image(1)); tb.addAction(next_a)
        tb.addSeparator()
        grp = QActionGroup(self); grp.setExclusive(True)
        for text, mode, key in [("平移", "translate", "W"), ("旋转", "rotate", "E"), ("根部", "root", "Q")]:
            a = QAction(text, self); a.setCheckable(True); a.setShortcut(key)
            a.triggered.connect(lambda checked, m=mode: self.canvas.set_mode(m)); grp.addAction(a); tb.addAction(a)
            if mode == "translate": a.setChecked(True)
        tb.addSeparator()
        reset = QAction("复位", self); reset.triggered.connect(self.reset_state); tb.addAction(reset)
        auto = QAction("Mask自动初始化", self); auto.triggered.connect(self.auto_initialize); tb.addAction(auto)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self); splitter.addWidget(self.canvas)
        panel = QWidget(); layout = QVBoxLayout(panel); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(10)

        g = QGroupBox("形状类型"); gl = QVBoxLayout(g)
        self.shape_combo = QComboBox(); self.shape_combo.addItem("圆锥 Cone", "cone"); self.shape_combo.addItem("长方体 Cuboid", "cuboid"); self.shape_combo.addItem("不适用 Invalid", "invalid")
        self.shape_combo.currentIndexChanged.connect(self._controls_to_state); gl.addWidget(self.shape_combo); layout.addWidget(g)

        g = QGroupBox("尺度无关形状参数"); form = QFormLayout(g)
        self.r_h = self._spin(0.02, 2.0, 0.01, 0.30, 4); self.l_h = self._spin(0.02, 3.0, 0.01, 0.60, 4); self.w_h = self._spin(0.02, 3.0, 0.01, 0.35, 4)
        form.addRow("r / h", self.r_h); form.addRow("L / H", self.l_h); form.addRow("W / H", self.w_h); layout.addWidget(g)

        g = QGroupBox("等效三维姿态（图片视角固定）"); form = QFormLayout(g)
        self.rot_x = self._spin(-180, 180, 1, 0, 2, "°"); self.rot_y = self._spin(-180, 180, 1, 0, 2, "°"); self.rot_z = self._spin(-180, 180, 1, 0, 2, "°")
        form.addRow("Rot X", self.rot_x); form.addRow("Rot Y", self.rot_y); form.addRow("Rot Z", self.rot_z); layout.addWidget(g)

        g = QGroupBox("图像对齐参数（非物理尺度）"); form = QFormLayout(g)
        self.u_norm = self._spin(-1, 2, .005, .50, 4); self.v_norm = self._spin(-1, 2, .005, .80, 4); self.scale_norm = self._spin(.01, 5, .01, .50, 4)
        form.addRow("U norm", self.u_norm); form.addRow("V norm", self.v_norm); form.addRow("Scale norm", self.scale_norm); layout.addWidget(g)

        g = QGroupBox("显示与拟合"); form = QFormLayout(g)
        self.show_mask = QCheckBox("显示实例分割Mask"); self.show_mask.setChecked(True); self.show_mask.toggled.connect(self._toggle_mask)
        self.show_proj = QCheckBox("显示3D投影"); self.show_proj.setChecked(True); self.show_proj.toggled.connect(self._toggle_projection)
        self.iou_label = QLabel("—")
        self.grade_combo = QComboBox(); [self.grade_combo.addItem(x) for x in ["unrated", "good", "acceptable", "poor", "invalid"]]; self.grade_combo.currentTextChanged.connect(self._controls_to_state)
        form.addRow(self.show_mask); form.addRow(self.show_proj); form.addRow("Projection IoU", self.iou_label); form.addRow("Fit grade", self.grade_combo); layout.addWidget(g)

        g = QGroupBox("火焰根部"); gl = QVBoxLayout(g)
        self.root_label = QLabel("未标注。按 Q 后依次点击根部两个端点。"); self.root_label.setWordWrap(True)
        clear = QPushButton("清除根部"); clear.clicked.connect(self.clear_root); gl.addWidget(self.root_label); gl.addWidget(clear); layout.addWidget(g)

        hint = QLabel("W：平移；E：旋转；滚轮：整体缩放；Q：根部两点标注。\n相机/图片视角始终固定，所有操作只改变3D形状或图像对齐参数。")
        hint.setWordWrap(True); hint.setStyleSheet("color:#666;"); layout.addWidget(hint); layout.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); scroll.setMinimumWidth(330)
        splitter.addWidget(scroll); splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 0); self.setCentralWidget(splitter)

        for w in [self.r_h, self.l_h, self.w_h, self.rot_x, self.rot_y, self.rot_z, self.u_norm, self.v_norm, self.scale_norm]:
            w.valueChanged.connect(self._controls_to_state)
        self._state_to_controls()

    @staticmethod
    def _spin(lo, hi, step, value, decimals, suffix="") -> QDoubleSpinBox:
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setSingleStep(step); w.setDecimals(decimals); w.setValue(value)
        if suffix: w.setSuffix(suffix)
        return w

    def _controls_to_state(self, *_) -> None:
        self.state.shape_type = self.shape_combo.currentData(); self.state.r_h = self.r_h.value(); self.state.l_h = self.l_h.value(); self.state.w_h = self.w_h.value()
        self.state.rot_x_deg = self.rot_x.value(); self.state.rot_y_deg = self.rot_y.value(); self.state.rot_z_deg = self.rot_z.value()
        if self.state.shape_type == "cone": self.state.rot_y_deg = 0.0
        self.state.u_norm = self.u_norm.value(); self.state.v_norm = self.v_norm.value(); self.state.scale_norm = self.scale_norm.value(); self.state.fit_grade = self.grade_combo.currentText()
        self.state.clamp(); self.canvas.update(); self._update_visibility(); self._update_metrics()

    def _state_to_controls(self) -> None:
        widgets = [self.shape_combo, self.r_h, self.l_h, self.w_h, self.rot_x, self.rot_y, self.rot_z, self.u_norm, self.v_norm, self.scale_norm, self.grade_combo]
        blockers = [QSignalBlocker(w) for w in widgets]
        idx = self.shape_combo.findData(self.state.shape_type)
        if idx >= 0: self.shape_combo.setCurrentIndex(idx)
        self.r_h.setValue(self.state.r_h); self.l_h.setValue(self.state.l_h); self.w_h.setValue(self.state.w_h)
        self.rot_x.setValue(self.state.rot_x_deg); self.rot_y.setValue(self.state.rot_y_deg); self.rot_z.setValue(self.state.rot_z_deg)
        self.u_norm.setValue(self.state.u_norm); self.v_norm.setValue(self.state.v_norm); self.scale_norm.setValue(self.state.scale_norm)
        gi = self.grade_combo.findText(self.state.fit_grade)
        if gi >= 0: self.grade_combo.setCurrentIndex(gi)
        self._update_visibility(); self._update_metrics(); self._update_root_label(); del blockers

    def _update_visibility(self) -> None:
        cone = self.state.shape_type == "cone"; cuboid = self.state.shape_type == "cuboid"
        self.r_h.setEnabled(cone); self.l_h.setEnabled(cuboid); self.w_h.setEnabled(cuboid); self.rot_y.setEnabled(cuboid)

    def _on_canvas_state_edited(self) -> None:
        self._state_to_controls()

    def _update_metrics(self) -> None:
        iou = self.canvas.iou(); self.iou_label.setText("—" if iou is None else f"{iou:.4f}")

    def _update_root_label(self) -> None:
        if self.state.root_p1 and self.state.root_p2:
            p1, p2 = self.state.root_p1, self.state.root_p2; self.root_label.setText(f"P1=({p1[0]:.4f}, {p1[1]:.4f})\nP2=({p2[0]:.4f}, {p2[1]:.4f})")
        else: self.root_label.setText("未完成。按 Q 后依次点击根部两个端点。")

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开火灾图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path: self.load_image_path(Path(path))

    def load_image_path(self, path: Path) -> None:
        if not self.canvas.load_image(path): QMessageBox.warning(self, "读取失败", f"无法读取图片：{path}"); return
        self._refresh_image_list(path); self.state = AnnotationState(); self.canvas.set_state(self.state)
        annotation_path = self._annotation_path(path)
        if annotation_path.exists():
            try:
                self.state = AnnotationState.from_annotation_dict(json.loads(annotation_path.read_text(encoding="utf-8"))); self.canvas.set_state(self.state)
            except Exception as exc: QMessageBox.warning(self, "标注读取失败", str(exc))
        auto_mask = self._find_auto_mask(path)
        if auto_mask:
            self.canvas.load_mask(auto_mask)
            if not annotation_path.exists(): self.canvas.auto_initialize_from_mask()
        self._state_to_controls(); self.setWindowTitle(f"Fire 3DMark v1.0 — {path.name}"); self.statusBar().showMessage(str(path))

    def open_mask(self) -> None:
        if self.canvas.image is None: QMessageBox.information(self, "提示", "请先打开火灾图片。"); return
        path, _ = QFileDialog.getOpenFileName(self, "加载火焰Mask", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path and self.canvas.load_mask(path): self.canvas.auto_initialize_from_mask(); self._state_to_controls()

    def save_annotation(self) -> None:
        if self.canvas.image is None or self.canvas.image_path is None: return
        data = self.state.to_annotation_dict(image_name=self.canvas.image_path.name, mask_name=self.canvas.mask_path.name if self.canvas.mask_path else None, image_width=self.canvas.image.width(), image_height=self.canvas.image.height(), iou=self.canvas.iou())
        out = self._annotation_path(self.canvas.image_path); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"); self.statusBar().showMessage(f"已保存：{out}", 5000)

    def navigate_image(self, step: int) -> None:
        if not self._image_files: return
        idx = self._image_index + step
        if 0 <= idx < len(self._image_files): self.load_image_path(self._image_files[idx])

    def _refresh_image_list(self, path: Path) -> None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}; self._image_files = sorted(p for p in path.parent.iterdir() if p.suffix.lower() in exts)
        self._image_index = self._image_files.index(path) if path in self._image_files else -1

    @staticmethod
    def _annotation_path(image_path: Path) -> Path:
        if image_path.parent.name.lower() == "images": return image_path.parent.parent / "annotations" / f"{image_path.stem}.json"
        return image_path.parent / "annotations" / f"{image_path.stem}.json"

    @staticmethod
    def _find_auto_mask(image_path: Path) -> Path | None:
        candidates = [image_path.parent / "masks" / f"{image_path.stem}.png", image_path.parent / f"{image_path.stem}_mask.png", image_path.with_suffix(".mask.png")]
        if image_path.parent.name.lower() == "images": candidates.insert(0, image_path.parent.parent / "masks" / f"{image_path.stem}.png")
        return next((p for p in candidates if p.exists()), None)

    def reset_state(self) -> None:
        shape = self.state.shape_type; self.state = AnnotationState(shape_type=shape); self.canvas.set_state(self.state)
        if self.canvas.mask is not None: self.canvas.auto_initialize_from_mask()
        self._state_to_controls()

    def auto_initialize(self) -> None:
        self.canvas.auto_initialize_from_mask(); self._state_to_controls()

    def clear_root(self) -> None:
        self.state.root_p1 = None; self.state.root_p2 = None; self.canvas.update(); self._update_root_label()

    def _toggle_mask(self, checked: bool) -> None:
        self.canvas.show_mask = checked; self.canvas.update()

    def _toggle_projection(self, checked: bool) -> None:
        self.canvas.show_projection = checked; self.canvas.update()


def run() -> int:
    app = QApplication.instance() or QApplication([]); app.setApplicationName("Fire 3DMark")
    window = MainWindow(); window.show(); return app.exec()
