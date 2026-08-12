from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget

from .geometry import cone_geometry, convex_hull, cuboid_geometry, project_weak_perspective
from .state import AnnotationState


class AnnotationCanvas(QWidget):
    stateEdited = Signal()
    rootEdited = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 540)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.state = AnnotationState()
        self.image: QImage | None = None
        self.image_path: Path | None = None
        self.mask: np.ndarray | None = None
        self.mask_path: Path | None = None
        self.mask_overlay: QImage | None = None
        self.show_mask = True
        self.show_projection = True
        self.mode = "translate"
        self._drag_start: QPointF | None = None
        self._start_u = self._start_v = 0.0
        self._start_rx = self._start_rz = 0.0
        self._pending_root: tuple[float, float] | None = None

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode != "root":
            self._pending_root = None
        self.update()

    def load_image(self, path: str | Path) -> bool:
        img = QImage(str(path))
        if img.isNull():
            return False
        self.image = img.convertToFormat(QImage.Format_RGB888)
        self.image_path = Path(path)
        self.mask = None
        self.mask_path = None
        self.mask_overlay = None
        self.update()
        return True

    def load_mask(self, path: str | Path) -> bool:
        if self.image is None:
            return False
        img = QImage(str(path))
        if img.isNull():
            return False
        img = img.convertToFormat(QImage.Format_Grayscale8)
        if img.width() != self.image.width() or img.height() != self.image.height():
            img = img.scaled(self.image.width(), self.image.height(), Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.mask = self._gray_qimage_to_numpy(img) > 127
        self.mask_path = Path(path)
        self.mask_overlay = self._make_mask_overlay(self.mask)
        self.update()
        return True

    def set_state(self, state: AnnotationState) -> None:
        self.state = state
        self.update()

    def image_rect(self) -> QRectF:
        if self.image is None:
            return QRectF()
        margin = 12.0
        scale = min((self.width() - 2 * margin) / self.image.width(), (self.height() - 2 * margin) / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def widget_to_image_norm(self, pos: QPointF) -> tuple[float, float] | None:
        rect = self.image_rect()
        if rect.isNull() or not rect.contains(pos):
            return None
        return ((pos.x() - rect.left()) / rect.width(), (pos.y() - rect.top()) / rect.height())

    def image_px_to_widget(self, xy: np.ndarray) -> np.ndarray:
        rect = self.image_rect()
        out = np.empty_like(np.asarray(xy, dtype=float))
        out[:, 0] = rect.left() + xy[:, 0] / self.image.width() * rect.width()
        out[:, 1] = rect.top() + xy[:, 1] / self.image.height() * rect.height()
        return out

    def projected_geometry(self):
        if self.image is None or self.state.shape_type == "invalid":
            return np.empty((0, 2)), [], np.empty((0, 2))
        if self.state.shape_type == "cone":
            vertices, edges = cone_geometry(self.state.r_h)
        else:
            vertices, edges = cuboid_geometry(self.state.l_h, self.state.w_h)
        uv, _ = project_weak_perspective(
            vertices, self.state.rotation_matrix(), self.state.u_norm, self.state.v_norm,
            self.state.scale_norm, self.image.width(), self.image.height()
        )
        return uv, edges, convex_hull(uv)

    def projected_mask(self) -> np.ndarray | None:
        if self.image is None or self.state.shape_type == "invalid":
            return None
        _, _, hull = self.projected_geometry()
        if len(hull) < 3:
            return None
        raster = QImage(self.image.width(), self.image.height(), QImage.Format_Grayscale8)
        raster.fill(0)
        p = QPainter(raster)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.white)
        p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in hull]))
        p.end()
        return self._gray_qimage_to_numpy(raster) > 127

    def iou(self) -> float | None:
        if self.mask is None:
            return None
        pred = self.projected_mask()
        if pred is None:
            return None
        inter = np.logical_and(pred, self.mask).sum()
        union = np.logical_or(pred, self.mask).sum()
        return 1.0 if union == 0 else float(inter / union)

    def auto_initialize_from_mask(self) -> None:
        if self.mask is None or self.image is None:
            return
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return
        xmin, xmax, ymin, ymax = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        h, w = max(1, ymax - ymin + 1), max(1, xmax - xmin + 1)
        self.state.u_norm = ((xmin + xmax) / 2.0) / self.image.width()
        self.state.v_norm = ymax / self.image.height()
        self.state.scale_norm = h / self.image.height()
        self.state.r_h = min(max(w / (2.0 * h), 0.05), 1.2)
        self.state.l_h = min(max(w / h, 0.05), 2.0)
        self.state.w_h = min(max(self.state.l_h * 0.5, 0.05), 1.5)
        self.state.rot_x_deg = self.state.rot_y_deg = self.state.rot_z_deg = 0.0
        self.state.root_p1 = (xmin / self.image.width(), ymax / self.image.height())
        self.state.root_p2 = (xmax / self.image.width(), ymax / self.image.height())
        self.stateEdited.emit(); self.rootEdited.emit(); self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(34, 37, 43))
        if self.image is None:
            p.setPen(QColor(210, 210, 210))
            p.drawText(self.rect(), Qt.AlignCenter, "打开一张火灾图片开始标注")
            return
        rect = self.image_rect()
        p.drawImage(rect, self.image)
        if self.show_mask and self.mask_overlay is not None:
            p.drawImage(rect, self.mask_overlay)
        if self.show_projection and self.state.shape_type != "invalid":
            uv, edges, hull = self.projected_geometry()
            if len(hull) >= 3:
                hw = self.image_px_to_widget(hull)
                poly = QPolygonF([QPointF(float(x), float(y)) for x, y in hw])
                p.setPen(QPen(QColor(0, 220, 255, 220), 2.0)); p.setBrush(QColor(0, 220, 255, 45)); p.drawPolygon(poly)
            uvw = self.image_px_to_widget(uv)
            p.setPen(QPen(QColor(235, 250, 255, 180), 1.0)); p.setBrush(Qt.NoBrush)
            for a, b in edges:
                p.drawLine(QPointF(*uvw[a]), QPointF(*uvw[b]))
        self._paint_root(p)
        p.setPen(QColor(255, 255, 255)); p.setBrush(QColor(0, 0, 0, 130))
        box = QRectF(18, 18, 140, 30); p.drawRoundedRect(box, 5, 5)
        p.drawText(box, Qt.AlignCenter, {"translate":"平移模式", "rotate":"旋转模式", "root":"根部标注模式"}[self.mode])

    def _paint_root(self, p: QPainter) -> None:
        if self.image is None:
            return
        rect = self.image_rect(); p.setPen(QPen(QColor(255, 210, 0), 3.0)); p.setBrush(QColor(255, 210, 0))
        p1, p2 = self.state.root_p1, self.state.root_p2
        q1 = q2 = None
        if p1:
            q1 = QPointF(rect.left()+p1[0]*rect.width(), rect.top()+p1[1]*rect.height()); p.drawEllipse(q1, 4, 4)
        if p2:
            q2 = QPointF(rect.left()+p2[0]*rect.width(), rect.top()+p2[1]*rect.height()); p.drawEllipse(q2, 4, 4)
        if q1 and q2:
            p.drawLine(q1, q2)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self.image is None:
            return
        if self.mode == "root":
            norm = self.widget_to_image_norm(event.position())
            if norm is None:
                return
            if self._pending_root is None:
                self._pending_root = norm; self.state.root_p1 = norm; self.state.root_p2 = None
            else:
                self.state.root_p1 = self._pending_root; self.state.root_p2 = norm; self._pending_root = None
                self.state.u_norm = (self.state.root_p1[0] + self.state.root_p2[0]) / 2.0
                self.state.v_norm = (self.state.root_p1[1] + self.state.root_p2[1]) / 2.0
                self.rootEdited.emit(); self.stateEdited.emit()
            self.update(); return
        self._drag_start = event.position()
        self._start_u, self._start_v = self.state.u_norm, self.state.v_norm
        self._start_rx, self._start_rz = self.state.rot_x_deg, self.state.rot_z_deg

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is None or not (event.buttons() & Qt.LeftButton):
            return
        rect = self.image_rect(); delta = event.position() - self._drag_start
        if self.mode == "translate":
            self.state.u_norm = self._start_u + delta.x() / rect.width(); self.state.v_norm = self._start_v + delta.y() / rect.height()
        elif self.mode == "rotate":
            self.state.rot_z_deg = self._start_rz + delta.x() * 0.35
            self.state.rot_x_deg = self._start_rx - delta.y() * 0.35
            if self.state.shape_type == "cone":
                self.state.rot_y_deg = 0.0
        self.state.clamp(); self.stateEdited.emit(); self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_start = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self.image is None or self.widget_to_image_norm(event.position()) is None:
            return
        self.state.scale_norm *= math.exp(event.angleDelta().y() / 1200.0)
        self.state.clamp(); self.stateEdited.emit(); self.update(); event.accept()

    @staticmethod
    def _gray_qimage_to_numpy(img: QImage) -> np.ndarray:
        q = img.convertToFormat(QImage.Format_Grayscale8)
        arr = np.frombuffer(q.bits(), dtype=np.uint8, count=q.sizeInBytes())
        return arr.reshape((q.height(), q.bytesPerLine()))[:, :q.width()].copy()

    @staticmethod
    def _make_mask_overlay(mask: np.ndarray) -> QImage:
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 80, 40
        rgba[..., 3] = np.where(mask, 90, 0).astype(np.uint8)
        return QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888).copy()
