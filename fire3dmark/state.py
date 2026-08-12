from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import rotation_6d, rotation_matrix_xyz


@dataclass
class AnnotationState:
    shape_type: str = "cone"
    r_h: float = 0.30
    l_h: float = 0.60
    w_h: float = 0.35
    rot_x_deg: float = 0.0
    rot_y_deg: float = 0.0
    rot_z_deg: float = 0.0
    u_norm: float = 0.50
    v_norm: float = 0.80
    scale_norm: float = 0.50
    root_p1: tuple[float, float] | None = None
    root_p2: tuple[float, float] | None = None
    fit_grade: str = "unrated"

    def clamp(self) -> None:
        self.r_h = min(max(self.r_h, 0.02), 2.0)
        self.l_h = min(max(self.l_h, 0.02), 3.0)
        self.w_h = min(max(self.w_h, 0.02), 3.0)
        self.u_norm = min(max(self.u_norm, -1.0), 2.0)
        self.v_norm = min(max(self.v_norm, -1.0), 2.0)
        self.scale_norm = min(max(self.scale_norm, 0.01), 5.0)

    def rotation_matrix(self):
        return rotation_matrix_xyz(self.rot_x_deg, self.rot_y_deg, self.rot_z_deg)

    def to_annotation_dict(
        self,
        *,
        image_name: str,
        mask_name: str | None,
        image_width: int,
        image_height: int,
        iou: float | None,
    ) -> dict[str, Any]:
        rotation = self.rotation_matrix()
        result: dict[str, Any] = {
            "format": "fire_3dmark",
            "version": "1.0",
            "image": {"file": image_name, "width": image_width, "height": image_height},
            "mask": {"file": mask_name} if mask_name else None,
            "projection": {
                "type": "weak_perspective",
                "camera_intrinsics_required": False,
                "note": "Scale-free equivalent 3D pose/shape annotation.",
            },
            "shape": {"class": self.shape_type, "valid": self.shape_type != "invalid"},
            "pose": {
                "representation": "canonical_equivalent_pose",
                "euler_xyz_deg": [self.rot_x_deg, self.rot_y_deg, self.rot_z_deg],
                "rotation_6d": rotation_6d(rotation),
            },
            "image_alignment": {
                "u_norm": self.u_norm,
                "v_norm": self.v_norm,
                "scale_norm": self.scale_norm,
            },
            "fit_quality": {"iou": iou, "grade": self.fit_grade},
        }

        if self.shape_type == "cone":
            result["shape_parameters"] = {"height_norm": 1.0, "r_h": self.r_h}
        elif self.shape_type == "cuboid":
            result["shape_parameters"] = {
                "height_norm": 1.0,
                "l_h": self.l_h,
                "w_h": self.w_h,
            }
        else:
            result["shape_parameters"] = None

        if self.root_p1 is not None and self.root_p2 is not None:
            p1, p2 = self.root_p1, self.root_p2
            result["root"] = {
                "p1_norm": list(p1),
                "p2_norm": list(p2),
                "center_norm": [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0],
                "width_norm_image": ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5,
            }
        else:
            result["root"] = None
        return result

    @classmethod
    def from_annotation_dict(cls, data: dict[str, Any]) -> "AnnotationState":
        state = cls()
        state.shape_type = data.get("shape", {}).get("class", "cone")
        params = data.get("shape_parameters") or {}
        state.r_h = float(params.get("r_h", state.r_h))
        state.l_h = float(params.get("l_h", state.l_h))
        state.w_h = float(params.get("w_h", state.w_h))

        euler = data.get("pose", {}).get("euler_xyz_deg", [0.0, 0.0, 0.0])
        if len(euler) == 3:
            state.rot_x_deg, state.rot_y_deg, state.rot_z_deg = map(float, euler)

        align = data.get("image_alignment", {})
        state.u_norm = float(align.get("u_norm", state.u_norm))
        state.v_norm = float(align.get("v_norm", state.v_norm))
        state.scale_norm = float(align.get("scale_norm", state.scale_norm))
        state.fit_grade = data.get("fit_quality", {}).get("grade", "unrated")

        root = data.get("root")
        if root:
            p1 = root.get("p1_norm")
            p2 = root.get("p2_norm")
            if p1 and p2:
                state.root_p1 = (float(p1[0]), float(p1[1]))
                state.root_p2 = (float(p2[0]), float(p2[1]))
        state.clamp()
        return state
