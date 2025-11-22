from dataclasses import dataclass
from typing import List

# ==================== 配置信息 ======================
HAS_360_DEGREE_LIDAR = False  # 量产车为 False
NO_FRONT_WIDE_SIDE_CAMERAS = False
# ==================================================


@dataclass(frozen=True)
class CamSpec:
    key: str  # 原始传感器名称
    name: str
    width: int
    height: int
    comment: str = ""


# ---------- 完整 13-cam 规格 ----------
_ALL_CAM_SPECS: List[CamSpec] = [
    CamSpec(
        key="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",
        name="front_wide_110",
        width=1024,
        height=512,
        comment="广角前视 FOV110",
    ),
    CamSpec(
        key="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
        name="front_wide_60",
        width=1024,
        height=512,
        comment="广角前视 FOV60",
    ),
    CamSpec(
        key="CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
        name="front_tele_30",
        width=1024,
        height=512,
        comment="长焦前视 FOV30",
    ),
    CamSpec(
        key="CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
        name="front_tele_15",
        width=1024,
        height=512,
        comment="长焦前视 FOV15",
    ),
    CamSpec(
        key="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60",
        name="front_wide_left_60",
        width=1024,
        height=512,
        comment="广角左前 FOV60",
    ),
    CamSpec(
        key="CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
        name="front_left_99",
        width=1024,
        height=512,
        comment="左前 FOV99",
    ),
    CamSpec(
        key="CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
        name="rear_left_99",
        width=1024,
        height=512,
        comment="左后 FOV99",
    ),
    CamSpec(
        key="CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
        name="rear_left_30",
        width=512,
        height=256,
        comment="左后 FOV30",
    ),
    CamSpec(
        key="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60",
        name="front_wide_right_60",
        width=1024,
        height=512,
        comment="广角右前 FOV60",
    ),
    CamSpec(
        key="CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
        name="front_right_99",
        width=1024,
        height=512,
        comment="右前 FOV99",
    ),
    CamSpec(
        key="CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
        name="rear_right_99",
        width=1024,
        height=512,
        comment="右后 FOV99",
    ),
    CamSpec(
        key="CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
        name="rear_right_30",
        width=512,
        height=256,
        comment="右后 FOV30",
    ),
    CamSpec(
        key="CAM_PBQ_REAR_RESET_OPTICAL_H50",
        name="rear_50",
        width=1024,
        height=512,
        comment="后视 FOV50",
    ),
]

SIDE_WIDE_KEYS = {
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60",
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60",
}

if NO_FRONT_WIDE_SIDE_CAMERAS:
    FINAL_CAM_SPECS = [c for c in _ALL_CAM_SPECS if c.key not in SIDE_WIDE_KEYS]
else:
    FINAL_CAM_SPECS = _ALL_CAM_SPECS

FINAL_CAM_SPECS = {cam_id: c for cam_id, c in enumerate(FINAL_CAM_SPECS)}
ORIGINAL_CAM_NAME_TO_CAM_ID = {c.key : cam_id for cam_id, c in FINAL_CAM_SPECS.items()}

if HAS_360_DEGREE_LIDAR:
    MAIN_LIDAR_NAME = "LDR_CENTER"
else:
    MAIN_LIDAR_NAME = "LDR_FRONT"