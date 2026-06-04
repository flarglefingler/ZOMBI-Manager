from __future__ import annotations

import os
import sys
import tempfile
import traceback
import json
from typing import Dict, Optional

from PySide6.QtCore import Qt, QSize, QUrl, QPointF, QObject, QRunnable, QThreadPool, Signal, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QBrush,
    QFont,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utilities import (
    bfz,
    geo_format,
    material_format,
    mdf_format,
    obj_format,
    oli,
    previewers,
    skn_format,
    tdt,
    trl_format,
    wor_format,
)


MAX_PREVIEW_FACES = 3500
MAX_DRAG_PREVIEW_FACES = 1400
MAX_PREVIEW_CACHE_ITEMS = 8
SIDECAR_PREVIEW_EXTS = (".vii", ".mtn", ".mat", ".mta", ".tex")
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zombi_manager_settings.json")


def bytes_preview(data: bytes, n: int = 384) -> str:
    sample = data[:n]
    hex_text = " ".join(f"{byte:02x}" for byte in sample)
    ascii_text = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in sample)
    return f"Hex (first {min(len(data), n)} bytes):\n{hex_text}\n\nASCII:\n{ascii_text}"


def file_kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".geo"):
        return "3D model"
    if lower.endswith(".obj"):
        return "Game Object"
    if lower.endswith(".vii"):
        return "Visual Instance" # this is a guess
    if lower.endswith(".mtn"):
        return "Motion Setup" # also a guess
    if lower.endswith(".mat"):
        return "Material"
    if lower.endswith(".mta"):
        return "Texture Material"
    if lower.endswith(".mdf"):
        return "Modifier Data"
    if lower.endswith(".wor"):
        return "World"
    if lower.endswith(".tdt"):
        return "Texture"
    if lower.endswith(".skn"):
        return "Skeletal Data"
    if lower.endswith(".trl"):
        return "Animation Track"
    if lower.endswith(".oli"):
        return "Localization"
    if lower.endswith(".tex"):
        return "Texture Descriptor"
    if lower.endswith(".son"):
        return "Audio"
    ext = os.path.splitext(name)[1].lower()
    return ext[1:].upper() if ext else "File"


def converted_stem(name: str) -> str:
    base = os.path.basename(name)
    if base.lower().endswith(".pc.tdt"):
        return base[:-7]
    return os.path.splitext(base)[0]


def entry_key_text(entry: bfz.BFZFileEntry) -> str:
    return bfz.BFZArchive.key_hex(entry.key) if entry.key else ""


def load_app_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_app_settings(settings: dict) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
        handle.write("\n")


def game_data_dir(game_dir: str) -> Optional[str]:
    if not game_dir:
        return None
    path = os.path.abspath(os.path.expanduser(game_dir))
    if os.path.basename(path).lower() == "data" and os.path.isdir(path):
        return path
    data_dir = os.path.join(path, "Data")
    if os.path.isdir(data_dir):
        return data_dir
    return None


def normalized_game_dir(game_dir: str) -> Optional[str]:
    data_dir = game_data_dir(game_dir)
    if not data_dir:
        return None
    if os.path.basename(os.path.abspath(game_dir)).lower() == "data":
        return os.path.dirname(data_dir)
    return os.path.abspath(os.path.expanduser(game_dir))


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size:,} B"


class WorkerSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class TaskWorker(QRunnable):
    def __init__(self, token: int, fn, *args):
        super().__init__()
        self.token = token
        self.fn = fn
        self.args = args
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.token, self.fn(*self.args))
        except Exception:
            self.signals.failed.emit(self.token, traceback.format_exc())


def parse_archive_task(path: str) -> bfz.BFZArchive:
    archive = bfz.BFZArchive(path)
    archive.parse()
    return archive


def scan_game_archives_task(game_dir: str):
    data_dir = game_data_dir(game_dir)
    if not data_dir:
        raise ValueError("Could not find a Data folder in that game directory.")
    return bfz.scan_archives_in_directory(data_dir)


def build_geo_preview_task(data: bytes):
    model = geo_format.parse_geo_model(data)
    face_count = sum(len(part.faces) for part in model.parts)
    part_names = "\n".join(f"- {part.name}: {len(part.faces):,} faces" for part in model.parts)
    meta = (
        f"{len(model.points):,} vertices\n"
        f"{face_count:,} faces\n"
        f"{len(model.parts)} part(s)\n\n"
        f"{part_names}"
    )
    return model, meta


def build_tdt_preview_task(name: str, data: bytes):
    info = tdt.parse_tdt_texture_data(data, name)
    width, height, rgba = tdt.decode_tdt_top_mip_rgba_data(data, info)
    meta = f"{width} x {height}\n{info.format_name}\n{len(info.levels)} mip(s)\n{len(data):,} bytes"
    return width, height, rgba, meta


def build_skn_preview_task(name: str, data: bytes):
    skn = skn_format.parse_skn(data, name)
    children: Dict[int, list] = {}
    known_indices = {bone.index for bone in skn.bones}
    for bone in skn.bones:
        parent = bone.parent_index if bone.parent_index in known_indices else -1
        children.setdefault(parent, []).append(bone)

    lines = ["Skeleton hierarchy", ""]

    def add_bone(parent_index: int, depth: int):
        for bone in sorted(children.get(parent_index, []), key=lambda item: item.index):
            lines.append(f"{'  ' * depth}- [{bone.index:03d}] {bone.name}")
            add_bone(bone.index, depth + 1)

    add_bone(-1, 0)
    if len(lines) == 2:
        lines.append("(no hierarchy decoded)")

    pose_blocks = "\n".join(
        f"- {block.tag}: {len(block.transforms):,} transform(s) at 0x{block.offset:x}"
        for block in skn.pose_blocks
    ) or "- none decoded"
    masks = "\n".join(
        f"- {mask.name}: {len(mask.entries):,} bone(s)"
        for mask in skn.masks
    ) or "- none"
    meta = (
        f"Skeletal Data\n"
        f"{skn.bone_count:,} bones\n"
        f"{len(children.get(-1, [])):,} root bone(s)\n"
        f"version {skn.version}\n"
        f"{len(data):,} bytes\n\n"
        f"Pose blocks:\n{pose_blocks}\n\n"
        f"Masks:\n{masks}"
    )
    return "Skeletal Data", "\n".join(lines), meta


def build_trl_preview_task(name: str, data: bytes):
    trl = trl_format.parse_trl(data, name)
    sampled = trl_format.decode_trl_dense_animation(data, trl)
    fps = trl.fps
    if fps <= 0 and trl.duration > 0:
        fps = trl.frame_count / trl.duration
    length = trl.duration if trl.duration > 0 else (trl.frame_count / fps if fps > 0 else 0.0)

    groups = "\n".join(
        f"- {group.kind.replace('_', ' ')}: {group.length:,} track(s)"
        for group in trl.channel_groups
    ) or "- none"
    sections = "\n".join(
        f"- {section.name}: 0x{section.offset:x} + 0x{section.length:x}"
        for section in trl.sections
    ) or "- none"
    notes = "\n".join(f"- {note}" for note in sampled.notes) or "- no sampled keys decoded"
    keyframes = len({sample.frame for sample in sampled.samples})

    body = (
        "Animation track summary\n\n"
        f"Frames: {trl.frame_start} - {trl.frame_end} ({trl.frame_count:,} total)\n"
        f"Decoded keyframes: {keyframes:,}\n"
        f"Sample windows: {len(sampled.frame_windows):,}\n"
        f"Dense in-window keys: {'yes' if sampled.dense_frame_keys else 'no'}\n\n"
        f"Channel groups:\n{groups}\n\n"
        f"Decode notes:\n{notes}\n\n"
        f"Sections:\n{sections}"
    )
    meta = (
        f"Animation Track\n"
        f"approx {fps:.2f} fps\n"
        f"{length:.2f} seconds\n"
        f"{trl.frame_count:,} frame(s)\n"
        f"{keyframes:,} decoded keyframe(s)\n"
        f"{trl.bone_count:,} animated bone slot(s)\n"
        f"version {trl.version}\n"
        f"{len(data):,} bytes"
    )
    return "Animation Track", body, meta


def build_wor_preview_task(name: str, data: bytes, key_map: Dict[int, list]):
    world = wor_format.parse_wor(data, name)
    resolved_count = sum(1 for ref in world.object_refs if ref.object_key in key_map)
    metadata_count = sum(1 for ref in world.object_refs if ref.has_metadata)

    lines = ["World object refs", ""]
    lines.extend(wor_format.refs_to_lines(world.object_refs, key_map, limit=350))
    if not world.object_refs:
        lines.append("(no object refs decoded)")

    meta = (
        f"World\n"
        f"version {world.version}\n"
        f"{world.object_count:,} object ref(s)\n"
        f"{resolved_count:,} resolved by archive key\n"
        f"{metadata_count:,} ref(s) with metadata\n"
        f"world chunk 0x{world.world_chunk_size:x} bytes\n"
        f"object group 0x{world.object_group_size:x} bytes\n"
        f"{world.stored_size:,} bytes"
    )
    return "World", "\n".join(lines), meta


def _archive_key_matches(key_map: Dict[int, list], key: int) -> list[tuple[str, int, list]]:
    matches = []
    seen = set()
    for label, value in (
        ("exact", key & 0xFFFFFFFF),
        ("stored + 1", (key + 1) & 0xFFFFFFFF),
        ("stored - 1", (key - 1) & 0xFFFFFFFF),
    ):
        paths = key_map.get(value)
        if not paths or value in seen:
            continue
        seen.add(value)
        matches.append((label, value, paths))
    return matches


def _format_key_line(key_map: Dict[int, list], key: int, limit_paths: int = 3) -> str:
    matches = _archive_key_matches(key_map, key)
    if not matches:
        return f"{key & 0xffffffff:08X}"

    bits = []
    for label, value, paths in matches:
        shown = ", ".join(os.path.basename(path) for path in paths[:limit_paths])
        if len(paths) > limit_paths:
            shown += f", +{len(paths) - limit_paths} more"
        prefix = f"{value & 0xffffffff:08X}" if label != "exact" else ""
        bits.append(f"{label}: {prefix + ' -> ' if prefix else ''}{shown}")
    return f"{key & 0xffffffff:08X} ({'; '.join(bits)})"


def _key_lines(title: str, keys, key_map: Dict[int, list], limit: int = 120) -> list[str]:
    keys = list(keys)
    lines = [title, ""]
    if not keys:
        lines.append("- none decoded")
        return lines
    for key in keys[:limit]:
        lines.append(f"- {_format_key_line(key_map, key)}")
    if len(keys) > limit:
        lines.append(f"... {len(keys) - limit:,} more")
    return lines


def _texture_ref_line(key_map: Dict[int, list], texture_ref_map: Dict[int, list], key: int) -> str:
    bits = []
    texture_paths = texture_ref_map.get(key & 0xFFFFFFFF)
    if texture_paths:
        shown = ", ".join(os.path.basename(path) for path in texture_paths[:3])
        if len(texture_paths) > 3:
            shown += f", +{len(texture_paths) - 3} more"
        bits.append(f"texture: {shown}")

    archive_matches = _archive_key_matches(key_map, key)
    if archive_matches:
        match_text = _format_key_line(key_map, key)
        if "(" in match_text:
            bits.append(match_text.split("(", 1)[1].rstrip(")"))

    if not bits:
        return f"{key & 0xffffffff:08X}"
    return f"{key & 0xffffffff:08X} ({'; '.join(bits)})"


def _texture_ref_lines(
    title: str,
    keys,
    key_map: Dict[int, list],
    texture_ref_map: Dict[int, list],
    limit: int = 80,
) -> list[str]:
    keys = list(keys)
    lines = [title, ""]
    if not keys:
        lines.append("- none decoded")
        return lines
    for key in keys[:limit]:
        lines.append(f"- {_texture_ref_line(key_map, texture_ref_map, key)}")
    if len(keys) > limit:
        lines.append(f"... {len(keys) - limit:,} more")
    return lines


def build_obj_preview_task(name: str, data: bytes, key_map: Dict[int, list]):
    obj = obj_format.parse_obj(data, name)
    matrix_lines = []
    for matrix in obj.matrices:
        matrix_lines.append(f"- matrix at 0x{matrix.offset:x}")
    translation = obj.translation
    if obj.translation_offset >= 0:
        matrix_lines.append(
            f"- active translation at 0x{obj.translation_offset:x}: "
            f"({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})"
        )
    body_lines = ["Game object summary", "", "Matrices", "", *matrix_lines, ""]
    body_lines.extend(_key_lines("Resource keys", obj.resource_keys, key_map))
    meta = (
        "Game Object\n"
        f"version {obj.version}\n"
        f"{len(obj.matrices):,} matrix/matrices\n"
        f"{len(obj.resource_keys):,} possible resource key(s)\n"
        f"{len(data):,} bytes"
    )
    return "Game Object", "\n".join(body_lines), meta


def build_sidecar_preview_task(name: str, data: bytes, key_map: Dict[int, list], texture_ref_map: Optional[Dict[int, list]] = None):
    texture_ref_map = texture_ref_map or {}
    lower = name.lower()
    if lower.endswith((".vii", ".mtn")):
        sidecar = obj_format.parse_sidecar(data, name)
        label = "Visual Instance" if sidecar.kind == "VIS" else "Motion Setup"
        body_lines = [f"{label} summary", ""]
        body_lines.extend(_key_lines("Resource keys", sidecar.resource_keys, key_map))
        meta = (
            f"{label}\n"
            f"{sidecar.kind}\n"
            f"{len(sidecar.resource_keys):,} possible resource key(s)\n"
            f"{len(data):,} bytes"
        )
        return label, "\n".join(body_lines), meta

    if lower.endswith(".mat"):
        mat = material_format.parse_mat(data, name)
        body_lines = ["Material summary", ""]
        body_lines.extend(_key_lines("Sub-material keys", mat.submaterial_keys, key_map))
        meta = (
            "Material\n"
            f"{len(mat.submaterial_keys):,} sub-material key(s)\n"
            f"{len(data):,} bytes"
        )
        return "Material", "\n".join(body_lines), meta

    if lower.endswith(".mta"):
        mta = material_format.parse_mta(data, name)
        matched_refs = [
            ref for ref in mta.texture_refs
            if texture_ref_map.get(ref & 0xFFFFFFFF) or _archive_key_matches(key_map, ref)
        ]
        body_lines = [
            "Texture material summary",
            "",
            f"Shader: {mta.shader_name or 'unknown'}",
            f"Layer count: {mta.layer_count}",
            "",
        ]
        if mta.texture_slots:
            body_lines.append("Texture slot records")
            body_lines.append("")
            for slot in mta.texture_slots:
                transform = ", ".join(f"{value:.4g}" for value in slot.transform)
                body_lines.append(
                    f"- slot {slot.index} @ 0x{slot.offset:x}: "
                    f"{_texture_ref_line(key_map, texture_ref_map, slot.texture_ref)} "
                    f"transform[{transform}] flags=0x{slot.flags:08X}"
                )
        else:
            body_lines.extend(_texture_ref_lines("Texture slot records", (), key_map, texture_ref_map))
        body_lines.append("")
        extra_matches = [
            ref for ref in mta.extra_texture_refs
            if texture_ref_map.get(ref & 0xFFFFFFFF) or _archive_key_matches(key_map, ref)
        ]
        body_lines.extend(_texture_ref_lines("Extra decoded texture refs", extra_matches, key_map, texture_ref_map))
        meta = (
            "Texture Material\n"
            f"{mta.shader_name or 'unknown shader'}\n"
            f"{mta.layer_count:,} layer(s)\n"
            f"{len(mta.texture_slots):,} texture slot record(s)\n"
            f"{len(mta.extra_texture_refs):,} loose ref word(s)\n"
            f"{len(matched_refs):,} matched texture ref(s)\n"
            f"{len(data):,} bytes"
        )
        return "Texture Material", "\n".join(body_lines), meta

    if lower.endswith(".tex"):
        tex = material_format.parse_tex(data, name)
        body_lines = ["Texture descriptor summary", ""]
        if tex.width and tex.height:
            body_lines.append(f"- size: {tex.width} x {tex.height}")
        if tex.format_code is not None:
            body_lines.append(f"- format: {tex.format_name} (0x{tex.format_code:02X})")
        if tex.file_key is None:
            body_lines.append("- payload/resource key: not decoded")
            key_text = "none"
        else:
            body_lines.append("- payload/resource key: " + _format_key_line(key_map, tex.file_key))
            key_text = f"{tex.file_key & 0xffffffff:08X}"
        meta = (
            "Texture Descriptor\n"
            f"payload/resource key: {key_text}\n"
            f"{tex.format_name or 'unknown format'}\n"
            f"{tex.width or '?'} x {tex.height or '?'}\n"
            f"{len(data):,} bytes"
        )
        return "Texture Descriptor", "\n".join(body_lines), meta

    raise ValueError("unsupported sidecar")


def build_mdf_preview_task(name: str, data: bytes, key_map: Dict[int, list]):
    mdf = mdf_format.parse_mdf(data, name)
    lines = ["Modifier data summary", ""]
    if not mdf.modifier_links:
        lines.append("- no modifier links decoded")
    for link in mdf.modifier_links:
        order = f", order {link.order}" if link.order else ""
        lines.append(
            f"- [{link.index:03d}] {_format_key_line(key_map, link.resource_key)} "
            f"-> {link.type_name} ({link.modifier_type}) flags=0x{link.flags:08X}{order}"
        )
    meta = (
        "Modifier Data\n"
        f"version {mdf.version}\n"
        f"{mdf.link_count:,} modifier link(s)\n"
        f"link bytes 0x{mdf.links_size:x}\n"
        f"{len(data):,} bytes"
    )
    return "Modifier Data", "\n".join(lines), meta


def build_oli_preview_task(name: str, data: bytes):
    oli_file = oli.parse_oli_data(data, name, force=True)
    body_lines = ["Localization strings", ""]
    for index, text in enumerate(oli_file.texts):
        body_lines.append(f"{index:04d}: {text}")
    if not oli_file.texts:
        body_lines.append("(no strings decoded)")

    warnings = "\n".join(f"- {warning}" for warning in oli_file.warnings)
    meta = (
        f"Localization\n"
        f"{oli_file.string_count:,} string(s)\n"
        f"{len(data):,} bytes"
    )
    if oli_file.filename:
        meta += f"\nsource: {oli_file.filename}"
    if oli_file.lyn_output:
        meta += f"\nlyn: {oli_file.lyn_output}"
    if oli_file.agent:
        meta += f"\nagent: {oli_file.agent}"
    if warnings:
        meta += f"\n\nWarnings:\n{warnings}"
    return "Localization", "\n".join(body_lines), meta


class MeshViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.model: Optional[geo_format.GeoModel] = None
        self.yaw = -0.45
        self.pitch = 0.35
        self.zoom = 1.0
        self.center = (0.0, 0.0, 0.0)
        self.radius = 1.0
        self.last_mouse_pos = None
        self.render_faces = []
        self.face_count = 0
        self.setMouseTracking(True)

    def set_model(self, model: geo_format.GeoModel):
        self.model = model
        self.prepare_faces()
        self.reset_view()
        self.update()

    def prepare_faces(self):
        self.render_faces = []
        self.face_count = 0
        if not self.model:
            return
        faces = []
        for part_index, part in enumerate(self.model.parts):
            self.face_count += len(part.faces)
            for face in part.faces:
                faces.append((part_index, face))
        if len(faces) > MAX_PREVIEW_FACES:
            step = max(1, len(faces) // MAX_PREVIEW_FACES)
            faces = faces[::step]
        self.render_faces = faces

    def reset_view(self):
        self.yaw = -0.45
        self.pitch = 0.35
        self.zoom = 1.0
        if not self.model or not self.model.points:
            self.center = (0.0, 0.0, 0.0)
            self.radius = 1.0
            return
        xs = [point[0] for point in self.model.points]
        ys = [point[1] for point in self.model.points]
        zs = [point[2] for point in self.model.points]
        self.center = (
            (min(xs) + max(xs)) * 0.5,
            (min(ys) + max(ys)) * 0.5,
            (min(zs) + max(zs)) * 0.5,
        )
        self.radius = max(
            max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) * 0.5,
            0.001,
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
        pos = event.position()
        delta = pos - self.last_mouse_pos
        self.last_mouse_pos = pos
        self.yaw += delta.x() * 0.01
        self.pitch = max(-1.45, min(1.45, self.pitch + delta.y() * 0.01))
        self.update()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None
        self.update()

    def wheelEvent(self, event):
        self.zoom *= 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.15, min(12.0, self.zoom))
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()
        self.update()

    def rotate_point(self, point):
        import math

        x = point[0] - self.center[0]
        y = point[1] - self.center[1]
        z = point[2] - self.center[2]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y = x * cy - y * sy, x * sy + y * cy
        y, z = y * cp - z * sp, y * sp + z * cp
        return x, y, z

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, self.last_mouse_pos is None)
        painter.fillRect(self.rect(), QColor(28, 30, 34))

        if not self.model:
            painter.setPen(QColor(145, 150, 158))
            painter.drawText(self.rect(), Qt.AlignCenter, "select a .geo file to preview the mesh")
            return

        scale = min(self.width(), self.height()) * 0.42 * self.zoom / self.radius
        projected = []
        depths = []
        for point in self.model.points:
            x, y, z = self.rotate_point(point)
            projected.append(QPointF(self.width() * 0.5 + x * scale, self.height() * 0.52 - z * scale))
            depths.append(y)

        draw_faces = []
        colors = [
            QColor(128, 164, 255, 75),
            QColor(245, 185, 90, 70),
            QColor(118, 220, 170, 70),
            QColor(215, 130, 235, 70),
        ]
        render_faces = self.render_faces
        if self.last_mouse_pos is not None and len(render_faces) > MAX_DRAG_PREVIEW_FACES:
            step = max(1, len(render_faces) // MAX_DRAG_PREVIEW_FACES)
            render_faces = render_faces[::step]

        for part_index, face in render_faces:
            draw_faces.append((
                sum(depths[index] for index in face) / 3.0,
                face,
                colors[part_index % len(colors)],
            ))

        draw_faces.sort(key=lambda item: item[0])

        painter.setPen(QPen(QColor(185, 190, 200, 95), 1))
        for _depth, face, color in draw_faces:
            polygon = QPolygonF([projected[index] for index in face])
            painter.setBrush(QBrush(color))
            painter.drawPolygon(polygon)

        painter.setPen(QColor(220, 224, 230))
        painter.setFont(QFont("Menlo", 10))
        shown = len(render_faces)
        suffix = f", drawing {shown:,}" if shown < self.face_count else ""
        painter.drawText(12, 22, f"{len(self.model.parts)} part(s), {self.face_count:,} faces{suffix}")


class PreviewPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_temp_audio = None
        self.current_entry = None
        self.current_data = b""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.title = QLabel("No file selected")
        self.title.setObjectName("PreviewTitle")
        self.title.setWordWrap(True)
        self.subtitle = QLabel("Open an archive and select a file.")
        self.subtitle.setObjectName("PreviewSubtitle")
        self.subtitle.setWordWrap(True)

        self.stack = QStackedWidget()
        self.empty_label = QLabel("nothing selected")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(360, 320)
        self.mesh_view = MeshViewport()
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.stack.addWidget(self.empty_label)
        self.stack.addWidget(self.image_label)
        self.stack.addWidget(self.mesh_view)
        self.stack.addWidget(self.text_view)

        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.play_btn.clicked.connect(lambda: self.player.play())
        self.pause_btn.clicked.connect(lambda: self.player.pause())

        audio_row = QHBoxLayout()
        audio_row.addWidget(self.play_btn)
        audio_row.addWidget(self.pause_btn)
        audio_row.addStretch(1)

        self.meta = QTextEdit()
        self.meta.setReadOnly(True)
        self.meta.setFixedHeight(150)

        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.stack, 1)
        layout.addLayout(audio_row)
        layout.addWidget(self.meta)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

    def clear(self):
        self.title.setText("No file selected")
        self.subtitle.setText("Open an archive and select a file.")
        self.meta.clear()
        self.text_view.clear()
        self.image_label.clear()
        self.empty_label.setText("nothing selected")
        self.mesh_view.model = None
        self.mesh_view.render_faces = []
        self.mesh_view.face_count = 0
        self.stack.setCurrentWidget(self.empty_label)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.player.stop()
        if self.current_temp_audio:
            try:
                os.unlink(self.current_temp_audio)
            except Exception:
                pass
            self.current_temp_audio = None

    def set_text(self, title: str, subtitle: str, body: str, meta: str = ""):
        self.clear()
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self.text_view.setPlainText(body)
        self.meta.setPlainText(meta)
        self.stack.setCurrentWidget(self.text_view)

    def set_loading(self, title: str, subtitle: str):
        self.clear()
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self.empty_label.setText("loading...")
        self.stack.setCurrentWidget(self.empty_label)

    def set_image(self, title: str, subtitle: str, image: QImage, meta: str):
        self.clear()
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ))
        self.meta.setPlainText(meta)
        self.stack.setCurrentWidget(self.image_label)

    def set_geo_model(self, title: str, model: geo_format.GeoModel, meta: str):
        self.clear()
        self.title.setText(title)
        self.subtitle.setText("GEO model preview")
        self.mesh_view.set_model(model)
        self.stack.setCurrentWidget(self.mesh_view)
        self.meta.setPlainText(meta)

    def set_tdt_image(self, title: str, width: int, height: int, rgba: bytes, meta: str):
        try:
            image_format = QImage.Format.Format_RGBA8888
        except AttributeError:
            image_format = QImage.Format_RGBA8888
        image = QImage(rgba, width, height, width * 4, image_format).copy()
        self.set_image(title, "TDT texture preview", image, meta)

    def preview_bytes(self, name: str, data: bytes):
        self.current_data = data
        lower = name.lower()
        kind = file_kind(name)

        if lower.endswith(".geo"):
            try:
                model = geo_format.parse_geo_model(data)
                face_count = sum(len(part.faces) for part in model.parts)
                part_names = "\n".join(f"- {part.name}: {len(part.faces):,} faces" for part in model.parts)
                self.clear()
                self.title.setText(os.path.basename(name))
                self.subtitle.setText("GEO model preview")
                self.mesh_view.set_model(model)
                self.stack.setCurrentWidget(self.mesh_view)
                self.meta.setPlainText(
                    f"{len(model.points):,} vertices\n"
                    f"{face_count:,} faces\n"
                    f"{len(model.parts)} part(s)\n\n"
                    f"{part_names}"
                )
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "GEO preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".wor"):
            try:
                subtitle, body, meta = build_wor_preview_task(name, data, {})
                self.set_text(os.path.basename(name), subtitle, body, meta)
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "WOR preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".obj"):
            try:
                subtitle, body, meta = build_obj_preview_task(name, data, {})
                self.set_text(os.path.basename(name), subtitle, body, meta)
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "OBJ preview failed", bytes_preview(data), str(exc))
                return
            
        # TODO: come up with better name for sidecar it sounds too stupid
        if lower.endswith(SIDECAR_PREVIEW_EXTS):
            try:
                subtitle, body, meta = build_sidecar_preview_task(name, data, {})
                self.set_text(os.path.basename(name), subtitle, body, meta)
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "Sidecar preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".mdf"):
            try:
                subtitle, body, meta = build_mdf_preview_task(name, data, {})
                self.set_text(os.path.basename(name), subtitle, body, meta)
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "MDF preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".tdt"):
            try:
                info = tdt.parse_tdt_texture_data(data, name)
                width, height, rgba = tdt.decode_tdt_top_mip_rgba_data(data, info)
                try:
                    image_format = QImage.Format.Format_RGBA8888
                except AttributeError:
                    image_format = QImage.Format_RGBA8888
                image = QImage(rgba, width, height, width * 4, image_format).copy()
                self.set_image(
                    os.path.basename(name),
                    "TDT texture preview",
                    image,
                    f"{width} x {height}\n{info.format_name}\n{len(info.levels)} mip(s)\n{len(data):,} bytes",
                )
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "TDT preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".skn"):
            try:
                subtitle, body, meta = build_skn_preview_task(name, data)
                self.set_text(os.path.basename(name), subtitle, body, meta)
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "SKN preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".trl"):
            try:
                subtitle, body, meta = build_trl_preview_task(name, data)
                self.set_text(os.path.basename(name), subtitle, body, meta)
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "TRL preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".oli"):
            try:
                subtitle, body, meta = build_oli_preview_task(name, data)
                self.set_text(os.path.basename(name), subtitle, body, meta)
                return
            except Exception as exc:
                self.set_text(os.path.basename(name), "OLI preview failed", bytes_preview(data), str(exc))
                return

        if lower.endswith(".son"):
            wav = previewers.extract_wav_from_son(data)
            if wav:
                meta = previewers.get_wav_metadata(wav)
                details = (
                    f"{meta['channels']} channel(s)\n"
                    f"{meta['sample_rate']} Hz\n"
                    f"{meta['sampwidth'] * 8} bit\n"
                    f"{meta['duration']:.2f} seconds\n"
                    f"{len(wav):,} byte embedded wav"
                ) if meta else f"{len(wav):,} byte embedded wav"
                self.set_text(os.path.basename(name), "SON audio preview", "embedded wav ready to play", details)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.write(wav)
                tmp.close()
                self.current_temp_audio = tmp.name
                self.player.setSource(QUrl.fromLocalFile(tmp.name))
                self.play_btn.setEnabled(True)
                self.pause_btn.setEnabled(True)
                return
            self.set_text(os.path.basename(name), "SON audio", bytes_preview(data), "No RIFF/WAVE payload found.")
            return

        self.set_text(os.path.basename(name), kind, bytes_preview(data), f"{len(data):,} bytes")


class GameDirectoryDialog(QDialog):
    def __init__(self, current_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ZOMBI Game Folder")
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Game directory")
        title.setObjectName("PanelTitle")
        hint = QLabel("Choose the folder that contains the game's Data folder. Picking Data itself also works.")
        hint.setObjectName("PanelSubtitle")
        hint.setWordWrap(True)

        self.path_edit = QLineEdit(current_path)
        self.path_edit.setPlaceholderText("/path/to/ZOMBI")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse)

        row = QHBoxLayout()
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse_btn)

        form = QFormLayout()
        form.addRow("Folder", row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def browse(self):
        start = self.path_edit.text().strip() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Select ZOMBI game folder", start)
        if path:
            self.path_edit.setText(path)

    def selected_game_dir(self) -> str:
        return self.path_edit.text().strip()

    def accept(self):
        normalized = normalized_game_dir(self.selected_game_dir())
        if not normalized:
            QMessageBox.warning(
                self,
                "Invalid folder",
                "That folder does not contain a Data directory with the game's archives.",
            )
            return
        self.path_edit.setText(normalized)
        super().accept()


# TODO: clean this part up, its fuckin' huge
class ZombiManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZOMBI Manager")
        self.setMinimumSize(QSize(1180, 760))
        self._apply_theme()
        self.archive: Optional[bfz.BFZArchive] = None
        self.current_archive_path: Optional[str] = None
        self.settings = load_app_settings()
        self.game_dir: Optional[str] = normalized_game_dir(self.settings.get("game_dir", ""))
        self.archive_infos: list[bfz.BFZArchiveInfo] = []
        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers = set()
        self.scan_token = 0
        self.archive_token = 0
        self.preview_token = 0
        self.preview_cache = {}
        self.texture_ref_cache_path: Optional[str] = None
        self.texture_ref_cache: Dict[int, list] = {}

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header = QHBoxLayout()
        self.game_path_label = QLabel("Game folder: not set")
        self.game_path_label.setObjectName("PanelSubtitle")
        self.change_game_btn = QPushButton("Game Folder")
        self.change_game_btn.clicked.connect(self.choose_game_directory)
        self.rescan_btn = QPushButton("Scan")
        self.rescan_btn.clicked.connect(self.scan_game_archives)
        header.addWidget(self.game_path_label, 1)
        header.addWidget(self.change_game_btn)
        header.addWidget(self.rescan_btn)
        root_layout.addLayout(header)

        self.main_stack = QStackedWidget()
        self.library_panel = self._make_library_panel()
        self.archive_panel = self._make_archive_panel()
        self.main_stack.addWidget(self.library_panel)
        self.main_stack.addWidget(self.archive_panel)
        root_layout.addWidget(self.main_stack, 1)

        self.setMenuBar(self._make_menu())
        self.setStatusBar(QStatusBar())
        self.update_game_path_label()
        QTimer.singleShot(0, self.startup_game_scan)

    def _apply_theme(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(38, 40, 44))
        palette.setColor(QPalette.Base, QColor(30, 32, 36))
        palette.setColor(QPalette.AlternateBase, QColor(42, 45, 50))
        palette.setColor(QPalette.Button, QColor(55, 59, 66))
        palette.setColor(QPalette.Text, QColor(232, 235, 240))
        palette.setColor(QPalette.ButtonText, QColor(232, 235, 240))
        palette.setColor(QPalette.WindowText, QColor(240, 242, 246))
        palette.setColor(QPalette.Highlight, QColor(88, 125, 215))
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #26282c; color: #eceff4; }
            QTreeWidget, QTextEdit, QLineEdit {
                background: #1e2024;
                border: 1px solid #3a3d44;
                border-radius: 5px;
                selection-background-color: #3f5f9f;
            }
            QPushButton {
                background: #373b42;
                border: 1px solid #4b5059;
                border-radius: 5px;
                padding: 7px 12px;
            }
            QPushButton:hover { background: #444955; }
            QPushButton:disabled { color: #7c828d; background: #2e3035; }
            QHeaderView::section {
                background: #30333a;
                color: #dce1ea;
                padding: 6px;
                border: none;
                border-right: 1px solid #464a52;
            }
            QLabel#PanelTitle, QLabel#PreviewTitle {
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#PanelSubtitle, QLabel#PreviewSubtitle {
                color: #aeb6c4;
            }
            QFrame#Panel {
                border: 1px solid #3a3d44;
                border-radius: 7px;
                background: #2b2e34;
            }
        """)

    def _make_library_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Game Archives")
        title.setObjectName("PanelTitle")
        self.library_subtitle = QLabel("choose a game folder to scan archives")
        self.library_subtitle.setObjectName("PanelSubtitle")

        self.library_search_box = QLineEdit()
        self.library_search_box.setPlaceholderText("filter archives...")
        self.library_search_box.textChanged.connect(self.apply_library_filter)

        self.open_btn = QPushButton("Open BFZ...")
        self.open_btn.clicked.connect(self.on_open)

        search_row = QHBoxLayout()
        search_row.addWidget(self.library_search_box, 1)
        search_row.addWidget(self.open_btn)

        self.library_tree = QTreeWidget()
        self.library_tree.setColumnCount(7)
        self.library_tree.setHeaderLabels(["Name", "BFZ", "Files", "World", "GEO", "Textures", "Size"])
        self.library_tree.setSortingEnabled(True)
        self.library_tree.setAlternatingRowColors(True)
        self.library_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.library_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 7):
            self.library_tree.header().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.library_tree.itemClicked.connect(self.on_archive_info_clicked)

        layout.addWidget(title)
        layout.addWidget(self.library_subtitle)
        layout.addLayout(search_row)
        layout.addWidget(self.library_tree, 1)
        return panel

    def _make_archive_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self.back_to_archives_btn = QPushButton("Archives")
        self.back_to_archives_btn.clicked.connect(self.show_archive_library)
        self.archive_title = QLabel("No archive loaded")
        self.archive_title.setObjectName("PanelTitle")
        self.archive_title.setWordWrap(True)
        self.export_all_btn = QPushButton("Export Archive")
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(self.export_all)
        header.addWidget(self.back_to_archives_btn)
        header.addWidget(self.archive_title, 1)
        header.addWidget(self.export_all_btn)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._make_browser_panel())
        self.preview = PreviewPane()
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([430, 740])

        layout.addLayout(header)
        layout.addWidget(splitter, 1)
        return panel

    def _make_browser_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        title = QLabel("Archive Contents")
        title.setObjectName("PanelTitle")
        self.browser_subtitle = QLabel("no archive loaded")
        self.browser_subtitle.setObjectName("PanelSubtitle")

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("filter files...")
        self.search_box.textChanged.connect(self.apply_filter)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["File", "Size", "Kind", "Key"])
        self.tree.setSortingEnabled(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_context_menu)

        layout.addWidget(title)
        layout.addWidget(self.browser_subtitle)
        layout.addWidget(self.search_box)
        layout.addWidget(self.tree, 1)
        return panel

    def _make_menu(self):
        menubar = QMenuBar()
        file_menu = menubar.addMenu("File")

        act_open = QAction("Open BFZ...", self)
        act_open.triggered.connect(self.on_open)
        file_menu.addAction(act_open)

        act_game = QAction("Set Game Folder...", self)
        act_game.triggered.connect(self.choose_game_directory)
        file_menu.addAction(act_game)

        act_scan = QAction("Scan Game Archives", self)
        act_scan.triggered.connect(self.scan_game_archives)
        file_menu.addAction(act_scan)

        file_menu.addSeparator()
        act_import = QAction("Import Folder to BFZ (disabled)", self)
        act_import.setEnabled(False)
        file_menu.addAction(act_import)

        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)
        return menubar

    def start_task(self, token: int, fn, on_finished, on_failed, *args):
        worker = TaskWorker(token, fn, *args)
        self.active_workers.add(worker)
        worker.signals.finished.connect(lambda _token, _result, w=worker: self.active_workers.discard(w))
        worker.signals.failed.connect(lambda _token, _error, w=worker: self.active_workers.discard(w))
        worker.signals.finished.connect(on_finished)
        worker.signals.failed.connect(on_failed)
        self.thread_pool.start(worker)

    def update_game_path_label(self):
        if self.game_dir:
            display = os.path.basename(os.path.normpath(self.game_dir)) or self.game_dir
            self.game_path_label.setText(f"Game folder: {display}")
            self.game_path_label.setToolTip(self.game_dir)
        else:
            self.game_path_label.setText("Game folder: not set")
            self.game_path_label.setToolTip("")

    def show_archive_library(self):
        self.main_stack.setCurrentWidget(self.library_panel)
        if self.archive:
            self.statusBar().showMessage(f"Loaded {os.path.basename(self.archive.path)}")
        else:
            self.statusBar().clearMessage()

    def startup_game_scan(self):
        if self.game_dir and game_data_dir(self.game_dir):
            self.scan_game_archives()
            return
        self.choose_game_directory(startup=True)

    def choose_game_directory(self, startup: bool = False):
        dialog = GameDirectoryDialog(self.game_dir or self.settings.get("game_dir", ""), self)
        if dialog.exec() != QDialog.Accepted:
            if startup and not self.game_dir:
                self.library_subtitle.setText("no game folder selected")
            return

        chosen = normalized_game_dir(dialog.selected_game_dir())
        if not chosen:
            return
        self.game_dir = chosen
        self.settings["game_dir"] = chosen
        save_app_settings(self.settings)
        self.update_game_path_label()
        self.scan_game_archives()

    def scan_game_archives(self):
        if not self.game_dir or not game_data_dir(self.game_dir):
            self.choose_game_directory()
            return
        self.scan_token += 1
        token = self.scan_token
        self.main_stack.setCurrentWidget(self.library_panel)
        self.library_tree.clear()
        self.library_subtitle.setText("scanning Data archives...")
        self.statusBar().showMessage("Scanning game archives...")
        self.rescan_btn.setEnabled(False)
        self.start_task(token, scan_game_archives_task, self.on_scan_finished, self.on_scan_failed, self.game_dir)

    def on_scan_finished(self, token: int, archive_infos):
        if token != self.scan_token:
            return
        self.rescan_btn.setEnabled(True)
        self.archive_infos = list(archive_infos)
        self.populate_library()
        self.settings["game_dir"] = self.game_dir
        self.settings["last_scan_count"] = len(self.archive_infos)
        save_app_settings(self.settings)
        self.statusBar().showMessage(f"Scanned {len(self.archive_infos):,} archive(s)")

    def on_scan_failed(self, token: int, error_text: str):
        if token != self.scan_token:
            return
        self.rescan_btn.setEnabled(True)
        self.library_subtitle.setText("archive scan failed")
        self.statusBar().clearMessage()
        QMessageBox.critical(self, "Scan failed", f"Failed to scan game archives:\n\n{error_text}")

    def populate_library(self):
        self.library_tree.setUpdatesEnabled(False)
        self.library_tree.setSortingEnabled(False)
        self.library_tree.clear()
        try:
            for info in self.archive_infos:
                ext = info.extension_counts
                texture_count = ext.get(".tdt", 0) + ext.get(".tex", 0) + ext.get(".mta", 0)
                world_count = ext.get(".wor", 0)
                item = QTreeWidgetItem([
                    info.display_name,
                    os.path.basename(info.relative_path),
                    f"{info.file_count:,}",
                    f"{world_count:,}" if world_count else "",
                    f"{ext.get('.geo', 0):,}" if ext.get(".geo", 0) else "",
                    f"{texture_count:,}" if texture_count else "",
                    format_bytes(info.archive_size),
                ])
                item.setData(0, Qt.UserRole, info.path)
                item.setToolTip(0, self.archive_info_tooltip(info))
                item.setToolTip(1, info.relative_path)
                self.library_tree.addTopLevelItem(item)

            self.library_tree.setSortingEnabled(True)
            self.library_tree.sortItems(0, Qt.AscendingOrder)
            self.library_subtitle.setText(f"{len(self.archive_infos):,} archive(s) found")
            self.apply_library_filter(self.library_search_box.text())
        finally:
            self.library_tree.setUpdatesEnabled(True)
            self.library_tree.setSortingEnabled(True)

    def archive_info_tooltip(self, info: bfz.BFZArchiveInfo) -> str:
        ext = info.extension_counts
        top_counts = ", ".join(
            f"{key} {value:,}"
            for key, value in sorted(ext.items(), key=lambda item: (-item[1], item[0]))[:8]
        )
        worlds = ", ".join(info.world_names[:4])
        if len(info.world_names) > 4:
            worlds += f", +{len(info.world_names) - 4} more"
        return (
            f"{info.relative_path}\n"
            f"Name: {info.display_name}\n"
            f"Files: {info.file_count:,}\n"
            f"Unique keys: {info.key_count:,}\n"
            f"Duplicate paths: {info.duplicate_name_count:,}\n"
            f"Archive size: {format_bytes(info.archive_size)}\n"
            f"Unpacked size: {format_bytes(info.unpacked_size)}\n"
            f"Worlds: {worlds or 'none'}\n"
            f"Types: {top_counts or 'none'}"
        )

    def archive_info_for_path(self, path: str) -> Optional[bfz.BFZArchiveInfo]:
        absolute = os.path.abspath(path)
        return next((info for info in self.archive_infos if os.path.abspath(info.path) == absolute), None)

    def apply_library_filter(self, text: str):
        query = text.strip().lower()
        for index in range(self.library_tree.topLevelItemCount()):
            item = self.library_tree.topLevelItem(index)
            path = item.data(0, Qt.UserRole) or ""
            info = next((candidate for candidate in self.archive_infos if candidate.path == path), None)
            haystack = " ".join(item.text(column).lower() for column in range(item.columnCount()))
            if info:
                haystack += " " + info.relative_path.lower() + " " + " ".join(name.lower() for name in info.world_names)
            item.setHidden(bool(query) and query not in haystack)

    def on_archive_info_clicked(self, item: QTreeWidgetItem, _col: int):
        path = item.data(0, Qt.UserRole)
        if path:
            self.load_archive(path)

    def load_archive(self, path: str):
        self.archive_token += 1
        token = self.archive_token
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")
        self.archive_title.setText(os.path.basename(path))
        self.browser_subtitle.setText(f"loading {os.path.basename(path)}...")
        self.export_all_btn.setEnabled(False)
        self.tree.clear()
        self.preview.clear()
        self.main_stack.setCurrentWidget(self.archive_panel)
        self.start_task(token, parse_archive_task, self.on_archive_loaded, self.on_archive_failed, path)

    def remember_preview(self, cache_key, result):
        self.preview_cache[cache_key] = result
        result_keys = [key for key in self.preview_cache if isinstance(key, tuple)]
        while len(result_keys) > MAX_PREVIEW_CACHE_ITEMS:
            oldest = result_keys.pop(0)
            self.preview_cache.pop(oldest, None)

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open BFZ Archive", "", "BFZ Archives (*.bfz);;All Files (*)")
        if not path:
            return
        self.load_archive(path)

    def on_archive_loaded(self, token: int, archive: bfz.BFZArchive):
        if token != self.archive_token:
            return
        self.archive = archive
        self.current_archive_path = archive.path
        self.texture_ref_cache_path = None
        self.texture_ref_cache = {}
        self.preview_cache.clear()
        self.preview_token += 1
        self.preview.clear()
        self.populate_tree()
        self.export_all_btn.setEnabled(True)
        info = self.archive_info_for_path(archive.path)
        self.archive_title.setText(info.display_name if info else os.path.basename(archive.path))
        self.browser_subtitle.setText(f"{os.path.basename(archive.path)} · {len(archive.file_entries):,} files")
        self.statusBar().showMessage(f"Loaded {os.path.basename(archive.path)}")
        self.setWindowTitle(f"ZOMBI Manager - {os.path.basename(archive.path)}")

    def on_archive_failed(self, token: int, error_text: str):
        if token != self.archive_token:
            return
        self.statusBar().clearMessage()
        self.export_all_btn.setEnabled(bool(self.archive))
        self.browser_subtitle.setText("failed to load archive")
        QMessageBox.critical(self, "Open failed", f"Failed to load archive:\n\n{error_text}")

    def populate_tree(self):
        self.tree.setUpdatesEnabled(False)
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        try:
            if not self.archive:
                return

            root_map: Dict[tuple, QTreeWidgetItem] = {}
            grouped: Dict[str, list] = {}
            for entry in self.archive.file_entries:
                path = entry.name.replace("\\", "/").strip("/")
                grouped.setdefault(path, []).append(entry)

            for path, entries in grouped.items():
                parts = [part for part in path.split("/") if part]
                if not parts:
                    continue

                parent_item = None
                for index, part in enumerate(parts):
                    is_leaf = index == len(parts) - 1
                    key = (id(parent_item), part)
                    if key not in root_map:
                        if is_leaf:
                            columns = [
                                part,
                                f"{entries[0].size:,}",
                                file_kind(entries[0].name),
                                entry_key_text(entries[0]),
                            ]
                        else:
                            columns = [part, "", "Folder", ""]
                        item = QTreeWidgetItem(columns)
                        if parent_item is None:
                            self.tree.addTopLevelItem(item)
                        else:
                            parent_item.addChild(item)
                        root_map[key] = item
                    item = root_map[key]
                    parent_item = item

                    if is_leaf:
                        if len(entries) == 1:
                            item.setData(0, Qt.UserRole, entries[0])
                        else:
                            item.setText(1, "")
                            item.setText(2, f"{len(entries)} variants")
                            for dup_index, dup_entry in enumerate(entries):
                                sub = QTreeWidgetItem([
                                    f"Variant {dup_index + 1}",
                                    f"{dup_entry.size:,}",
                                    file_kind(dup_entry.name),
                                    entry_key_text(dup_entry),
                                ])
                                sub.setData(0, Qt.UserRole, dup_entry)
                                item.addChild(sub)

            self.tree.setSortingEnabled(True)
            self.tree.sortItems(0, Qt.AscendingOrder)
            self.tree.expandToDepth(1)
            self.apply_filter(self.search_box.text())
        finally:
            self.tree.setSortingEnabled(True)
            self.tree.setUpdatesEnabled(True)

    def apply_filter(self, text: str):
        query = text.strip().lower()

        def update_item(item: QTreeWidgetItem) -> bool:
            entry = item.data(0, Qt.UserRole)
            own_text = " ".join(item.text(column).lower() for column in range(item.columnCount()))
            if entry:
                own_text += " " + entry.name.lower()
            own_match = not query or query in own_text
            child_match = False
            for index in range(item.childCount()):
                child_match = update_item(item.child(index)) or child_match
            visible = own_match or child_match
            item.setHidden(not visible)
            if child_match and query:
                item.setExpanded(True)
            return visible

        for index in range(self.tree.topLevelItemCount()):
            update_item(self.tree.topLevelItem(index))

    def archive_key_map(self) -> Dict[int, list]:
        if not self.archive:
            return {}
        key_map: Dict[int, list] = {}
        for archive_entry in self.archive.file_entries:
            if archive_entry.key:
                key_map.setdefault(archive_entry.key, []).append(archive_entry.name)
        return key_map

    def archive_texture_ref_map(self) -> Dict[int, list]:
        if not self.archive:
            return {}
        if self.texture_ref_cache_path == self.archive.path:
            return self.texture_ref_cache

        ref_map: Dict[int, list] = {}
        for archive_entry in self.archive.file_entries:
            if not archive_entry.name.lower().endswith(".tex"):
                continue
            try:
                descriptor = material_format.parse_tex(
                    self.archive.read_file_bytes(archive_entry),
                    archive_entry.name,
                )
            except Exception:
                continue
            if descriptor.file_key is None:
                continue
            ref_map.setdefault(descriptor.file_key & 0xFFFFFFFF, []).append(archive_entry.name)
            ref_map.setdefault((descriptor.file_key - 1) & 0xFFFFFFFF, []).append(archive_entry.name)

        self.texture_ref_cache_path = self.archive.path
        self.texture_ref_cache = ref_map
        return ref_map

    def selected_entry(self) -> Optional[bfz.BFZFileEntry]:
        item = self.tree.currentItem()
        if not item:
            return None
        return item.data(0, Qt.UserRole)

    def on_item_clicked(self, item: QTreeWidgetItem, col: int):
        entry = item.data(0, Qt.UserRole)
        if not entry or not self.archive:
            self.preview.clear()
            return
        try:
            data = self.archive.read_file_bytes(entry)
            self.preview.current_entry = entry
            self.statusBar().showMessage(f"{entry.name} - {entry.size:,} bytes")
            lower = entry.name.lower()
            cache_key = (self.current_archive_path, entry.name, entry.offset, entry.size)
            self.preview_token += 1
            token = self.preview_token

            if lower.endswith(".geo"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    model, meta = cached
                    self.preview.set_geo_model(os.path.basename(entry.name), model, meta)
                    return
                self.preview.set_loading(os.path.basename(entry.name), "loading GEO model...")
                self.start_task(token, build_geo_preview_task, self.on_geo_preview_ready, self.on_preview_failed, data)
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(".tdt"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    width, height, rgba, meta = cached
                    self.preview.set_tdt_image(os.path.basename(entry.name), width, height, rgba, meta)
                    return
                self.preview.set_loading(os.path.basename(entry.name), "decoding TDT texture...")
                self.start_task(token, build_tdt_preview_task, self.on_tdt_preview_ready, self.on_preview_failed, entry.name, data)
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(".wor"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    subtitle, body, meta = cached
                    self.preview.set_text(os.path.basename(entry.name), subtitle, body, meta)
                    return
                key_map = self.archive_key_map()
                self.preview.set_loading(os.path.basename(entry.name), "reading world refs...")
                self.start_task(token, build_wor_preview_task, self.on_text_preview_ready, self.on_preview_failed, entry.name, data, key_map)
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(".obj"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    subtitle, body, meta = cached
                    self.preview.set_text(os.path.basename(entry.name), subtitle, body, meta)
                    return
                self.preview.set_loading(os.path.basename(entry.name), "reading game object...")
                self.start_task(
                    token,
                    build_obj_preview_task,
                    self.on_text_preview_ready,
                    self.on_preview_failed,
                    entry.name,
                    data,
                    self.archive_key_map(),
                )
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(SIDECAR_PREVIEW_EXTS):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    subtitle, body, meta = cached
                    self.preview.set_text(os.path.basename(entry.name), subtitle, body, meta)
                    return
                texture_ref_map = self.archive_texture_ref_map() if lower.endswith(".mta") else {}
                self.preview.set_loading(os.path.basename(entry.name), "reading sidecar data...")
                self.start_task(
                    token,
                    build_sidecar_preview_task,
                    self.on_text_preview_ready,
                    self.on_preview_failed,
                    entry.name,
                    data,
                    self.archive_key_map(),
                    texture_ref_map,
                )
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(".mdf"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    subtitle, body, meta = cached
                    self.preview.set_text(os.path.basename(entry.name), subtitle, body, meta)
                    return
                self.preview.set_loading(os.path.basename(entry.name), "reading modifier data...")
                self.start_task(
                    token,
                    build_mdf_preview_task,
                    self.on_text_preview_ready,
                    self.on_preview_failed,
                    entry.name,
                    data,
                    self.archive_key_map(),
                )
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(".skn"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    subtitle, body, meta = cached
                    self.preview.set_text(os.path.basename(entry.name), subtitle, body, meta)
                    return
                self.preview.set_loading(os.path.basename(entry.name), "reading skeleton...")
                self.start_task(token, build_skn_preview_task, self.on_text_preview_ready, self.on_preview_failed, entry.name, data)
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(".trl"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    subtitle, body, meta = cached
                    self.preview.set_text(os.path.basename(entry.name), subtitle, body, meta)
                    return
                self.preview.set_loading(os.path.basename(entry.name), "reading animation track...")
                self.start_task(token, build_trl_preview_task, self.on_text_preview_ready, self.on_preview_failed, entry.name, data)
                self.preview_cache[token] = cache_key
                return

            if lower.endswith(".oli"):
                cached = self.preview_cache.get(cache_key)
                if cached:
                    subtitle, body, meta = cached
                    self.preview.set_text(os.path.basename(entry.name), subtitle, body, meta)
                    return
                self.preview.set_loading(os.path.basename(entry.name), "reading localization...")
                self.start_task(token, build_oli_preview_task, self.on_text_preview_ready, self.on_preview_failed, entry.name, data)
                self.preview_cache[token] = cache_key
                return

            self.preview.preview_bytes(entry.name, data)
        except Exception as exc:
            QMessageBox.critical(self, "Preview failed", f"Failed to read file bytes:\n{exc}")

    def on_geo_preview_ready(self, token: int, result):
        if token != self.preview_token:
            self.preview_cache.pop(token, None)
            return
        cache_key = self.preview_cache.pop(token, None)
        model, meta = result
        if cache_key:
            self.remember_preview(cache_key, result)
        entry = self.selected_entry()
        title = os.path.basename(entry.name) if entry else "GEO model"
        self.preview.set_geo_model(title, model, meta)

    def on_tdt_preview_ready(self, token: int, result):
        if token != self.preview_token:
            self.preview_cache.pop(token, None)
            return
        cache_key = self.preview_cache.pop(token, None)
        if cache_key:
            self.remember_preview(cache_key, result)
        entry = self.selected_entry()
        title = os.path.basename(entry.name) if entry else "TDT texture"
        width, height, rgba, meta = result
        self.preview.set_tdt_image(title, width, height, rgba, meta)

    def on_text_preview_ready(self, token: int, result):
        if token != self.preview_token:
            self.preview_cache.pop(token, None)
            return
        cache_key = self.preview_cache.pop(token, None)
        if cache_key:
            self.remember_preview(cache_key, result)
        entry = self.selected_entry()
        title = os.path.basename(entry.name) if entry else "Preview"
        subtitle, body, meta = result
        self.preview.set_text(title, subtitle, body, meta)

    def on_preview_failed(self, token: int, error_text: str):
        if token != self.preview_token:
            self.preview_cache.pop(token, None)
            return
        self.preview_cache.pop(token, None)
        entry = self.selected_entry()
        title = os.path.basename(entry.name) if entry else "Preview"
        self.preview.set_text(title, "Preview failed", "", error_text)

    def on_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return
        entry = item.data(0, Qt.UserRole)
        if not entry:
            return

        lower = entry.name.lower()
        menu = QMenu(self)
        act_export = QAction("Export original bytes...", self)
        act_export.triggered.connect(lambda: self.export_original(entry))
        menu.addAction(act_export)

        if lower.endswith(".son"):
            act_wav = QAction("Export embedded WAV...", self)
            act_wav.triggered.connect(lambda: self.export_converted(entry))
            menu.addAction(act_wav)
        elif lower.endswith(".tdt"):
            act_png = QAction("Export texture as PNG...", self)
            act_png.triggered.connect(lambda: self.export_converted(entry))
            menu.addAction(act_png)
        elif lower.endswith(".oli"):
            act_csv = QAction("Export localization CSV...", self)
            act_csv.triggered.connect(lambda: self.export_converted(entry))
            menu.addAction(act_csv)

        menu.addSeparator()
        act_copy = QAction("Copy archive path", self)
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(entry.name))
        menu.addAction(act_copy)
        menu.exec(self.tree.mapToGlobal(pos))

    def export_original(self, entry: bfz.BFZFileEntry):
        if not self.archive:
            return
        default_name = os.path.basename(entry.name) or "file.bin"
        path, _ = QFileDialog.getSaveFileName(self, "Export original bytes", default_name)
        if not path:
            return
        try:
            with open(path, "wb") as handle:
                handle.write(self.archive.read_file_bytes(entry))
            QMessageBox.information(self, "Exported", f"Exported original file:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Failed to export:\n{exc}\n\n{traceback.format_exc()}")

    def export_converted(self, entry: bfz.BFZFileEntry):
        if not self.archive:
            return
        lower = entry.name.lower()
        if lower.endswith(".son"):
            default_name = converted_stem(entry.name) + ".wav"
            path, _ = QFileDialog.getSaveFileName(self, "Export embedded WAV", default_name, "WAV Audio (*.wav)")
            if not path:
                return
            try:
                wav = previewers.extract_wav_from_son(self.archive.read_file_bytes(entry))
                if not wav:
                    raise RuntimeError("No embedded RIFF/WAVE payload found.")
                if not path.lower().endswith(".wav"):
                    path += ".wav"
                with open(path, "wb") as handle:
                    handle.write(wav)
                QMessageBox.information(self, "Exported", f"Exported WAV:\n{path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", f"Failed to export WAV:\n{exc}")
            return

        if lower.endswith(".tdt"):
            default_name = converted_stem(entry.name) + ".png"
            path, _ = QFileDialog.getSaveFileName(self, "Export texture as PNG", default_name, "PNG Image (*.png)")
            if not path:
                return
            try:
                if not path.lower().endswith(".png"):
                    path += ".png"
                tdt.write_tdt_data_as_png(self.archive.read_file_bytes(entry), path)
                QMessageBox.information(self, "Exported", f"Exported PNG:\n{path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", f"Failed to export PNG:\n{exc}\n\n{traceback.format_exc()}")
            return

        if lower.endswith(".oli"):
            default_name = converted_stem(entry.name) + ".csv"
            path, _ = QFileDialog.getSaveFileName(self, "Export localization CSV", default_name, "CSV (*.csv)")
            if not path:
                return
            try:
                if not path.lower().endswith(".csv"):
                    path += ".csv"
                oli.write_oli_csv(self.archive.read_file_bytes(entry), path, entry.name, force=True)
                QMessageBox.information(self, "Exported", f"Exported CSV:\n{path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", f"Failed to export CSV:\n{exc}\n\n{traceback.format_exc()}")

    def export_all(self):
        if not self.archive:
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return
        try:
            progress = QProgressDialog("Exporting archive...", "Cancel", 0, len(self.archive.file_entries) + 1, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            exported_entries = []
            export_paths = self.archive.export_path_map()
            canceled = False
            for index, entry in enumerate(self.archive.file_entries):
                data = self.archive.read_file_bytes(entry)
                relative_path = export_paths.get(entry.index, self.archive.normalized_archive_path(entry.name))
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as handle:
                    handle.write(data)
                exported_entries.append(entry)
                progress.setValue(index + 1)
                QApplication.processEvents()
                if progress.wasCanceled():
                    canceled = True
                    break
            manifest_path = ""
            if exported_entries:
                progress.setLabelText("Writing manifest...")
                manifest_path = self.archive.write_manifest(output_dir, exported_entries, export_paths)
                progress.setValue(len(self.archive.file_entries) + 1)
                QApplication.processEvents()
            progress.close()
            if canceled:
                QMessageBox.information(
                    self,
                    "Export canceled",
                    f"Exported {len(exported_entries):,} file(s) before canceling.\nManifest:\n{manifest_path}",
                )
            else:
                QMessageBox.information(
                    self,
                    "Done",
                    f"Exported {len(exported_entries):,} files.\nManifest:\n{manifest_path}",
                )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Failed to export archive:\n{exc}\n\n{traceback.format_exc()}")


def main():
    app = QApplication(sys.argv)
    win = ZombiManager()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
