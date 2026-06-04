from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


OBJ_MAGIC = b"OBJ_"
VIS_MAGIC = b"VIS\x00"
MOTI_MAGIC = b"MOTI"
MAT_MAGIC = b"MAT_"
MTA_MAGIC = b"MTA\x00"
OBJ_REF_GRAPHIC = 0x02
OBJ_REF_MATERIAL = 0x03
RESOURCE_KEY_HIGH_BYTES = {
    0x06,
    0x1E,
    0x68,
    0x7C,
    0x81,
    0x9D,
    0xAD,
    0xAF,
    0xDA,
    0xDC,
    0xE3,
    0xF6,
}


@dataclass(frozen=True)
class ObjMatrix:
    offset: int
    rows: Tuple[Tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class ObjectResourceRef:
    tag: int
    key: int
    flags: int
    offset: int

    @property
    def key_hex(self) -> str:
        return f"{self.key & 0xffffffff:08X}"


@dataclass(frozen=True)
class ObjFile:
    name: str
    stored_size: int
    version: int
    editor_flags: int
    identity_flags: int
    status_mask: int
    matrices: Tuple[ObjMatrix, ...]
    translation: Tuple[float, float, float]
    translation_offset: int
    resource_keys: Tuple[int, ...]
    resource_refs: Tuple[ObjectResourceRef, ...]

    @property
    def matrix(self) -> ObjMatrix:
        return self.matrices[-1]

    @property
    def key_hex(self) -> Tuple[str, ...]:
        return tuple(f"{key & 0xffffffff:08X}" for key in self.resource_keys)

    @property
    def graphic_refs(self) -> Tuple[ObjectResourceRef, ...]:
        return tuple(ref for ref in self.resource_refs if ref.tag == OBJ_REF_GRAPHIC)

    @property
    def primary_graphic_ref(self) -> ObjectResourceRef | None:
        refs = self.graphic_refs
        return refs[0] if refs else None

    @property
    def primary_visual_ref(self) -> ObjectResourceRef | None:
        # Kept for older addon code. In the Jade object save/load code this is
        # the GRO graphic-object ref, not a final packed GEO file ref.
        return self.primary_graphic_ref

    @property
    def primary_resource_ref(self) -> ObjectResourceRef | None:
        for ref in self.resource_refs:
            if ref.tag in (0x02, 0x01, 0x04):
                return ref
        return None

    @property
    def material_refs(self) -> Tuple[ObjectResourceRef, ...]:
        return tuple(ref for ref in self.resource_refs if ref.tag == OBJ_REF_MATERIAL)


@dataclass(frozen=True)
class SidecarFile:
    name: str
    kind: str
    stored_size: int
    resource_keys: Tuple[int, ...]

    @property
    def key_hex(self) -> Tuple[str, ...]:
        return tuple(f"{key & 0xffffffff:08X}" for key in self.resource_keys)


def _u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        return 0
    return struct.unpack_from("<I", data, offset)[0]


def _score_affine_matrix(values: Sequence[float]) -> float:
    rows = [values[index : index + 4] for index in range(0, 16, 4)]
    basis = [row[:3] for row in rows[:3]]
    lengths = [math.sqrt(sum(component * component for component in axis)) for axis in basis]
    dots = [
        sum(basis[left][axis] * basis[right][axis] for axis in range(3))
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    return (
        sum(abs(length - 1.0) for length in lengths)
        + sum(abs(dot) for dot in dots)
        + sum(abs(rows[index][3]) for index in range(3))
        + abs(rows[3][3] - 1.0)
    )


def _find_matrices(data: bytes) -> List[ObjMatrix]:
    matrices: List[ObjMatrix] = []
    for offset in range(0, max(0, len(data) - 64 + 1)):
        values = struct.unpack_from("<16f", data, offset)
        if not all(math.isfinite(value) and abs(value) < 1_000_000.0 for value in values):
            continue
        if _score_affine_matrix(values) > 0.01:
            continue
        rows = tuple(tuple(values[index : index + 4]) for index in range(0, 16, 4))
        matrices.append(ObjMatrix(offset, rows))

    # ZombiU object records usually store local and absolute matrices at 0x9d and
    # 0xdd. Keep all candidates, but drop accidental overlaps if a weird file has any.
    filtered: List[ObjMatrix] = []
    for matrix in matrices:
        if filtered and matrix.offset - filtered[-1].offset < 16:
            continue
        filtered.append(matrix)
    return filtered


def _read_translation(data: bytes, matrix: ObjMatrix) -> Tuple[Tuple[float, float, float], int]:
    offset = matrix.offset + 64
    if offset + 12 <= len(data):
        values = struct.unpack_from("<3f", data, offset)
        if all(math.isfinite(value) and abs(value) < 1_000_000.0 for value in values):
            return (values[0], values[1], values[2]), offset
    return (0.0, 0.0, 0.0), -1


def extract_object_resource_refs(data: bytes, translation_offset: int) -> Tuple[ObjectResourceRef, ...]:
    if translation_offset < 0:
        return ()

    # Modern LyN object files keep a compact tagged ref list after the saved
    # matrix translation and two zero fields. Tag 0x02 is the GRO graphic object
    # and tag 0x03 is the material, matching OBJsave/OBJload's Visu block.
    cursor = translation_offset + 20
    refs: List[ObjectResourceRef] = []
    for _index in range(64):
        if cursor + 4 > len(data):
            break
        tag_offset = cursor
        tag = _u32(data, cursor)
        cursor += 4
        if tag in (0x44, 0x45, 0xFFFFFFFF):
            break
        if cursor + 8 > len(data):
            break
        key = _u32(data, cursor)
        flags = _u32(data, cursor + 4)
        cursor += 8
        if tag == 0 and key == 0 and flags == 0:
            break
        if tag > 0x200:
            break
        refs.append(ObjectResourceRef(tag, key, flags, tag_offset))
    return tuple(refs)


def extract_resource_keys(data: bytes) -> Tuple[int, ...]:
    keys: List[int] = []
    seen = set()
    for offset in range(0, max(0, len(data) - 3)):
        key = _u32(data, offset)
        high = (key >> 24) & 0xFF
        if high not in RESOURCE_KEY_HIGH_BYTES:
            continue
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def _append_key(keys: List[int], seen: set[int], value: int) -> None:
    if value in seen or value in {0, 0xFFFFFFFF, 0x10000001}:
        return
    low = value & 0x00FFFFFF
    if low in {0, 0x00FFFFFF}:
        return
    high = (value >> 24) & 0xFF
    if high == 0:
        return
    seen.add(value)
    keys.append(value)


def extract_sidecar_resource_keys(data: bytes, magic: bytes) -> Tuple[int, ...]:
    # VIS and MOTI keep their primary resource links in stable unaligned fields.
    # Scanning still helps with odd files, but these offsets catch real keys with
    # high bytes outside the old scanner filter, such as 57D0xxxx and CED0xxxx.
    keys: List[int] = []
    seen: set[int] = set()
    if magic == VIS_MAGIC:
        offsets = (0x36, 0x92)
    elif magic == MOTI_MAGIC:
        offsets = (0x36, 0xDE, 0x152, 0x15E)
    else:
        offsets = ()

    for offset in offsets:
        if offset + 4 <= len(data):
            _append_key(keys, seen, _u32(data, offset))

    for key in extract_resource_keys(data):
        _append_key(keys, seen, key)
    return tuple(keys)


def parse_obj(data: bytes, name: str = "") -> ObjFile:
    if len(data) < 0x18 or data[0x14:0x18] != OBJ_MAGIC:
        raise ValueError("missing OBJ_ magic")

    matrices = tuple(_find_matrices(data))
    if not matrices:
        raise ValueError("no object matrix found")

    translation, translation_offset = _read_translation(data, matrices[-1])
    resource_refs = extract_object_resource_refs(data, translation_offset)
    return ObjFile(
        name=name,
        stored_size=len(data),
        version=_u32(data, 0x18),
        editor_flags=_u32(data, 0x1C),
        identity_flags=_u32(data, 0x20),
        status_mask=_u32(data, 0x24),
        matrices=matrices,
        translation=translation,
        translation_offset=translation_offset,
        resource_keys=extract_resource_keys(data),
        resource_refs=resource_refs,
    )


def parse_obj_file(path: str) -> ObjFile:
    with open(path, "rb") as handle:
        return parse_obj(handle.read(), os.path.basename(path))


def parse_sidecar(data: bytes, name: str = "") -> SidecarFile:
    if len(data) < 0x18:
        raise ValueError("file is too small")
    magic = data[0x14:0x18]
    if magic == VIS_MAGIC:
        kind = "VIS"
    elif magic == MOTI_MAGIC:
        kind = "MOTI"
    elif magic == MAT_MAGIC:
        kind = "MAT"
    elif magic == MTA_MAGIC:
        kind = "MTA"
    else:
        raise ValueError("unknown sidecar magic")
    return SidecarFile(name, kind, len(data), extract_sidecar_resource_keys(data, magic))


def parse_sidecar_file(path: str) -> SidecarFile:
    with open(path, "rb") as handle:
        return parse_sidecar(handle.read(), os.path.basename(path))


def find_sidecar(path: str, extension: str) -> Optional[str]:
    base, _ = os.path.splitext(path)
    directory = os.path.dirname(path)
    stem = os.path.basename(base)

    stems = [stem]
    variant_match = re.match(r"^(.*)__variant_([0-9]+)$", stem)
    if variant_match:
        stems.append(variant_match.group(1))

    for candidate_stem in list(stems):
        if candidate_stem.startswith("SH_"):
            stripped = candidate_stem[3:]
            if stripped not in stems:
                stems.append(stripped)

    for candidate_stem in stems:
        candidate = os.path.join(directory, candidate_stem + extension)
        if os.path.exists(candidate):
            return candidate
    return None
