from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterable, Tuple


MAT_MAGIC = b"MAT_"
MTA_MAGIC = b"MTA\x00"
TEX_MAGIC = b"TEX_"
TEX_TAIL_MARKER = 0x10000001
TEX_FORMAT_NAMES = {
    0x00: "RGBA8",
    0x09: "BC1/DXT1",
    0x0B: "BC3/DXT5",
    0x13: "L16/AO",
    0x1F: "BC4/ATI1",
    0x20: "BC5/ATI2",
}


@dataclass(frozen=True)
class MatDescriptor:
    name: str
    submaterial_keys: Tuple[int, ...]


@dataclass(frozen=True)
class MtaTextureSlot:
    index: int
    offset: int
    texture_ref: int
    transform: Tuple[float, float, float, float, float]
    flags: int


@dataclass(frozen=True)
class MtaDescriptor:
    name: str
    shader_name: str
    layer_count: int
    texture_slots: Tuple[MtaTextureSlot, ...]
    texture_refs: Tuple[int, ...]
    primary_texture_refs: Tuple[int, ...]
    extra_texture_refs: Tuple[int, ...]


@dataclass(frozen=True)
class TexDescriptor:
    name: str
    file_key: int | None
    width: int | None = None
    height: int | None = None
    format_code: int | None = None
    format_name: str = ""


def _u16(data: bytes, offset: int) -> int:
    if offset + 2 > len(data):
        return 0
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        return 0
    return struct.unpack_from("<I", data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    if offset + 4 > len(data):
        return 0.0
    return struct.unpack_from("<f", data, offset)[0]


def _cstr(data: bytes, offset: int, size: int) -> str:
    if offset >= len(data):
        return ""
    raw = data[offset:min(len(data), offset + size)]
    raw = raw.split(b"\x00", 1)[0]
    return raw.decode("utf-8", "replace").strip()


def _looks_like_ref(value: int) -> bool:
    if value in {
        0,
        1,
        2,
        3,
        4,
        0xFFFFFFFF,
        0x0000FFFF,
        TEX_TAIL_MARKER,
        0x3F000000,
        0x3F800000,
    }:
        return False
    return True


def _unique_refs(*groups) -> Tuple[int, ...]:
    seen = set()
    refs = []
    for group in groups:
        for value in group:
            value &= 0xFFFFFFFF
            if value in seen:
                continue
            seen.add(value)
            refs.append(value)
    return tuple(refs)


def parse_mat(data: bytes, name: str = "") -> MatDescriptor:
    if len(data) < 0x28 or data[0x14:0x18] != MAT_MAGIC:
        raise ValueError("missing MAT_ magic")

    count = _u16(data, 0x26)
    if count > 256 or 0x28 + count * 4 > len(data):
        count = 0

    keys = tuple(_u32(data, 0x28 + index * 4) for index in range(count))
    return MatDescriptor(name=name, submaterial_keys=keys)


def parse_mat_file(path: str) -> MatDescriptor:
    with open(path, "rb") as handle:
        return parse_mat(handle.read(), os.path.basename(path))


def mat_key_to_mta_stem_map(mat_paths: Iterable[str], mta_paths: Iterable[str]) -> dict[int, str]:
    """
    TODO: Clean this up. 
    This function specifically was helped with AI, it works but I feel like it could be better.
    (Only use of AI in this project besides the horrible ass README)
    """

    material_keys = []
    seen = set()
    for path in mat_paths:
        try:
            descriptor = parse_mat_file(path)
        except Exception:
            continue
        for key in descriptor.submaterial_keys:
            key &= 0xFFFFFFFF
            if key in seen:
                continue
            seen.add(key)
            material_keys.append(key)

    mta_stems = [os.path.splitext(os.path.basename(path))[0] for path in mta_paths]
    if not material_keys or len(mta_stems) != len(material_keys):
        return {}

    return {
        key: mta_stems[index]
        for index, key in enumerate(material_keys)
    }


def mat_link_keys(mat_paths: Iterable[str]) -> set[int]:
    """Return sidecar link keys that point at MAT slots instead of GEO files."""

    keys: set[int] = set()
    for path in mat_paths:
        try:
            descriptor = parse_mat_file(path)
        except Exception:
            continue
        for key in descriptor.submaterial_keys:
            keys.add((key + 1) & 0xFFFFFFFF)
    return keys


def parse_mta(data: bytes, name: str = "") -> MtaDescriptor:
    if len(data) < 0x18 or data[0x14:0x18] != MTA_MAGIC:
        raise ValueError("missing MTA magic")

    raw_layer_count = _u16(data, 0x36)
    layer_count = raw_layer_count if raw_layer_count <= 64 else 0
    shader_name = _cstr(data, 0x38, 64)

    slots = []
    for index in range(16):
        offset = 0x94 + index * 0x1C
        if offset + 0x1C > len(data):
            break
        value = _u32(data, offset)
        if _looks_like_ref(value):
            slots.append(
                MtaTextureSlot(
                    index=index,
                    offset=offset,
                    texture_ref=value,
                    transform=tuple(_f32(data, offset + 4 + axis * 4) for axis in range(5)),
                    flags=_u32(data, offset + 0x18),
                )
            )

    # The compact slot table gives the strongest texture signal. A loose scan
    # still catches normal/encoded/detail maps that are stored later in the MTA.
    loose_refs = []
    seen = set()
    for offset in range(0, max(0, len(data) - 3), 4):
        value = _u32(data, offset)
        if not _looks_like_ref(value) or value in seen:
            continue
        seen.add(value)
        loose_refs.append(value)

    slot_refs = tuple(slot.texture_ref for slot in slots)
    primary_refs = slot_refs[:1]
    extra_refs = _unique_refs(slot_refs[1:], (value for value in loose_refs if value not in set(slot_refs)))
    refs = _unique_refs(primary_refs, extra_refs)
    return MtaDescriptor(
        name=name,
        shader_name=shader_name,
        layer_count=layer_count,
        texture_slots=tuple(slots),
        texture_refs=refs,
        primary_texture_refs=primary_refs,
        extra_texture_refs=extra_refs,
    )


def parse_mta_file(path: str) -> MtaDescriptor:
    with open(path, "rb") as handle:
        return parse_mta(handle.read(), os.path.basename(path))


def parse_tex(data: bytes, name: str = "") -> TexDescriptor:
    if len(data) < 0x18 or data[0x14:0x18] != TEX_MAGIC:
        raise ValueError("missing TEX_ magic")

    key = None
    width = _u16(data, 0x4E) if len(data) >= 0x52 else None
    height = _u16(data, 0x50) if len(data) >= 0x52 else None
    if width == 0:
        width = None
    if height == 0:
        height = None
    format_code = data[0x2A] if len(data) > 0x2A else None
    format_name = TEX_FORMAT_NAMES.get(format_code, f"0x{format_code:02X}" if format_code is not None else "")
    if len(data) >= 16 and _u32(data, len(data) - 8) == TEX_TAIL_MARKER:
        first = _u32(data, len(data) - 16)
        second = _u32(data, len(data) - 12)
        if first == second and first not in {0, 0xFFFFFFFF}:
            key = first

    if key is None:
        start = max(0, len(data) - 0x80)
        for offset in range(start, max(start, len(data) - 11)):
            first = _u32(data, offset)
            if (
                first not in {0, 0xFFFFFFFF}
                and first == _u32(data, offset + 4)
                and _u32(data, offset + 8) == TEX_TAIL_MARKER
            ):
                key = first
                break

    return TexDescriptor(
        name=name,
        file_key=key,
        width=width,
        height=height,
        format_code=format_code,
        format_name=format_name,
    )


def parse_tex_file(path: str) -> TexDescriptor:
    with open(path, "rb") as handle:
        return parse_tex(handle.read(), os.path.basename(path))
