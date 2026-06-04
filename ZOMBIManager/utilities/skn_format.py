from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import List, Sequence, Tuple

SKN_MAGIC = b"SKN_"
SKN_MAGIC_OFFSET = 0x14
SKN_VERSION_OFFSET = 0x18
SKN_BONE_COUNT_OFFSET = 0x20
SKN_ROOT_MATRIX_OFFSET = 0x24
SKN_BONE_RECORD_OFFSET = 0x64


@dataclass
class SknBone:
    index: int
    parent_index: int
    name: str
    name_offset: int
    record_offset: int
    hash_value: int
    flags: Tuple[int, int, int]
    matrix: Tuple[float, ...] | None


@dataclass
class SknMaskEntry:
    bone_index: int
    weight: float


@dataclass
class SknBoneMask:
    name: str
    entries: List[SknMaskEntry]


@dataclass
class SknPoseTransform:
    index: int
    offset: int
    rotation: Tuple[float, float, float, float]
    translation: Tuple[float, float, float]
    extra: Tuple[float, ...]


@dataclass
class SknPoseBlock:
    offset: int
    size: int
    tag: str
    transforms: List[SknPoseTransform]


@dataclass
class SknFile:
    path: str
    stored_size: int
    version: int
    root_matrix: Tuple[float, ...]
    bones: List[SknBone]
    masks: List[SknBoneMask]
    pose_blocks: List[SknPoseBlock]

    @property
    def bone_count(self) -> int:
        return len(self.bones)


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _read_matrix(data: bytes, offset: int) -> Tuple[float, ...]:
    return struct.unpack_from("<16f", data, offset)


def _is_printable_ascii(raw: bytes) -> bool:
    return all(32 <= byte < 127 for byte in raw)


def _record_header_at(data: bytes, offset: int) -> tuple[int, int, int, int, int, int, int, bytes] | None:
    if offset < 0 or offset + 24 > len(data):
        return None

    parent_index = _i32(data, offset)
    flag_a, flag_b, flag_c = struct.unpack_from("<HHH", data, offset + 4)
    hash_value = _u32(data, offset + 10)
    bone_index = struct.unpack_from("<H", data, offset + 14)[0]
    name_len = _u32(data, offset + 16)
    name_size = _u32(data, offset + 20)
    name_offset = offset + 24

    if not (-1 <= parent_index < 512):
        return None
    if flag_a != 1 or flag_b != 0x00FF or flag_c != 0:
        return None
    if bone_index >= 512:
        return None
    if name_len <= 0 or name_size != name_len + 1 or name_size > 96:
        return None
    if name_offset + name_size > len(data):
        return None

    raw_name = data[name_offset:name_offset + name_size]
    if not raw_name.endswith(b"\0"):
        return None
    if not _is_printable_ascii(raw_name[:-1]):
        return None

    return parent_index, flag_a, flag_b, flag_c, hash_value, bone_index, name_size, raw_name


def _find_next_record(data: bytes, start_offset: int, wanted_index: int) -> int | None:
    # all known files have a 64-byte matrix between records, but scan a little
    # wider so malformed or slightly different files still have a chance.
    for candidate in range(start_offset, min(start_offset + 160, len(data) - 24)):
        header = _record_header_at(data, candidate)
        if header and header[5] == wanted_index:
            return candidate
    return None


def _parse_bones(data: bytes, bone_count: int) -> tuple[List[SknBone], int]:
    bones: List[SknBone] = []
    offset = SKN_BONE_RECORD_OFFSET

    for ordinal in range(bone_count):
        header = _record_header_at(data, offset)
        if header is None:
            raise ValueError(f"Invalid SKN bone record at 0x{offset:x}.")

        parent_index, flag_a, flag_b, flag_c, hash_value, bone_index, name_size, raw_name = header
        name_offset = offset + 24
        name = raw_name[:-1].decode("ascii", "replace")
        name_end = name_offset + name_size

        matrix = None
        next_offset = None
        if ordinal + 1 < bone_count:
            next_offset = _find_next_record(data, name_end, ordinal + 1)
            if next_offset is None:
                raise ValueError(f"Could not find SKN bone record {ordinal + 1}.")
            if next_offset - name_end >= 64:
                matrix = _read_matrix(data, name_end)
        else:
            next_offset = name_end

        bones.append(
            SknBone(
                index=bone_index,
                parent_index=parent_index,
                name=name,
                name_offset=name_offset,
                record_offset=offset,
                hash_value=hash_value,
                flags=(flag_a, flag_b, flag_c),
                matrix=matrix,
            )
        )
        offset = next_offset

    bones.sort(key=lambda bone: bone.index)
    return bones, offset


def _parse_masks(data: bytes, offset: int, bone_count: int) -> tuple[List[SknBoneMask], int]:
    if offset + 8 > len(data) or _u32(data, offset) != 0:
        return [], offset

    cursor = offset + 4
    mask_count = _u32(data, cursor)
    cursor += 4
    if mask_count > 64:
        return [], offset

    masks: List[SknBoneMask] = []
    for _ in range(mask_count):
        if cursor + 4 > len(data):
            return masks, cursor
        name_size = _u32(data, cursor)
        cursor += 4
        if name_size <= 0 or name_size > 128 or cursor + name_size > len(data):
            return masks, cursor

        raw_name = data[cursor:cursor + name_size]
        cursor += name_size
        name = raw_name.split(b"\0", 1)[0].decode("ascii", "replace")

        if cursor + 4 > len(data):
            return masks, cursor
        entry_count = _u32(data, cursor)
        cursor += 4
        if entry_count > max(bone_count * 2, 256):
            return masks, cursor

        entries: List[SknMaskEntry] = []
        for _entry_index in range(entry_count):
            if cursor + 8 > len(data):
                return masks, cursor
            bone_index = _u32(data, cursor)
            weight = struct.unpack_from("<f", data, cursor + 4)[0]
            cursor += 8
            entries.append(SknMaskEntry(bone_index=bone_index, weight=weight))

        masks.append(SknBoneMask(name=name, entries=entries))

    return masks, cursor


def _transform_score(data: bytes, offset: int, count: int) -> int:
    score = 0
    for index in range(min(count, 24)):
        record_offset = offset + index * 48
        if record_offset + 48 > len(data):
            break
        values = struct.unpack_from("<12f", data, record_offset)
        quat_len = sum(value * value for value in values[:4]) ** 0.5
        extra_error = sum(abs(value - 1.0) for value in values[7:12])
        if abs(quat_len - 1.0) < 0.02 and extra_error < 0.2 and all(abs(value) < 10.0 for value in values[4:7]):
            score += 1
    return score


def _find_pose_transform_offset(data: bytes, block_offset: int, block_size: int, bone_count: int) -> int | None:
    block_end = block_offset + block_size
    if block_offset + 0x20 <= len(data):
        candidate = block_offset + _u32(data, block_offset + 0x1C) + 0x1C
        if candidate + bone_count * 48 <= block_end and _transform_score(data, candidate, bone_count) >= min(12, bone_count):
            return candidate

    best_score = 0
    best_offset = None
    for candidate in range(block_offset + 0x180, min(block_offset + 0x240, block_end - 48), 4):
        if candidate + bone_count * 48 > block_end:
            continue
        score = _transform_score(data, candidate, bone_count)
        if score > best_score:
            best_score = score
            best_offset = candidate

    if best_score >= min(12, bone_count):
        return best_offset
    return None


def _parse_pose_transforms(data: bytes, block_offset: int, block_size: int, bone_count: int) -> List[SknPoseTransform]:
    transform_offset = _find_pose_transform_offset(data, block_offset, block_size, bone_count)
    if transform_offset is None:
        return []

    transforms: List[SknPoseTransform] = []
    block_end = block_offset + block_size
    max_count = min(bone_count, (block_end - transform_offset) // 48)
    for index in range(max_count):
        record_offset = transform_offset + index * 48
        values = struct.unpack_from("<12f", data, record_offset)
        transforms.append(
            SknPoseTransform(
                index=index,
                offset=record_offset,
                rotation=values[:4],
                translation=values[4:7],
                extra=values[7:],
            )
        )
    return transforms


def _parse_pose_blocks(data: bytes, offset: int, bone_count: int) -> List[SknPoseBlock]:
    blocks: List[SknPoseBlock] = []
    cursor = offset
    while cursor + 8 <= len(data):
        size = _u32(data, cursor)
        tag_raw = data[cursor + 4:cursor + 8]
        if size <= 0 or cursor + size > len(data):
            break
        if not _is_printable_ascii(tag_raw):
            break

        tag = tag_raw.decode("ascii", "replace")
        transforms = _parse_pose_transforms(data, cursor, size, bone_count)
        blocks.append(SknPoseBlock(offset=cursor, size=size, tag=tag, transforms=transforms))
        cursor += size

    return blocks


def parse_skn(data: bytes, path: str = "") -> SknFile:
    if len(data) < SKN_BONE_RECORD_OFFSET:
        raise ValueError("File is too small to be a supported SKN.")
    if data[SKN_MAGIC_OFFSET:SKN_MAGIC_OFFSET + 4] != SKN_MAGIC:
        raise ValueError("Missing SKN_ marker.")

    stored_size = _u32(data, 0)
    version = _u32(data, SKN_VERSION_OFFSET)
    bone_count = _u32(data, SKN_BONE_COUNT_OFFSET)
    if bone_count <= 0 or bone_count > 512:
        raise ValueError(f"Implausible SKN bone count: {bone_count}")

    root_matrix = _read_matrix(data, SKN_ROOT_MATRIX_OFFSET)
    bones, offset = _parse_bones(data, bone_count)
    masks, offset = _parse_masks(data, offset, bone_count)
    pose_blocks = _parse_pose_blocks(data, offset, bone_count)

    return SknFile(
        path=path,
        stored_size=stored_size,
        version=version,
        root_matrix=root_matrix,
        bones=bones,
        masks=masks,
        pose_blocks=pose_blocks,
    )


def parse_skn_file(filepath: str) -> SknFile:
    with open(filepath, "rb") as handle:
        return parse_skn(handle.read(), os.path.abspath(filepath))


def bone_hash_sequence(skn: SknFile) -> List[int]:
    return [bone.hash_value for bone in sorted(skn.bones, key=lambda bone: bone.index)]


def mask_summary(mask: SknBoneMask) -> str:
    return ",".join(f"{entry.bone_index}:{entry.weight:.4g}" for entry in mask.entries)


def matrix_translation(matrix: Sequence[float] | None) -> Tuple[float, float, float]:
    if not matrix:
        return (0.0, 0.0, 0.0)
    return (matrix[12], matrix[13], matrix[14])
