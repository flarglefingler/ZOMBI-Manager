from __future__ import annotations

import os
import struct
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

TRL_MAGIC = b"TRL_"
TRL_MAGIC_OFFSET = 0x14
TRL_VERSION_OFFSET = 0x18
TRL_BLOCK_TAG = b"50AE"

CHANNEL_GROUP_SLOTS = (
    ("static_rotation", 0x16),
    ("static_translation", 0x18),
    ("animated_rotation", 0x1E),
    ("animated_translation", 0x20),
)

SECTION_FIELDS = (
    ("track_table", 0x38),
    ("packed_keys", 0x3C),
    ("static_rotation", 0x40),
    ("static_translation", 0x44),
    ("small_block_b", 0x48),
    ("small_block_a", 0x4C),
)

DEFAULT_ROTATION_VARIANT = "p48_be_bottom"
DEFAULT_STATIC_ROTATION_VARIANT = "legacy_be_xyz"
ROTATION_MODE_ABSOLUTE = "absolute"
ROTATION_MODE_REST_DELTA = "rest_delta"

LEGACY_ROTATION_VARIANTS: dict[str, tuple[tuple[int, int, int], int]] = {
    "legacy_be_xyz": ((0, 1, 2), 1),
    "legacy_be_xzy": ((0, 2, 1), 1),
    "legacy_be_yxz": ((1, 0, 2), 1),
    "legacy_be_yzx": ((1, 2, 0), 1),
    "legacy_be_zxy": ((2, 0, 1), 1),
    "legacy_be_zyx": ((2, 1, 0), 1),
    "legacy_be_xyz_negw": ((0, 1, 2), -1),
    "legacy_be_xzy_negw": ((0, 2, 1), -1),
    "legacy_be_yxz_negw": ((1, 0, 2), -1),
    "legacy_be_yzx_negw": ((1, 2, 0), -1),
    "legacy_be_zxy_negw": ((2, 0, 1), -1),
    "legacy_be_zyx_negw": ((2, 1, 0), -1),
}

PACKED48_ROTATION_VARIANTS: dict[str, tuple[str, str, tuple[int, int, int, int], tuple[int, int, int], int]] = {
    # 48-bit smallest-three quaternion: 2-bit dropped component index plus
    # three unsigned 15-bit components mapped to [-sqrt(.5), sqrt(.5)].
    "p48_be_bottom": ("big", "bottom", (0, 1, 2, 3), (0, 1, 2), 1),
    "p48_be_bottom_xzy": ("big", "bottom", (0, 1, 2, 3), (0, 2, 1), 1),
    "p48_be_bottom_yxz": ("big", "bottom", (0, 1, 2, 3), (1, 0, 2), 1),
    "p48_be_bottom_neg": ("big", "bottom", (0, 1, 2, 3), (0, 1, 2), -1),
    "p48_be_bottom_w2": ("big", "bottom", (0, 1, 3, 2), (0, 1, 2), 1),
    "p48_be_bottom_w2_xzy": ("big", "bottom", (0, 1, 3, 2), (0, 2, 1), 1),
    "p48_be_top": ("big", "top", (0, 1, 2, 3), (0, 1, 2), 1),
    "p48_le_bottom": ("little", "bottom", (0, 1, 2, 3), (0, 1, 2), 1),
}

ROTATION_VARIANTS: dict[str, str] = {
    **{name: name for name in PACKED48_ROTATION_VARIANTS},
    **{name: name for name in LEGACY_ROTATION_VARIANTS},
}


@dataclass
class TrlChannelGroup:
    index: int
    kind: str
    field_offset: int
    length: int
    bone_indices: List[int]


@dataclass
class TrlSection:
    name: str
    offset: int
    end_offset: int
    field_offset: int | None = None

    @property
    def length(self) -> int:
        return max(0, self.end_offset - self.offset)


@dataclass
class TrlFile:
    path: str
    stored_size: int
    file_size: int
    version: int
    block_offset: int
    duration: float
    fps: float
    section_size: int
    bone_count: int
    frame_count: int
    unknown_count: int
    flags: int
    unknown_value: int
    channel_groups: List[TrlChannelGroup]
    sections: List[TrlSection]
    section_offsets: List[int]
    bone_hashes: List[int]
    bone_hash_offset: int | None
    payload_offset: int | None

    @property
    def frame_start(self) -> int:
        return 1

    @property
    def frame_end(self) -> int:
        return max(1, self.frame_count)

    @property
    def payload_size(self) -> int:
        if self.payload_offset is None:
            return 0
        return max(0, self.file_size - self.payload_offset)


@dataclass
class TrlBasePose:
    rotations: Dict[int, Tuple[float, float, float, float]]
    translations: Dict[int, Tuple[float, float, float]]
    notes: List[str]
    rotation_modes: Dict[int, str] = field(default_factory=dict)


@dataclass
class TrlFrameWindow:
    start_frame: int
    frame_count: int


@dataclass
class TrlKeySample:
    frame: int
    rotations: Dict[int, Tuple[float, float, float, float]]
    translations: Dict[int, Tuple[float, float, float]]
    source_offset: int
    translation_offset: int
    score: float
    dense_key: bool = False
    rotation_updates: Dict[int, Tuple[float, float, float, float]] | None = None
    translation_updates: Dict[int, Tuple[float, float, float]] | None = None


@dataclass
class TrlSampledAnimation:
    samples: List[TrlKeySample]
    frame_windows: List[TrlFrameWindow]
    notes: List[str]
    dense_frame_keys: bool = False


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _read_hashes(data: bytes, offset: int, count: int) -> List[int]:
    return [_u32(data, offset + index * 4) for index in range(count)]


def _normalize_quat(value: Sequence[float]) -> Tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 0.000001:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / length for component in value)  # type: ignore[return-value]


def _decode_rotation48(
    data: bytes,
    offset: int,
    variant: str = DEFAULT_ROTATION_VARIANT,
) -> Tuple[float, float, float, float]:
    if variant in PACKED48_ROTATION_VARIANTS:
        byte_order, index_position, missing_order, component_order, missing_sign = PACKED48_ROTATION_VARIANTS[variant]
        packed = int.from_bytes(data[offset:offset + 6], byte_order)
        if index_position == "top":
            missing_code = (packed >> 46) & 0x03
            component_bits = packed & ((1 << 45) - 1)
        else:
            missing_code = packed & 0x03
            component_bits = packed >> 2

        raw_components = (
            (component_bits >> 30) & 0x7FFF,
            (component_bits >> 15) & 0x7FFF,
            component_bits & 0x7FFF,
        )
        scale = math.sqrt(0.5)
        decoded = [
            ((raw_components[index] / 32767.0) * 2.0 - 1.0) * scale
            for index in component_order
        ]

        missing_component = missing_order[missing_code]
        components = [0.0, 0.0, 0.0, 0.0]
        stored_components = [index for index in range(4) if index != missing_component]
        for component_index, value in zip(stored_components, decoded):
            components[component_index] = value

        missing_squared = max(0.0, 1.0 - sum(value * value for value in components))
        components[missing_component] = math.sqrt(missing_squared) * missing_sign
        return _normalize_quat(components)

    # Kept for comparison only. This was the earlier "three shorts" read; it
    # gives recognizable timing, but it is not the real packed quaternion form.
    raw = struct.unpack_from(">hhh", data, offset)
    order, w_sign = LEGACY_ROTATION_VARIANTS.get(variant, LEGACY_ROTATION_VARIANTS["legacy_be_xyz_negw"])
    scale = math.sqrt(0.5) / 65535.0
    x = raw[order[0]] * scale
    y = raw[order[1]] * scale
    z = raw[order[2]] * scale
    w_squared = max(0.0, 1.0 - x * x - y * y - z * z)
    return _normalize_quat((x, y, z, math.sqrt(w_squared) * w_sign))


def _group_by_kind(trl: TrlFile, kind: str) -> TrlChannelGroup | None:
    for group in trl.channel_groups:
        if group.kind == kind:
            return group
    return None


def _section_by_name(trl: TrlFile, name: str) -> TrlSection | None:
    for section in trl.sections:
        if section.name == name:
            return section
    return None


def _packed_section(trl: TrlFile) -> TrlSection | None:
    return _section_by_name(trl, "packed_keys")


def _channel_group_lengths(data: bytes, block_offset: int) -> List[tuple[int, str, int, int]]:
    lengths: List[tuple[int, str, int, int]] = []
    for index, (kind, offset) in enumerate(CHANNEL_GROUP_SLOTS):
        length = _u16(data, block_offset + offset)
        if length:
            lengths.append((index, kind, offset, length))
    return lengths


def _read_channel_groups(data: bytes, block_offset: int, bone_count: int) -> List[TrlChannelGroup]:
    lengths = _channel_group_lengths(data, block_offset)
    groups: List[TrlChannelGroup] = []
    cursor = block_offset + 0x70

    for index, kind, field_offset, length in lengths:
        bone_indices: List[int] = []
        while len(bone_indices) < length and cursor + 2 <= len(data):
            value = _u16(data, cursor)
            cursor += 2
            if value < bone_count:
                bone_indices.append(value)

        # groups are padded with one or more bone_count sentinels.
        while cursor + 2 <= len(data) and _u16(data, cursor) >= bone_count:
            cursor += 2

        groups.append(
            TrlChannelGroup(
                index=index,
                kind=kind,
                field_offset=field_offset,
                length=length,
                bone_indices=bone_indices,
            )
        )

    return groups


def _section_field_offsets(data: bytes, block_offset: int) -> List[tuple[str, int, int]]:
    base = block_offset + 0x60
    offsets: List[tuple[str, int, int]] = []
    for name, field_offset in SECTION_FIELDS:
        relative = _u32(data, block_offset + field_offset)
        if relative:
            offset = base + relative
            if 0 <= offset <= len(data):
                offsets.append((name, field_offset, offset))
    return offsets


def _read_section_offsets(data: bytes, block_offset: int) -> List[int]:
    return sorted({offset for _name, _field_offset, offset in _section_field_offsets(data, block_offset)})


def _read_sections(data: bytes, block_offset: int, payload_offset: int | None) -> List[TrlSection]:
    named_offsets = _section_field_offsets(data, block_offset)
    by_offset: dict[int, tuple[str, int]] = {}
    for name, field_offset, offset in named_offsets:
        by_offset.setdefault(offset, (name, field_offset))

    sorted_offsets = sorted(by_offset)
    boundaries: List[int] = []
    if payload_offset is not None:
        boundaries.append(payload_offset)
    boundaries.extend(sorted_offsets)
    boundaries.append(len(data))
    boundaries = sorted(set(offset for offset in boundaries if 0 <= offset <= len(data)))

    sections: List[TrlSection] = []
    for offset, end_offset in zip(boundaries, boundaries[1:]):
        if offset >= end_offset:
            continue
        name, field_offset = by_offset.get(offset, ("prelude", None))
        sections.append(TrlSection(name=name, offset=offset, end_offset=end_offset, field_offset=field_offset))

    return sections


def _hash_table_candidate(data: bytes, block_offset: int, bone_count: int) -> int | None:
    base = block_offset + 0x60
    if base + 4 > len(data):
        return None
    offset = base + _u32(data, base)
    if offset < 0 or offset + bone_count * 4 > len(data):
        return None
    return offset


def find_bone_hash_table(
    data: bytes,
    block_offset: int,
    bone_count: int,
    target_hashes: Sequence[int],
) -> tuple[int | None, List[int]]:
    if bone_count <= 0 or not target_hashes:
        return None, []

    target_order = {hash_value: index for index, hash_value in enumerate(target_hashes)}
    target_set = set(target_order)

    direct_offset = _hash_table_candidate(data, block_offset, bone_count)
    if direct_offset is not None:
        hashes = _read_hashes(data, direct_offset, bone_count)
        if all(value in target_set for value in hashes):
            return direct_offset, hashes

    search_start = block_offset + 0x20
    search_end = len(data) - bone_count * 4
    best: tuple[int, int, int, int, int] | None = None
    best_hashes: List[int] = []

    for offset in range(search_start, max(search_start, search_end) + 1, 2):
        hashes = _read_hashes(data, offset, bone_count)
        in_target = [value for value in hashes if value in target_set]
        if len(in_target) < bone_count:
            continue

        indices = [target_order[value] for value in hashes]
        increasing = sum(1 for left, right in zip(indices, indices[1:]) if right > left)
        unique_count = len(set(hashes))
        # lower offset wins after quality, because every observed file has one real table.
        score = (len(in_target), increasing, unique_count, -offset, offset)
        if best is None or score > best:
            best = score
            best_hashes = hashes

    if best is None:
        return None, []
    return best[4], best_hashes


def parse_trl(
    data: bytes,
    path: str = "",
    target_hashes: Sequence[int] | None = None,
) -> TrlFile:
    if len(data) < 0x30:
        raise ValueError("File is too small to be a supported TRL.")
    if data[TRL_MAGIC_OFFSET:TRL_MAGIC_OFFSET + 4] != TRL_MAGIC:
        raise ValueError("Missing TRL_ marker.")

    block_offset = data.find(TRL_BLOCK_TAG)
    if block_offset < 0:
        raise ValueError("Missing 50AE animation block.")
    if block_offset + 0x18 > len(data):
        raise ValueError("Incomplete TRL animation block.")

    bone_count = _u16(data, block_offset + 0x0E)
    unknown_count = _u16(data, block_offset + 0x12)
    bone_hash_offset = None
    bone_hashes: List[int] = []
    if target_hashes:
        bone_hash_offset, bone_hashes = find_bone_hash_table(data, block_offset, bone_count, target_hashes)
    payload_offset = (bone_hash_offset + bone_count * 4) if bone_hash_offset is not None else None

    return TrlFile(
        path=path,
        stored_size=_u32(data, 0),
        file_size=len(data),
        version=_u32(data, TRL_VERSION_OFFSET),
        block_offset=block_offset,
        duration=_f32(data, block_offset + 0x04),
        fps=_f32(data, block_offset + 0x08),
        section_size=_u16(data, block_offset + 0x0C),
        bone_count=bone_count,
        frame_count=_u16(data, block_offset + 0x10),
        unknown_count=unknown_count,
        flags=_u16(data, block_offset + 0x14),
        unknown_value=_u16(data, block_offset + 0x16),
        channel_groups=_read_channel_groups(data, block_offset, bone_count),
        sections=_read_sections(data, block_offset, payload_offset),
        section_offsets=_read_section_offsets(data, block_offset),
        bone_hashes=bone_hashes,
        bone_hash_offset=bone_hash_offset,
        payload_offset=payload_offset,
    )


def parse_trl_file(filepath: str, target_hashes: Sequence[int] | None = None) -> TrlFile:
    with open(filepath, "rb") as handle:
        return parse_trl(handle.read(), os.path.abspath(filepath), target_hashes)


def _decode_rotation_records(
    data: bytes,
    offset: int,
    bone_indices: Sequence[int],
    available_length: int,
    rotation_variant: str = DEFAULT_ROTATION_VARIANT,
    variants_by_bone: Mapping[int, str] | None = None,
) -> Dict[int, Tuple[float, float, float, float]]:
    rotations: Dict[int, Tuple[float, float, float, float]] = {}
    count = min(len(bone_indices), max(0, available_length // 6))
    for record_index in range(count):
        bone_index = bone_indices[record_index]
        rotations[bone_index] = _decode_rotation48(
            data,
            offset + record_index * 6,
            variants_by_bone.get(bone_index, rotation_variant) if variants_by_bone else rotation_variant,
        )
    return rotations


def _decode_translation_records(
    data: bytes,
    offset: int,
    bone_indices: Sequence[int],
    available_length: int,
) -> Dict[int, Tuple[float, float, float]]:
    translations: Dict[int, Tuple[float, float, float]] = {}
    count = min(len(bone_indices), max(0, available_length // 12))
    for record_index in range(count):
        translations[bone_indices[record_index]] = struct.unpack_from("<fff", data, offset + record_index * 12)
    return translations


def _bit_value(data: bytes, bit_index: int) -> int:
    if bit_index < 0 or bit_index >= len(data) * 8:
        return 0
    # the surrounding TRL bit fields use high bits first.
    return (data[bit_index // 8] >> (7 - (bit_index % 8))) & 1


def _mask_count(data: bytes, start_bit: int, bit_count: int) -> int:
    return sum(_bit_value(data, start_bit + index) for index in range(max(0, bit_count)))


def _static_rotation_start(section: TrlSection, count: int) -> tuple[int, int]:
    expected = count * 6
    if section.length >= expected and section.length - expected <= 8:
        padding = section.length - expected
        return section.offset + padding, expected
    return section.offset, section.length


def _static_translation_start(section: TrlSection, count: int) -> tuple[int, int]:
    expected = count * 12
    if section.length >= expected:
        return section.offset, expected
    return section.offset, section.length


def _small_section_bytes(data: bytes, trl: TrlFile) -> List[tuple[TrlSection, bytes]]:
    sections = []
    for section in trl.sections:
        if section.name == "packed_keys" or section.length > 64:
            continue
        sections.append((section, data[section.offset:section.end_offset]))
    return sections


def _raw_base_sizes(data: bytes, trl: TrlFile) -> tuple[int, int] | None:
    section = _section_by_name(trl, "small_block_b")
    if section is None or section.length < 12:
        return None
    values = [
        struct.unpack_from("<H", data, section.offset + index * 2)[0]
        for index in range(section.length // 2)
    ]
    if len(values) < 6:
        return None
    rotation_size = values[4]
    translation_size = values[5]
    if rotation_size == 0 and translation_size == 0:
        return None
    return rotation_size, translation_size


def _find_packed_descriptor(
    data: bytes,
    trl: TrlFile,
    accepted_lengths: Sequence[int],
    preferred_section: str | None = None,
    strict_section: str | None = None,
) -> tuple[int, int, int] | None:
    packed = _packed_section(trl)
    if packed is None:
        return None

    candidates: List[tuple[int, int, int, int]] = []
    accepted = set(accepted_lengths)
    for section, raw in _small_section_bytes(data, trl):
        if strict_section is not None and section.name != strict_section:
            continue
        for byte_offset in range(0, max(0, len(raw) - 3), 2):
            length, offset = struct.unpack_from("<HH", raw, byte_offset)
            if length not in accepted:
                continue
            if offset + length > packed.length:
                continue
            preferred = 0 if preferred_section is None or section.name == preferred_section else 1
            candidates.append((preferred, section.offset + byte_offset, length, offset))

    if not candidates:
        return None

    _preferred, source_offset, length, offset = sorted(candidates)[0]
    return source_offset, length, offset


def _translation_table_is_plausible(data: bytes, offset: int, count: int) -> bool:
    if count <= 0:
        return False
    try:
        values = struct.unpack_from("<" + "f" * count * 3, data, offset)
    except struct.error:
        return False
    return all(math.isfinite(value) and -20.0 <= value <= 20.0 for value in values)


def _decode_frame_windows(data: bytes, trl: TrlFile) -> List[TrlFrameWindow]:
    """Read the coarse frame windows used by sampled/compressed TRL files.

    Moving tracks store most window pairs after the packed stream descriptors,
    but the last few pairs can sit at the start of packed_keys. Keeping this in
    one place makes the later key-table scan much less guessy.
    """

    def direct_windows(raw: bytes) -> List[TrlFrameWindow]:
        best: List[TrlFrameWindow] = []
        for start_offset in range(0, max(0, len(raw) - 3), 2):
            found: List[TrlFrameWindow] = []
            last_start = -1
            for offset in range(start_offset, max(start_offset, len(raw) - 3), 4):
                start_frame, frame_count = struct.unpack_from("<HH", raw, offset)
                if start_frame >= trl.frame_count or frame_count > trl.frame_count:
                    break
                if start_frame <= last_start:
                    break
                if start_frame + frame_count > trl.frame_count:
                    break
                if frame_count == 0 and start_frame < trl.frame_count - 1:
                    break
                found.append(TrlFrameWindow(start_frame=start_frame, frame_count=frame_count))
                last_start = start_frame
                if len(found) >= trl.unknown_count:
                    break
                if frame_count == 0 and start_frame >= trl.frame_count - 1:
                    break

            if len(found) > len(best):
                best = found

        return best[:trl.unknown_count]

    windows: List[TrlFrameWindow] = []
    track = _section_by_name(trl, "track_table")
    if track is not None:
        raw = data[track.offset:track.end_offset]
        tail_offset = None
        for offset in range(0, max(0, len(raw) - 7), 8):
            _unused, bit_length, _bit_offset = struct.unpack_from("<HHI", raw, offset)
            if offset > 0 and bit_length == 0:
                tail_offset = offset + 8
                break

        if tail_offset is not None:
            last_start = -1
            for offset in range(tail_offset, max(tail_offset, len(raw) - 3), 4):
                start_frame, frame_count = struct.unpack_from("<HH", raw, offset)
                if start_frame >= trl.frame_count or frame_count > trl.frame_count:
                    break
                if start_frame <= last_start:
                    break
                if frame_count == 0 and start_frame < trl.frame_count - 1:
                    break
                windows.append(TrlFrameWindow(start_frame=start_frame, frame_count=frame_count))
                last_start = start_frame

        direct = direct_windows(raw)
        if len(direct) > len(windows):
            windows = direct

    packed = _packed_section(trl)
    if packed is not None and len(windows) < trl.unknown_count:
        last_start = windows[-1].start_frame if windows else -1
        cursor = packed.offset
        while cursor + 4 <= packed.end_offset and len(windows) < trl.unknown_count:
            start_frame, frame_count = struct.unpack_from("<HH", data, cursor)
            if start_frame >= trl.frame_count or frame_count > trl.frame_count:
                break
            if start_frame <= last_start:
                break
            if frame_count == 0 and start_frame < trl.frame_count - 1:
                break
            windows.append(TrlFrameWindow(start_frame=start_frame, frame_count=frame_count))
            last_start = start_frame
            cursor += 4
            if frame_count == 0 and start_frame >= trl.frame_count - 1:
                break

    return windows[:trl.unknown_count]


def _frame_window_score(windows: Sequence[TrlFrameWindow], frame_count: int) -> tuple[int, int, int, int]:
    if not windows:
        return (0, 0, 0, 0)

    monotonic = 1
    previous = -1
    covered = 0
    zero_middle = 0
    for window in windows:
        if window.start_frame <= previous:
            monotonic = 0
        previous = window.start_frame
        if window.frame_count == 0 and window.start_frame < frame_count - 1:
            zero_middle += 1
        covered = max(covered, min(frame_count, window.start_frame + window.frame_count + 1))

    starts_at_zero = 1 if windows[0].start_frame == 0 else 0
    ends_at_last = 1 if windows[-1].start_frame >= max(0, frame_count - 1) else 0
    return (monotonic, starts_at_zero + ends_at_last, covered, -zero_middle)


def _find_header_sample_table_candidates(
    data: bytes,
    trl: TrlFile,
    rest_translations: Dict[int, Tuple[float, float, float]] | None,
) -> List[tuple[float, int, int]]:
    packed = _packed_section(trl)
    animated_rotation_group = _group_by_kind(trl, "animated_rotation")
    animated_translation_group = _group_by_kind(trl, "animated_translation")
    if packed is None or animated_rotation_group is None or animated_translation_group is None:
        return []

    rotation_length = animated_rotation_group.length * 6
    translation_length = animated_translation_group.length * 12
    if rotation_length <= 0 or translation_length <= 0:
        return []

    candidates: List[tuple[float, int, int]] = []
    float_count = animated_translation_group.length * 3
    scan_start = max(0, packed.offset - 128)
    scan_end = max(scan_start, packed.end_offset - 15)
    for header_offset in range(scan_start, scan_end, 2):
        (
            rotation_table_size,
            translation_table_size,
            mask_prefix_size,
            _unk_b,
            rotation_block_size,
            translation_block_size,
            _tail_a,
            _tail_b,
        ) = struct.unpack_from("<HHHHHHHH", data, header_offset)

        if rotation_table_size != rotation_length or translation_table_size != translation_length:
            continue
        if mask_prefix_size > 8:
            continue
        if rotation_block_size % 6 != 0 or translation_block_size % 12 != 0:
            continue

        sample_relative = header_offset + 16 - packed.offset
        translation_relative = sample_relative + rotation_length
        sample_offset = packed.offset + sample_relative
        translation_offset = packed.offset + translation_relative
        if sample_offset < 0 or translation_offset < 0:
            continue
        if translation_offset + translation_length > len(data):
            continue

        try:
            values = struct.unpack_from("<" + "f" * float_count, data, translation_offset)
        except struct.error:
            continue
        if not all(math.isfinite(value) and -20.0 < value < 20.0 for value in values):
            continue

        score, _exact_score, _near_score, _small_vectors = _translation_table_score(
            values,
            animated_translation_group.bone_indices,
            rest_translations,
        )
        candidates.append((score + 100.0, sample_relative, translation_relative))

    return sorted(candidates, key=lambda item: item[1])


def _translation_table_score(
    values: Sequence[float],
    bone_indices: Sequence[int],
    rest_translations: Dict[int, Tuple[float, float, float]] | None,
) -> tuple[float, int, int, int]:
    small_vectors = 0
    for index in range(len(bone_indices)):
        x, y, z = values[index * 3:index * 3 + 3]
        if math.sqrt(x * x + y * y + z * z) < 1.0:
            small_vectors += 1

    exact_score = 0
    near_score = 0
    if rest_translations:
        for index, bone_index in enumerate(bone_indices):
            rest = rest_translations.get(bone_index)
            if rest is None:
                continue
            x, y, z = values[index * 3:index * 3 + 3]
            error = math.sqrt((x - rest[0]) ** 2 + (y - rest[1]) ** 2 + (z - rest[2]) ** 2)
            if error < 0.0001:
                exact_score += 2
            elif error < 0.02:
                exact_score += 1
            if error < 0.08:
                near_score += 1

    return exact_score + near_score + small_vectors * 0.1, exact_score, near_score, small_vectors


def _find_sample_table_candidates(
    data: bytes,
    trl: TrlFile,
    rest_translations: Dict[int, Tuple[float, float, float]] | None,
) -> List[tuple[float, int, int]]:
    packed = _packed_section(trl)
    animated_rotation_group = _group_by_kind(trl, "animated_rotation")
    animated_translation_group = _group_by_kind(trl, "animated_translation")
    if packed is None or animated_rotation_group is None or animated_translation_group is None:
        return []

    rotation_length = animated_rotation_group.length * 6
    translation_length = animated_translation_group.length * 12
    if rotation_length <= 0 or translation_length <= 0:
        return []

    header_candidates = _find_header_sample_table_candidates(data, trl, rest_translations)
    expected_count = trl.unknown_count if trl.unknown_count else len(header_candidates)
    if header_candidates and len(header_candidates) >= expected_count:
        return header_candidates

    candidates: List[tuple[float, int, int]] = []
    float_count = animated_translation_group.length * 3
    first_translation = rotation_length
    last_translation = packed.length - translation_length
    if last_translation < first_translation:
        return []

    minimum_small = max(3, animated_translation_group.length // 3)
    minimum_near = max(4, animated_translation_group.length // 4)

    for translation_relative in range(first_translation, last_translation + 1, 4):
        translation_offset = packed.offset + translation_relative
        try:
            values = struct.unpack_from("<" + "f" * float_count, data, translation_offset)
        except struct.error:
            break
        if not all(math.isfinite(value) and -20.0 < value < 20.0 for value in values):
            continue

        score, exact_score, near_score, small_vectors = _translation_table_score(
            values,
            animated_translation_group.bone_indices,
            rest_translations,
        )
        if small_vectors < minimum_small:
            continue
        if rest_translations and exact_score < 4 and near_score < minimum_near:
            continue

        sample_relative = translation_relative - rotation_length
        candidates.append((score, sample_relative, translation_relative))

    # a true table produces many plausible 4-byte shifted neighbors. Keep the
    # best local candidate instead of letting those shifted copies crowd out
    # later samples.
    selected: List[tuple[float, int, int]] = []
    current_group: List[tuple[float, int, int]] = []
    for candidate in candidates:
        if current_group and candidate[2] - current_group[-1][2] > 128:
            selected.append(max(current_group, key=lambda item: item[0]))
            current_group = []
        current_group.append(candidate)
    if current_group:
        selected.append(max(current_group, key=lambda item: item[0]))

    if header_candidates and len(header_candidates) >= len(selected):
        return header_candidates

    return selected


def _mask_frame_count(mask_bytes: int, prefix_size: int, channel_count: int, frame_limit: int) -> int | None:
    if mask_bytes < 0 or channel_count <= 0:
        return None
    prefix_bits = prefix_size * 8
    matches = [
        frame_count
        for frame_count in range(frame_limit + 1)
        if (prefix_bits + frame_count * channel_count + 7) // 8 == mask_bytes
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[0]
    estimate = ((mask_bytes * 8) - prefix_bits) / channel_count
    rounded = int(round(estimate))
    if 0 <= rounded <= frame_limit:
        return rounded
    return None


def _derive_frame_windows_from_sample_headers(
    data: bytes,
    trl: TrlFile,
    candidates: Sequence[tuple[float, int, int]],
) -> List[TrlFrameWindow]:
    packed = _packed_section(trl)
    animated_rotation_group = _group_by_kind(trl, "animated_rotation")
    animated_translation_group = _group_by_kind(trl, "animated_translation")
    if packed is None or animated_rotation_group is None or animated_translation_group is None:
        return []
    if len(candidates) < 2:
        return []

    channel_count = animated_rotation_group.length + animated_translation_group.length
    ordered = sorted(candidates, key=lambda item: item[1])
    gap_counts: List[int] = []
    for index, (_score, sample_relative, _translation_relative) in enumerate(ordered[:-1]):
        source_offset = packed.offset + sample_relative
        header_offset = source_offset - 16
        next_header = packed.offset + ordered[index + 1][1] - 16
        if header_offset < 0 or header_offset + 16 > len(data):
            return []
        if next_header <= header_offset:
            return []

        (
            rotation_table_size,
            translation_table_size,
            mask_prefix_size,
            _unk_b,
            rotation_block_size,
            translation_block_size,
            tail_a,
            tail_b,
        ) = struct.unpack_from("<HHHHHHHH", data, header_offset)

        gap_start = source_offset + rotation_table_size + translation_table_size
        mask_and_padding = next_header - gap_start - rotation_block_size - translation_block_size
        mask_bytes = mask_and_padding - tail_a - tail_b
        frame_count = _mask_frame_count(mask_bytes, mask_prefix_size, channel_count, trl.frame_count)
        if frame_count is None:
            return []
        gap_counts.append(frame_count)

    windows: List[TrlFrameWindow] = []
    start_frame = 0
    for frame_count in gap_counts:
        if start_frame >= trl.frame_count or start_frame + frame_count > trl.frame_count:
            return []
        windows.append(TrlFrameWindow(start_frame=start_frame, frame_count=frame_count))
        start_frame += frame_count + 1

    if start_frame >= trl.frame_count:
        start_frame = max(0, trl.frame_count - 1)
    windows.append(TrlFrameWindow(start_frame=start_frame, frame_count=0))
    return windows[:trl.unknown_count]


def decode_trl_sampled_animation(
    data: bytes,
    trl: TrlFile,
    rest_translations: Dict[int, Tuple[float, float, float]] | None = None,
    rotation_variant: str = DEFAULT_ROTATION_VARIANT,
    rotation_variants_by_bone: Mapping[int, str] | None = None,
) -> TrlSampledAnimation:
    """Decode full sampled key tables from moving TRL files.

    This does not yet unpack every in-between bitstream delta. It decodes the
    full rotation/translation pose tables that mark each compressed frame
    window, which is the first real animated layer after the base pose.
    """

    notes: List[str] = []
    windows = _decode_frame_windows(data, trl)
    candidates = _find_sample_table_candidates(data, trl, rest_translations)
    if not windows:
        notes.append("no frame windows found")
    if not candidates:
        notes.append("no sampled pose tables found")

    if candidates:
        header_windows = _derive_frame_windows_from_sample_headers(data, trl, candidates)
        if len(header_windows) >= len(candidates) and (
            len(header_windows) > len(windows)
            or _frame_window_score(header_windows, trl.frame_count) > _frame_window_score(windows, trl.frame_count)
        ):
            windows = header_windows
            notes.append("frame windows derived from sample headers")

    if not windows or not candidates:
        return TrlSampledAnimation(samples=[], frame_windows=windows, notes=notes)

    expected_count = min(len(windows), trl.unknown_count if trl.unknown_count else len(windows))
    if len(candidates) > expected_count:
        candidates = sorted(
            sorted(candidates, key=lambda item: item[0], reverse=True)[:expected_count],
            key=lambda item: item[1],
        )
        notes.append(f"sample tables selected {len(candidates)} from scan")
    else:
        candidates = sorted(candidates, key=lambda item: item[1])

    sample_count = min(len(windows), len(candidates))
    packed = _packed_section(trl)
    animated_rotation_group = _group_by_kind(trl, "animated_rotation")
    animated_translation_group = _group_by_kind(trl, "animated_translation")
    if packed is None or animated_rotation_group is None or animated_translation_group is None:
        return TrlSampledAnimation(samples=[], frame_windows=windows, notes=notes)

    rotation_length = animated_rotation_group.length * 6
    translation_length = animated_translation_group.length * 12
    samples: List[TrlKeySample] = []
    for window, (score, sample_relative, translation_relative) in zip(windows[:sample_count], candidates[:sample_count]):
        source_offset = packed.offset + sample_relative
        translation_offset = packed.offset + translation_relative
        rotations = _decode_rotation_records(
            data,
            source_offset,
            animated_rotation_group.bone_indices,
            rotation_length,
            rotation_variant,
            rotation_variants_by_bone,
        )
        translations = _decode_translation_records(
            data,
            translation_offset,
            animated_translation_group.bone_indices,
            translation_length,
        )
        samples.append(
            TrlKeySample(
                frame=window.start_frame + 1,
                rotations=rotations,
                translations=translations,
                source_offset=source_offset,
                translation_offset=translation_offset,
                score=score,
            )
        )

    if samples:
        notes.append(f"sampled pose tables {len(samples)}/{trl.unknown_count}")
    if len(windows) != len(samples):
        notes.append(f"frame windows {len(windows)}, samples {len(samples)}")

    return TrlSampledAnimation(samples=samples, frame_windows=windows, notes=notes)


def _decode_dense_gap_samples(
    data: bytes,
    trl: TrlFile,
    samples: Sequence[TrlKeySample],
    windows: Sequence[TrlFrameWindow],
    sample_index: int,
    rotation_variant: str,
    rotation_variants_by_bone: Mapping[int, str] | None,
) -> tuple[List[TrlKeySample], List[str]]:
    animated_rotation_group = _group_by_kind(trl, "animated_rotation")
    animated_translation_group = _group_by_kind(trl, "animated_translation")
    if animated_rotation_group is None or animated_translation_group is None:
        return [], []
    if sample_index + 1 >= len(samples) or sample_index >= len(windows):
        return [], []

    sample = samples[sample_index]
    next_sample = samples[sample_index + 1]
    window = windows[sample_index]
    if window.frame_count <= 0:
        return [], []

    header_offset = sample.source_offset - 16
    if header_offset < 0 or header_offset + 16 > len(data):
        return [], [f"gap {sample_index}: missing sample header"]

    rotation_table_size, translation_table_size, mask_prefix_size, _unk_b, rotation_block_size, translation_block_size, _pad_a, _pad_b = struct.unpack_from(
        "<HHHHHHHH",
        data,
        header_offset,
    )
    if rotation_table_size <= 0 or translation_table_size <= 0:
        return [], [f"gap {sample_index}: missing pose table sizes"]

    gap_start = sample.source_offset + rotation_table_size + translation_table_size
    next_header = next_sample.source_offset - 16
    meta_offset = gap_start
    mask_prefix_bits = mask_prefix_size * 8
    rotation_mask_bits = window.frame_count * animated_rotation_group.length
    translation_mask_bits = window.frame_count * animated_translation_group.length
    mask_bit_count = mask_prefix_bits + rotation_mask_bits + translation_mask_bits
    mask_length = (mask_bit_count + 7) // 8
    tail_padding = next_header - gap_start - mask_length - rotation_block_size - translation_block_size
    if tail_padding < 0:
        return [], [f"gap {sample_index}: negative packed key padding"]

    rotation_block_offset = meta_offset + mask_length
    translation_block_offset = rotation_block_offset + rotation_block_size
    if translation_block_offset + translation_block_size > next_header:
        return [], [f"gap {sample_index}: record blocks overrun"]

    rotation_record_count = rotation_block_size // 6
    translation_record_count = translation_block_size // 12
    translation_records = [
        struct.unpack_from("<fff", data, translation_block_offset + record_index * 12)
        for record_index in range(translation_record_count)
    ]

    meta = data[meta_offset:meta_offset + mask_length]
    rotation_start = mask_prefix_bits
    translation_start = mask_prefix_bits + rotation_mask_bits
    layout_name = "P_R_T" if mask_prefix_bits else "R_T"
    rotation_mask_count = _mask_count(meta, rotation_start, rotation_mask_bits)
    translation_mask_count = _mask_count(meta, translation_start, translation_mask_bits)
    notes: List[str] = []
    if rotation_mask_count != rotation_record_count or translation_mask_count != translation_record_count:
        notes.append(
            f"gap {sample_index}: mask/record count mismatch "
            f"R {rotation_mask_count}/{rotation_record_count} "
            f"T {translation_mask_count}/{translation_record_count} ({layout_name})"
        )

    current_rotations = dict(sample.rotations)
    current_translations = dict(sample.translations)
    rotation_cursor = 0
    translation_cursor = 0
    dense_samples: List[TrlKeySample] = []

    rotation_updates: List[Dict[int, Tuple[float, float, float, float]]] = [
        {} for _frame_index in range(window.frame_count)
    ]
    translation_updates: List[Dict[int, Tuple[float, float, float]]] = [
        {} for _frame_index in range(window.frame_count)
    ]

    # Masks are channel-major: all frames for bone/channel 0, then all frames
    # for bone/channel 1, and so on. Reading them frame-major gives plausible
    # record counts but assigns rotations to the wrong bones.
    for channel_index, bone_index in enumerate(animated_rotation_group.bone_indices):
        channel_start = rotation_start + channel_index * window.frame_count
        for frame_offset in range(window.frame_count):
            if not _bit_value(meta, channel_start + frame_offset):
                continue
            if rotation_cursor < rotation_record_count:
                variant = (
                    rotation_variants_by_bone.get(bone_index, rotation_variant)
                    if rotation_variants_by_bone
                    else rotation_variant
                )
                rotation_updates[frame_offset][bone_index] = _decode_rotation48(
                    data,
                    rotation_block_offset + rotation_cursor * 6,
                    variant,
                )
            rotation_cursor += 1

    for channel_index, bone_index in enumerate(animated_translation_group.bone_indices):
        channel_start = translation_start + channel_index * window.frame_count
        for frame_offset in range(window.frame_count):
            if not _bit_value(meta, channel_start + frame_offset):
                continue
            if translation_cursor < len(translation_records):
                translation_updates[frame_offset][bone_index] = translation_records[translation_cursor]
            translation_cursor += 1

    for frame_offset in range(window.frame_count):
        current_rotations.update(rotation_updates[frame_offset])
        current_translations.update(translation_updates[frame_offset])

        dense_samples.append(
            TrlKeySample(
                frame=sample.frame + frame_offset + 1,
                rotations=dict(current_rotations),
                translations=dict(current_translations),
                source_offset=rotation_block_offset,
                translation_offset=translation_block_offset,
                score=0.0,
                dense_key=True,
                rotation_updates=dict(rotation_updates[frame_offset]),
                translation_updates=dict(translation_updates[frame_offset]),
            )
        )

    if rotation_cursor != rotation_record_count or translation_cursor != translation_record_count:
        notes.append(
            f"gap {sample_index}: consumed R {rotation_cursor}/{rotation_record_count} "
            f"T {translation_cursor}/{translation_record_count}"
        )

    return dense_samples, notes


def decode_trl_dense_animation(
    data: bytes,
    trl: TrlFile,
    rest_translations: Dict[int, Tuple[float, float, float]] | None = None,
    rotation_variant: str = DEFAULT_ROTATION_VARIANT,
    rotation_variants_by_bone: Mapping[int, str] | None = None,
) -> TrlSampledAnimation:
    sampled = decode_trl_sampled_animation(
        data,
        trl,
        rest_translations,
        rotation_variant,
        rotation_variants_by_bone,
    )
    if len(sampled.samples) < 2:
        return sampled

    dense_samples: List[TrlKeySample] = []
    notes = list(sampled.notes)
    decoded_gaps = 0
    for sample_index, sample in enumerate(sampled.samples):
        dense_samples.append(sample)
        gap_samples, gap_notes = _decode_dense_gap_samples(
            data,
            trl,
            sampled.samples,
            sampled.frame_windows,
            sample_index,
            rotation_variant,
            rotation_variants_by_bone,
        )
        if gap_samples:
            decoded_gaps += 1
            dense_samples.extend(gap_samples)
        notes.extend(gap_notes[:2])

    if decoded_gaps:
        keyed_frames = sorted({sample.frame for sample in dense_samples})
        notes.append(f"in-window keys decoded {len(keyed_frames)} frames from {decoded_gaps} gaps")
        return TrlSampledAnimation(
            samples=sorted(dense_samples, key=lambda item: item.frame),
            frame_windows=sampled.frame_windows,
            notes=notes,
            dense_frame_keys=True,
        )

    notes.append("packed in-window keys not decoded")
    return TrlSampledAnimation(
        samples=sampled.samples,
        frame_windows=sampled.frame_windows,
        notes=notes,
        dense_frame_keys=False,
    )


def decode_trl_base_pose(
    data: bytes,
    trl: TrlFile,
    rotation_variant: str = DEFAULT_ROTATION_VARIANT,
    static_rotation_variant: str = DEFAULT_STATIC_ROTATION_VARIANT,
    rotation_variants_by_bone: Mapping[int, str] | None = None,
) -> TrlBasePose:
    """Decode the raw/base pose data that appears before the packed delta keys.

    This is intentionally conservative. It handles the constant channels and the
    raw base tables we can identify, but it leaves the bit-packed delta stream
    untouched until that layout is fully mapped.
    """

    rotations: Dict[int, Tuple[float, float, float, float]] = {}
    rotation_modes: Dict[int, str] = {}
    translations: Dict[int, Tuple[float, float, float]] = {}
    notes: List[str] = []

    static_rotation_group = _group_by_kind(trl, "static_rotation")
    static_rotation_section = _section_by_name(trl, "static_rotation")
    if static_rotation_group and static_rotation_section:
        offset, length = _static_rotation_start(static_rotation_section, static_rotation_group.length)
        decoded = _decode_rotation_records(data, offset, static_rotation_group.bone_indices, length, static_rotation_variant)
        rotations.update(decoded)
        for bone_index in decoded:
            rotation_modes[bone_index] = ROTATION_MODE_REST_DELTA
        notes.append(
            f"static rotations {len(decoded)}/{static_rotation_group.length} "
            f"({static_rotation_variant}, rest deltas)"
        )

    static_translation_group = _group_by_kind(trl, "static_translation")
    static_translation_section = _section_by_name(trl, "static_translation")
    if static_translation_group and static_translation_section:
        offset, length = _static_translation_start(static_translation_section, static_translation_group.length)
        decoded = _decode_translation_records(data, offset, static_translation_group.bone_indices, length)
        translations.update(decoded)
        notes.append(f"static translations {len(decoded)}/{static_translation_group.length}")

    packed = _packed_section(trl)
    raw_base_sizes = _raw_base_sizes(data, trl)
    animated_rotation_group = _group_by_kind(trl, "animated_rotation")
    if packed and animated_rotation_group:
        expected_length = animated_rotation_group.length * 6
        if raw_base_sizes and raw_base_sizes[0] == expected_length:
            decoded = _decode_rotation_records(
                data,
                packed.offset,
                animated_rotation_group.bone_indices,
                expected_length,
                rotation_variant,
                rotation_variants_by_bone,
            )
            rotations.update(decoded)
            for bone_index in decoded:
                rotation_modes[bone_index] = ROTATION_MODE_ABSOLUTE
            notes.append(
                f"animated rotation base {len(decoded)}/{animated_rotation_group.length} "
                "from packed start"
            )
        else:
            descriptor = _find_packed_descriptor(
                data,
                trl,
                [expected_length],
                preferred_section="small_block_b",
                strict_section="small_block_b",
            )
            if descriptor:
                source_offset, length, relative_offset = descriptor
                decoded = _decode_rotation_records(
                    data,
                    packed.offset + relative_offset,
                    animated_rotation_group.bone_indices,
                    length,
                    rotation_variant,
                    rotation_variants_by_bone,
                )
                rotations.update(decoded)
                for bone_index in decoded:
                    rotation_modes[bone_index] = ROTATION_MODE_ABSOLUTE
                notes.append(
                    f"animated rotation base {len(decoded)}/{animated_rotation_group.length} "
                    f"from 0x{source_offset:x}"
                )

    animated_translation_group = _group_by_kind(trl, "animated_translation")
    if packed and animated_translation_group:
        base_length = animated_translation_group.length * 12
        if raw_base_sizes and raw_base_sizes[1] == base_length:
            table_offset = packed.offset + raw_base_sizes[0]
            if _translation_table_is_plausible(data, table_offset, animated_translation_group.length):
                decoded = _decode_translation_records(
                    data,
                    table_offset,
                    animated_translation_group.bone_indices,
                    base_length,
                )
                translations.update(decoded)
                notes.append(
                    f"animated translation base {len(decoded)}/{animated_translation_group.length} "
                    "after rotation base"
                )
        else:
            accepted_lengths = [base_length, base_length + 4, base_length + 20, base_length + 24]
            descriptor = _find_packed_descriptor(data, trl, accepted_lengths)
            if descriptor:
                source_offset, length, relative_offset = descriptor
                extra = length - base_length
                table_offset = packed.offset + relative_offset + (20 if extra >= 20 else 0)
                if _translation_table_is_plausible(data, table_offset, animated_translation_group.length):
                    decoded = _decode_translation_records(
                        data,
                        table_offset,
                        animated_translation_group.bone_indices,
                        base_length,
                    )
                    translations.update(decoded)
                    notes.append(
                        f"animated translation base {len(decoded)}/{animated_translation_group.length} "
                        f"from 0x{source_offset:x}"
                    )

    if not notes:
        notes.append("no base pose channels decoded")

    return TrlBasePose(
        rotations=rotations,
        translations=translations,
        notes=notes,
        rotation_modes=rotation_modes,
    )


def hash_hex(hash_value: int) -> str:
    return f"0x{hash_value:08x}"
