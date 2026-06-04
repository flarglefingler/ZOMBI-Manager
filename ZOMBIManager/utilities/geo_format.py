from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

GEO_MARKER = b"\xde\xc0\xde\xc0"
RENDER_INDEX_MARKER = b"\x01\x00\xde\xc0"
SOURCE_TRIANGLE_SENTINEL = b"\xff\xff"
POSITION_OFFSET = 0x7E
COUNT_OFFSET = 0x52
MIN_INDEX_RUN_WORDS = 64
MIN_USEFUL_FACES = 8
SOURCE_TRIANGLE_STRIDE = 30
SOURCE_TRIANGLE_WORDS = SOURCE_TRIANGLE_STRIDE // 2
SOURCE_TAIL_EXTRA_MIN_FACES = 8
AREA_EPSILON = 1.0e-12
TAIL_SCAN_FRACTION = 0.75
LIST_COHERENCE_MIN = 0.20

@dataclass
class GeoHeader:
    stored_size: int
    marker_offset: int
    table_count: int
    position_count: int
    normal_count: int
    tangent_count: int
    binormal_count: int
    packed_attribute_count: int
    primary_uv_count: int
    secondary_uv_count: int
    submesh_hint: int
    vertex_flags: int


@dataclass
class IndexRun:
    offset: int
    word_count: int
    face_count: int
    mode: str


@dataclass
class RenderIndexBlock:
    marker_offset: int
    payload_offset: int
    word_count: int
    aux_count: int
    words: List[int]
    position_mode: str = "raw"


@dataclass
class SourceTriangleBlock:
    preamble_offset: int
    triangle_offset: int
    triangle_count: int
    element_index: int
    flags: int
    name: str


@dataclass
class MeshPart:
    name: str
    faces: List[Tuple[int, int, int]]
    run: IndexRun
    face_uvs: List[Tuple[int, int, int]] | None = None
    face_normals: List[Tuple[int, int, int]] | None = None


@dataclass
class GeoModel:
    header: GeoHeader
    points: List[Tuple[float, float, float]]
    uvs: List[Tuple[float, float]]
    normals: List[Tuple[float, float, float]]
    parts: List[MeshPart]



def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _f32x3(data: bytes, offset: int) -> Tuple[float, float, float]:
    return struct.unpack_from("<fff", data, offset)



def parse_header(data: bytes) -> GeoHeader:
    if len(data) < POSITION_OFFSET + 12:
        raise ValueError("File is too small to be a supported GEO.")

    marker_offset = data.find(GEO_MARKER)
    if marker_offset < 0:
        raise ValueError("Missing c0 de c0 de GEO marker.")

    if data[0x14:0x18] != b"VISU":
        raise ValueError("Only VISU GEO payloads are currently supported.")

    counts = [_u32(data, COUNT_OFFSET + i * 4) for i in range(11)]
    position_count = counts[1]
    if position_count <= 0 or POSITION_OFFSET + position_count * 12 > len(data):
        raise ValueError(f"Implausible position count: {position_count}")

    return GeoHeader(
        stored_size=_u32(data, 0),
        marker_offset=marker_offset,
        table_count=counts[0],
        position_count=position_count,
        normal_count=counts[2],
        tangent_count=counts[3],
        binormal_count=counts[4],
        packed_attribute_count=counts[5],
        primary_uv_count=counts[6],
        secondary_uv_count=counts[7],
        submesh_hint=counts[8],
        vertex_flags=counts[10],
    )


def transform_point(
    point: Tuple[float, float, float],
    scale: float,
    axis_mode: str = "blender",
) -> Tuple[float, float, float]:
    x, y, z = point
    if axis_mode == "raw":
        return (x * scale, y * scale, z * scale)
    if axis_mode == "blender":
        return (x * scale, -z * scale, y * scale)
    raise ValueError(f"unknown geo axis mode: {axis_mode}")


def read_positions(
    data: bytes,
    header: GeoHeader,
    scale: float,
    axis_mode: str = "blender",
) -> List[Tuple[float, float, float]]:
    points = []
    for i in range(header.position_count):
        point = _f32x3(data, POSITION_OFFSET + i * 12)
        points.append(transform_point(point, scale, axis_mode))
    return points


def attribute_offsets(header: GeoHeader) -> Tuple[int, int, int, int, int, int]:
    normal_offset = POSITION_OFFSET + header.position_count * 12
    tangent_offset = normal_offset + header.normal_count * 12
    binormal_offset = tangent_offset + header.tangent_count * 12
    packed_offset = binormal_offset + header.binormal_count * 12
    primary_uv_offset = packed_offset + header.packed_attribute_count * 4
    secondary_uv_offset = primary_uv_offset + header.primary_uv_count * 8
    return (
        normal_offset,
        tangent_offset,
        binormal_offset,
        packed_offset,
        primary_uv_offset,
        secondary_uv_offset,
    )


def read_vectors(
    data: bytes,
    offset: int,
    count: int,
    axis_mode: str = "blender",
) -> List[Tuple[float, float, float]]:
    vectors = []
    for i in range(count):
        vector = _f32x3(data, offset + i * 12)
        vectors.append(transform_point(vector, 1.0, axis_mode))
    return vectors


def read_primary_uvs(data: bytes, header: GeoHeader, flip_v: bool) -> List[Tuple[float, float]]:
    _, _, _, _, primary_uv_offset, _ = attribute_offsets(header)
    uvs = []
    for i in range(header.primary_uv_count):
        u, v = struct.unpack_from("<ff", data, primary_uv_offset + i * 8)
        uvs.append((u, 1.0 - v if flip_v else v))
    return uvs


def read_normals(data: bytes, header: GeoHeader, axis_mode: str = "blender") -> List[Tuple[float, float, float]]:
    normal_offset, _, _, _, _, _ = attribute_offsets(header)
    return read_vectors(data, normal_offset, header.normal_count, axis_mode)


def triangle_area(points: Sequence[Tuple[float, float, float]], tri: Tuple[int, int, int]) -> float:
    ax, ay, az = points[tri[0]]
    bx, by, bz = points[tri[1]]
    cx, cy, cz = points[tri[2]]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    return (nx * nx + ny * ny + nz * nz) ** 0.5 * 0.5


def triangle_max_edge(points: Sequence[Tuple[float, float, float]], tri: Tuple[int, int, int]) -> float:
    ax, ay, az = points[tri[0]]
    bx, by, bz = points[tri[1]]
    cx, cy, cz = points[tri[2]]
    ab = ((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5
    bc = ((cx - bx) ** 2 + (cy - by) ** 2 + (cz - bz) ** 2) ** 0.5
    ca = ((ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2) ** 0.5
    return max(ab, bc, ca)


def triangle_list_faces(
    words: Sequence[int],
    points: Sequence[Tuple[float, float, float]],
    skip: int = 2,
) -> List[Tuple[int, int, int]]:
    faces: List[Tuple[int, int, int]] = []
    for i in range(skip, len(words) - 2, 3):
        tri = (words[i] & 0x7FFF, words[i + 1] & 0x7FFF, words[i + 2] & 0x7FFF)
        if len({tri[0], tri[1], tri[2]}) != 3:
            continue
        if triangle_area(points, tri) <= AREA_EPSILON:
            continue
        faces.append(tri)
    return faces


def render_block_faces(
    words: Sequence[int],
    points: Sequence[Tuple[float, float, float]],
    start_face: int = 0,
    end_face: int | None = None,
) -> List[Tuple[int, int, int]]:
    if end_face is None:
        end_face = len(words) // 3

    faces: List[Tuple[int, int, int]] = []
    first_word = max(0, start_face) * 3
    last_word = min(end_face * 3, len(words) - 2)

    for i in range(first_word, last_word, 3):
        tri = (words[i], words[i + 1], words[i + 2])
        if max(tri) >= len(points):
            continue
        if len({tri[0], tri[1], tri[2]}) != 3:
            continue
        if triangle_area(points, tri) <= AREA_EPSILON:
            continue
        faces.append(tri)

    return faces


def clean_geo_name(raw_name: bytes, fallback: str) -> str:
    text = raw_name.split(b"\0", 1)[0].decode("utf-8", "replace").strip()
    return text or fallback


def iter_source_triangle_blocks(data: bytes, header: GeoHeader) -> Iterable[SourceTriangleBlock]:
    index_hint = max(header.primary_uv_count, header.secondary_uv_count, header.position_count)
    max_triangle_count = max(index_hint * max(header.submesh_hint + 2, 4), 1024)
    sentinel_offset = data.find(SOURCE_TRIANGLE_SENTINEL)

    while sentinel_offset >= 2:
        offset = sentinel_offset - 2
        header_offset = offset + 4
        if header_offset + 10 <= len(data):
            element_index = _u16(data, offset)
            flags = _u16(data, header_offset)
            triangle_count = _u32(data, header_offset + 2)
            name_len = _u32(data, header_offset + 6)
            name_offset = header_offset + 10
            triangle_offset = name_offset + name_len
            triangle_end = triangle_offset + triangle_count * SOURCE_TRIANGLE_STRIDE

            if (
                0 < triangle_count <= max_triangle_count
                and 0 < name_len <= 128
                and triangle_end <= len(data)
            ):
                raw_name = data[name_offset:triangle_offset]
                stripped_name = raw_name.rstrip(b"\0")
                printable = sum(
                    1
                    for byte in stripped_name
                    if byte in b" _-.()0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                )
                name_is_valid = (
                    not stripped_name
                    or printable >= max(1, len(stripped_name) - 2)
                )
                if name_is_valid:
                    yield SourceTriangleBlock(
                        preamble_offset=offset,
                        triangle_offset=triangle_offset,
                        triangle_count=triangle_count,
                        element_index=element_index,
                        flags=flags,
                        name=clean_geo_name(raw_name, f"element_{element_index}"),
                    )

        sentinel_offset = data.find(SOURCE_TRIANGLE_SENTINEL, sentinel_offset + 1)


def source_block_faces(
    data: bytes,
    block: SourceTriangleBlock,
    header: GeoHeader,
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int]] | None,
    List[Tuple[int, int, int]] | None,
]:
    faces: List[Tuple[int, int, int]] = []
    face_uvs: List[Tuple[int, int, int]] = []
    face_normals: List[Tuple[int, int, int]] = []
    has_all_uvs = header.primary_uv_count > 0
    has_all_normals = header.normal_count > 0

    for i in range(block.triangle_count):
        offset = block.triangle_offset + i * SOURCE_TRIANGLE_STRIDE
        record = struct.unpack_from("<15H", data, offset)
        tri = record[0:3]
        if max(tri) >= len(points):
            continue
        if len({tri[0], tri[1], tri[2]}) != 3:
            continue
        if triangle_area(points, tri) <= AREA_EPSILON:
            continue

        uv = record[3:6]
        normal = record[6:9]
        if max(uv) >= header.primary_uv_count:
            has_all_uvs = False
        if max(normal) >= header.normal_count:
            has_all_normals = False

        faces.append(tri)
        face_uvs.append(uv)
        face_normals.append(normal)

    return (
        faces,
        face_uvs if has_all_uvs and len(face_uvs) == len(faces) else None,
        face_normals if has_all_normals and len(face_normals) == len(faces) else None,
    )


def source_block_end(block: SourceTriangleBlock) -> int:
    return block.triangle_offset + block.triangle_count * SOURCE_TRIANGLE_STRIDE


def find_source_mesh_parts(
    data: bytes,
    header: GeoHeader,
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[List[MeshPart], List[SourceTriangleBlock]]:
    parts: List[MeshPart] = []
    accepted_blocks: List[SourceTriangleBlock] = []

    for block in iter_source_triangle_blocks(data, header):
        faces, face_uvs, face_normals = source_block_faces(data, block, header, points)
        minimum_faces = max(MIN_USEFUL_FACES, int(block.triangle_count * 0.80))
        if len(faces) < minimum_faces:
            continue

        run = IndexRun(
            offset=block.triangle_offset,
            word_count=block.triangle_count * SOURCE_TRIANGLE_WORDS,
            face_count=len(faces),
            mode=f"source_triangle_records_30:{block.element_index}:{block.name}",
        )
        parts.append(MeshPart(name=block.name, faces=faces, run=run, face_uvs=face_uvs, face_normals=face_normals))
        accepted_blocks.append(block)

    return parts, accepted_blocks


def split_face_components(
    faces: Sequence[Tuple[int, int, int]],
) -> List[List[int]]:
    vertex_faces: dict[int, List[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_faces.setdefault(vertex_index, []).append(face_index)

    components: List[List[int]] = []
    visited: set[int] = set()

    for start_index in range(len(faces)):
        if start_index in visited:
            continue

        stack = [start_index]
        visited.add(start_index)
        component_indices: List[int] = []

        while stack:
            face_index = stack.pop()
            component_indices.append(face_index)
            for vertex_index in faces[face_index]:
                for neighbor_index in vertex_faces[vertex_index]:
                    if neighbor_index not in visited:
                        visited.add(neighbor_index)
                        stack.append(neighbor_index)

        components.append(component_indices)

    return components


def source_tail_faces(
    data: bytes,
    start_offset: int,
    end_offset: int,
    header: GeoHeader,
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int]] | None,
    List[Tuple[int, int, int]] | None,
]:
    faces: List[Tuple[int, int, int]] = []
    face_uvs: List[Tuple[int, int, int]] = []
    face_normals: List[Tuple[int, int, int]] = []
    record_count = max(0, (end_offset - start_offset) // SOURCE_TRIANGLE_STRIDE)

    for i in range(record_count):
        offset = start_offset + i * SOURCE_TRIANGLE_STRIDE
        record = struct.unpack_from("<15H", data, offset)
        tri = record[0:3]
        if max(tri) >= len(points):
            continue
        if len({tri[0], tri[1], tri[2]}) != 3:
            continue
        if triangle_area(points, tri) <= AREA_EPSILON:
            continue

        uv = record[3:6]
        normal = record[6:9]

        faces.append(tri)
        face_uvs.append(uv)
        face_normals.append(normal)

    return (
        faces,
        face_uvs if header.primary_uv_count and len(face_uvs) == len(faces) else None,
        face_normals if header.normal_count and len(face_normals) == len(faces) else None,
    )


def find_extra_source_tail_parts(
    data: bytes,
    header: GeoHeader,
    points: Sequence[Tuple[float, float, float]],
    source_parts: Sequence[MeshPart],
    source_blocks: Sequence[SourceTriangleBlock],
) -> List[MeshPart]:
    if header.submesh_hint <= len(source_parts):
        return []
    if not source_blocks:
        return []

    source_vertices = {
        vertex_index
        for part in source_parts
        for face in part.faces
        for vertex_index in face
    }
    parts: List[MeshPart] = []
    seen_components: set[frozenset[Tuple[int, int, int]]] = set()

    # rough fallback for source tails that lack a named block.
    for block in source_blocks:
        start_offset = source_block_end(block)
        render_cache_offset = data.find(RENDER_INDEX_MARKER, start_offset)
        end_offset = render_cache_offset if render_cache_offset >= 0 else len(data)
        tail_faces, tail_uvs, tail_normals = source_tail_faces(data, start_offset, end_offset, header, points)
        extra_indices = [
            index
            for index, face in enumerate(tail_faces)
            if set(face).isdisjoint(source_vertices)
        ]
        extra_faces = [tail_faces[index] for index in extra_indices]
        extra_uvs = [tail_uvs[index] for index in extra_indices] if tail_uvs is not None else None
        extra_normals = [tail_normals[index] for index in extra_indices] if tail_normals is not None else None

        for component_indices in split_face_components(extra_faces):
            if len(component_indices) < SOURCE_TAIL_EXTRA_MIN_FACES:
                continue

            component_faces = [extra_faces[index] for index in component_indices]
            component_uvs = [extra_uvs[index] for index in component_indices] if extra_uvs is not None else None
            component_normals = [extra_normals[index] for index in component_indices] if extra_normals is not None else None
            if component_uvs is not None and any(max(uv) >= header.primary_uv_count for uv in component_uvs):
                component_uvs = None
            if component_normals is not None and any(max(normal) >= header.normal_count for normal in component_normals):
                component_normals = None

            key = frozenset(tuple(sorted(face)) for face in component_faces)
            if key in seen_components:
                continue
            seen_components.add(key)

            run = IndexRun(
                offset=start_offset,
                word_count=((end_offset - start_offset) // SOURCE_TRIANGLE_STRIDE) * SOURCE_TRIANGLE_WORDS,
                face_count=len(component_faces),
                mode=f"source_tail_extra_records_30:{block.element_index}:{block.name}",
            )
            parts.append(
                MeshPart(
                    name=f"submesh_{len(source_parts) + len(parts)}",
                    faces=component_faces,
                    run=run,
                    face_uvs=component_uvs,
                    face_normals=component_normals,
                )
            )

    parts.sort(key=lambda part: (-len(part.faces), min(index for face in part.faces for index in face)))
    wanted = max(0, header.submesh_hint - len(source_parts))
    return parts[:wanted]


def iter_render_index_blocks(data: bytes) -> Iterable[RenderIndexBlock]:
    offset = data.find(RENDER_INDEX_MARKER)
    while offset >= 0:
        if offset + 20 <= len(data):
            marker = _u32(data, offset)
            word_count = _u32(data, offset + 4)
            index_size = _u32(data, offset + 8)
            sentinel = _u32(data, offset + 12)
            aux_count = _u32(data, offset + 16)
            payload_offset = offset + 20
            payload_size = word_count * 2

            if (
                marker == 0xC0DE0001
                and index_size == 2
                and sentinel == 0xFFFFFFFF
                and word_count > 0
                and payload_offset + payload_size <= len(data)
            ):
                        # headers are little-endian, but the u16 index payload is big-endian.
                words = [
                    struct.unpack_from(">H", data, payload_offset + i * 2)[0]
                    for i in range(word_count)
                ]
                yield RenderIndexBlock(
                    marker_offset=offset,
                    payload_offset=payload_offset,
                    word_count=word_count,
                    aux_count=aux_count,
                    words=words,
                )

        offset = data.find(RENDER_INDEX_MARKER, offset + 1)


def read_render_position_remap(data: bytes, block: RenderIndexBlock, header: GeoHeader) -> List[int] | None:
    offset = block.payload_offset + block.word_count * 2
    if offset + 8 > len(data):
        return None
    if data[offset:offset + 4] not in (b"\x3f\xc8\x00\x00", b"\x7b\x48\x00\x00"):
        return None

    count = _u32(data, offset + 4)
    values_offset = offset + 8
    if count <= 0 or values_offset + count * 2 > len(data):
        return None

    values = [
        _u16(data, values_offset + index * 2)
        for index in range(count)
    ]
    if any(value >= header.position_count for value in values):
        return None
    return values


def render_position_words(
    data: bytes,
    block: RenderIndexBlock,
    header: GeoHeader,
) -> tuple[List[int], str] | None:
    remap = read_render_position_remap(data, block, header)
    if remap is not None and all(word < len(remap) for word in block.words):
        return [remap[word] for word in block.words], "remap"

    if all(word < header.position_count for word in block.words):
        return list(block.words), "direct"

    return None


def find_render_index_block(
    data: bytes,
    header: GeoHeader,
) -> RenderIndexBlock | None:
    best_score = None
    best_block = None
    previous_block: RenderIndexBlock | None = None

    for block in iter_render_index_blocks(data):
        if block.word_count < 3 or block.word_count % 3:
            previous_block = block
            continue

        resolved = render_position_words(data, block, header)
        if resolved is None:
            previous_block = block
            continue
        resolved_words, resolve_mode = resolved

        # static/world GEOs often only have the paired render-cache blocks.
        # the first block indexes render vertices, the second block carries the
        # point index stream. Keep the second block even for tiny flat meshes.
        paired = (
            previous_block is not None
            and previous_block.word_count == block.word_count
            and previous_block.aux_count == block.aux_count
            and previous_block.marker_offset < block.marker_offset
        )
        if not paired and block.word_count < MIN_INDEX_RUN_WORDS:
            previous_block = block
            continue

        score = (0 if paired else 1, -block.word_count, block.marker_offset)
        if best_score is None or score < best_score:
            best_score = score
            best_block = RenderIndexBlock(
                marker_offset=block.marker_offset,
                payload_offset=block.payload_offset,
                word_count=block.word_count,
                aux_count=block.aux_count,
                words=resolved_words,
                position_mode=resolve_mode,
            )
        previous_block = block

    return best_block


def best_triangle_list_alignment(
    words: Sequence[int],
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[int, List[Tuple[int, int, int]]]:
    best_score = None
    best_skip = 0
    best_faces: List[Tuple[int, int, int]] = []

    for skip in range(6):
        faces = triangle_list_faces(words, points, skip=skip)
        if len(faces) < MIN_INDEX_RUN_WORDS // 3:
            continue

        areas = sorted(triangle_area(points, face) for face in faces)
        max_edges = sorted(triangle_max_edge(points, face) for face in faces)
        p95_area = areas[int(0.95 * (len(areas) - 1))]
        max_area = areas[-1]
        p95_edge = max_edges[int(0.95 * (len(max_edges) - 1))]
        max_edge = max_edges[-1]
        score = (p95_area, max_area, p95_edge, max_edge, -len(faces))

        if best_score is None or score < best_score:
            best_score = score
            best_skip = skip
            best_faces = faces

    return best_skip, best_faces


def triangle_strip_faces(
    words: Sequence[int],
    points: Sequence[Tuple[float, float, float]],
) -> List[Tuple[int, int, int]]:
    faces: List[Tuple[int, int, int]] = []
    p1 = p2 = None
    have = 0
    flip = False

    for atom in words[2:]:
        p3 = atom & 0x7FFF
        if atom & 0x8000:
            p1 = None
            p2 = p3
            have = 1
            flip = True
            continue

        if have >= 2:
            tri = (p2, p1, p3) if flip else (p1, p2, p3)
            if len({tri[0], tri[1], tri[2]}) == 3 and triangle_area(points, tri) > AREA_EPSILON:
                faces.append(tri)
            flip = not flip

        if have == 0:
            have = 1
        elif have == 1:
            have = 2
        p1 = p2
        p2 = p3

    return faces


def adjacent_face_coherence(faces: Sequence[Tuple[int, int, int]]) -> float:
    if len(faces) < 2:
        return 0.0

    shared = 0
    for previous, current in zip(faces, faces[1:]):
        if len(set(previous) & set(current)) >= 2:
            shared += 1
    return shared / (len(faces) - 1)


def best_faces_for_run(
    words: Sequence[int],
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[str, List[Tuple[int, int, int]]]:
    list_faces = triangle_list_faces(words, points)
    list_coherence = adjacent_face_coherence(list_faces)

    # some fallback chunks are strips rather than plain triangle lists.
    if list_faces and list_coherence >= LIST_COHERENCE_MIN:
        return "list", list_faces

    strip_faces = triangle_strip_faces(words, points)
    if len(strip_faces) > len(list_faces):
        return "strip", strip_faces
    return "list", list_faces


def find_triangle_table(
    data: bytes,
    header: GeoHeader,
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[List[Tuple[int, int, int]], List[IndexRun]]:
    if header.primary_uv_count <= 0:
        return [], []

    scan_start = int(len(data) * TAIL_SCAN_FRACTION)
    table_size = header.primary_uv_count * 6
    candidates = []

    for parity in (0, 1):
        start = scan_start if scan_start % 2 == parity else scan_start + 1
        for offset in range(start, len(data) - table_size, 2):
            if offset < 4:
                continue

            prefix_0 = _u16(data, offset - 4)
            prefix_1 = _u16(data, offset - 2)
            if prefix_0 > 0x100 or prefix_1 != 0:
                continue

            faces: List[Tuple[int, int, int]] = []
            degenerate_count = 0
            area_sum = 0.0
            valid = True

            for i in range(header.primary_uv_count):
                tri = struct.unpack_from("<HHH", data, offset + i * 6)
                if tri[0] >= len(points) or tri[1] >= len(points) or tri[2] >= len(points):
                    valid = False
                    break
                if len({tri[0], tri[1], tri[2]}) != 3:
                    degenerate_count += 1
                    continue
                area = triangle_area(points, tri)
                if area <= AREA_EPSILON:
                    degenerate_count += 1
                    continue
                area_sum += area
                faces.append(tri)

            if not valid or len(faces) < header.primary_uv_count * 0.90:
                continue

            candidates.append((degenerate_count, -len(faces), -area_sum, offset, faces))

    if not candidates:
        return [], []

    _, _, _, offset, faces = sorted(candidates)[0]
    return faces, [IndexRun(offset=offset, word_count=header.primary_uv_count * 3, face_count=len(faces), mode="triangle_table")]


def iter_index_runs(data: bytes, max_index: int) -> Iterable[Tuple[int, List[int]]]:
    # useful fallback index streams live late in the file; earlier tables can look similar.
    scan_start = int(len(data) * TAIL_SCAN_FRACTION)
    for parity in (0, 1):
        run_start = None
        words: List[int] = []
        for offset in range(parity, len(data) - 1, 2):
            value = _u16(data, offset)
            if offset >= scan_start and (value & 0x7FFF) < max_index:
                if run_start is None:
                    run_start = offset
                    words = [value]
                else:
                    words.append(value)
            else:
                if run_start is not None and len(words) >= MIN_INDEX_RUN_WORDS:
                    yield run_start, words
                run_start = None
                words = []
        if run_start is not None and len(words) >= MIN_INDEX_RUN_WORDS:
            yield run_start, words


def decode_faces(
    data: bytes,
    header: GeoHeader,
    points: Sequence[Tuple[float, float, float]],
    include_small_chunks: bool,
) -> Tuple[List[Tuple[int, int, int]], List[IndexRun]]:
    table_faces, table_runs = find_triangle_table(data, header, points)
    if table_faces:
        return table_faces, table_runs

    faces: List[Tuple[int, int, int]] = []
    seen = set()
    accepted_runs: List[IndexRun] = []
    min_faces = MIN_USEFUL_FACES if include_small_chunks else 64

    for offset, words in iter_index_runs(data, len(points)):
        mode, run_faces = best_faces_for_run(words, points)

        if len(run_faces) < min_faces:
            continue

        for tri in run_faces:
            key = tuple(sorted(tri))
            if key in seen:
                continue
            seen.add(key)
            faces.append(tri)
        accepted_runs.append(IndexRun(offset=offset, word_count=len(words), face_count=len(run_faces), mode=mode))

    if not faces:
        raise ValueError("No usable triangle chunks found. The file may use a different GEO variant.")

    return faces, accepted_runs


def parse_mesh_parts(
    data: bytes,
    header: GeoHeader,
    points: Sequence[Tuple[float, float, float]],
) -> List[MeshPart]:
    source_parts, source_blocks = find_source_mesh_parts(data, header, points)
    if source_parts:
        extra_parts = find_extra_source_tail_parts(data, header, points, source_parts, source_blocks)
        return source_parts + extra_parts

    render_block = find_render_index_block(data, header)
    if render_block:
        total_faces = render_block.word_count // 3
        faces = render_block_faces(render_block.words, points, 0, total_faces)
        if faces:
            run = IndexRun(
                offset=render_block.payload_offset,
                word_count=render_block.word_count,
                face_count=len(faces),
                mode=f"render_index_block_be_{render_block.position_mode}",
            )
            return [MeshPart(name="main", faces=faces, run=run)]

    table_faces, table_runs = find_triangle_table(data, header, points)
    if table_faces:
        return [MeshPart(name="main", faces=table_faces, run=table_runs[0])]

    faces, runs = decode_faces(data, header, points, False)
    return [MeshPart(name="mesh", faces=faces, run=runs[0])]


def parse_geo_model(
    data: bytes,
    scale: float = 1.0,
    axis_mode: str = "blender",
    flip_uv_v: bool = True,
) -> GeoModel:
    header = parse_header(data)
    points = read_positions(data, header, scale, axis_mode)
    uvs = read_primary_uvs(data, header, flip_uv_v)
    normals = read_normals(data, header, axis_mode)
    parts = parse_mesh_parts(data, header, points)
    return GeoModel(header=header, points=points, uvs=uvs, normals=normals, parts=parts)


def load_geo_model(
    path: str,
    scale: float = 1.0,
    axis_mode: str = "blender",
    flip_uv_v: bool = True,
) -> GeoModel:
    with open(path, "rb") as handle:
        return parse_geo_model(handle.read(), scale, axis_mode, flip_uv_v)
