"""blender object creation and the high-level geo import flow."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import bpy

from .geo_format import GeoHeader, MeshPart, parse_header, parse_mesh_parts, read_normals, read_positions, read_primary_uvs
from .texture import TextureResolver

def build_mesh_object(
    object_name: str,
    points: Sequence[Tuple[float, float, float]],
    uvs: Sequence[Tuple[float, float]],
    normals: Sequence[Tuple[float, float, float]],
    part: MeshPart,
    header: GeoHeader,
) -> bpy.types.Object:
    used_indices = sorted(set(index for face in part.faces for index in face))
    index_map = {old_index: new_index for new_index, old_index in enumerate(used_indices)}
    vertices = [points[index] for index in used_indices]
    faces = [tuple(index_map[index] for index in face) for face in part.faces]

    mesh = bpy.data.meshes.new(object_name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)

    if part.face_uvs is not None and uvs:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for polygon, uv_tri in zip(mesh.polygons, part.face_uvs):
            for loop_index, uv_index in zip(polygon.loop_indices, uv_tri):
                if uv_index < len(uvs):
                    uv_layer.data[loop_index].uv = uvs[uv_index]

    if part.face_normals is not None and normals:
        loop_normals = []
        for normal_tri in part.face_normals:
            for normal_index in normal_tri:
                loop_normals.append(normals[normal_index] if normal_index < len(normals) else (0.0, 0.0, 1.0))
        if len(loop_normals) == len(mesh.loops):
            for polygon in mesh.polygons:
                polygon.use_smooth = True
            try:
                mesh.normals_split_custom_set(loop_normals)
            except Exception as exc:
                mesh["geo_custom_normals_error"] = str(exc)
            mesh.update()

    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)

    mesh["geo_stored_size"] = header.stored_size
    mesh["geo_position_count"] = header.position_count
    mesh["geo_normal_count"] = header.normal_count
    mesh["geo_tangent_count"] = header.tangent_count
    mesh["geo_binormal_count"] = header.binormal_count
    mesh["geo_packed_attribute_count"] = header.packed_attribute_count
    mesh["geo_primary_uv_count"] = header.primary_uv_count
    mesh["geo_secondary_uv_count"] = header.secondary_uv_count
    mesh["geo_vertex_flags"] = header.vertex_flags
    mesh["geo_source_offset"] = f"0x{part.run.offset:x}"
    mesh["geo_source_mode"] = part.run.mode
    mesh["geo_source_name"] = part.name
    mesh["geo_source_words"] = part.run.word_count
    mesh["geo_source_faces"] = part.run.face_count
    mesh["geo_has_uvs"] = bool(part.face_uvs is not None and uvs)
    mesh["geo_has_custom_normals"] = bool(part.face_normals is not None and normals)
    mesh["geo_original_indices"] = ",".join(str(index) for index in used_indices)

    return obj


def blender_object_name(base_name: str, part: MeshPart) -> str:
    if part.run.mode.startswith("source_triangle_records_30:"):
        return part.name
    return f"{base_name}_{part.name}"


def import_geo(
    filepath: str,
    scale: float,
    flip_uv_v: bool,
    resolve_textures: bool,
    convert_tdt_textures: bool,
) -> List[bpy.types.Object]:
    with open(filepath, "rb") as handle:
        data = handle.read()

    header = parse_header(data)
    points = read_positions(data, header, scale)
    uvs = read_primary_uvs(data, header, flip_uv_v)
    normals = read_normals(data, header)
    parts = parse_mesh_parts(data, header, points)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    resolver = TextureResolver(filepath, convert_tdt_textures) if resolve_textures else None
    objects: List[bpy.types.Object] = []

    for part in parts:
        obj = build_mesh_object(blender_object_name(base_name, part), points, uvs, normals, part, header)
        if resolver:
            resolved = resolver.resolve(base_name, part.name)
            if resolved:
                material = resolver.material_for_resolution(resolved)
                if material:
                    obj.data.materials.append(material)
                    obj.data["geo_texture_group"] = resolved.group.key
                    obj.data["geo_texture_key"] = resolved.material_key
                    obj.data["geo_texture_match"] = resolved.method
                    obj.data["geo_texture_score"] = resolved.score
        objects.append(obj)

    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]

    return objects


