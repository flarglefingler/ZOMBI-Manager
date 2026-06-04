"""blender object creation and high-level import flow."""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import bpy
from mathutils import Matrix, Quaternion, Vector

from .geo_format import (
    GeoHeader,
    MeshPart,
    SkinWeightList,
    parse_header,
    parse_mesh_parts,
    parse_skin_weight_lists,
    read_normals,
    read_positions,
    read_primary_uvs,
)
from .skn_format import SknBone, SknFile, bone_hash_sequence, mask_summary, parse_skn_file
from .texture import TextureResolver
from .trl_format import (
    DEFAULT_ROTATION_VARIANT,
    DEFAULT_STATIC_ROTATION_VARIANT,
    ROTATION_MODE_ABSOLUTE,
    ROTATION_MODE_REST_DELTA,
    ROTATION_VARIANTS,
    TrlBasePose,
    TrlKeySample,
    decode_trl_base_pose,
    decode_trl_dense_animation,
    hash_hex,
    parse_trl_file,
)
from . import bfz_archive, material_format, mdf_format, obj_format, texture, wor_format


DEBUG_ROTATION_VARIANTS = (
    DEFAULT_ROTATION_VARIANT,
    "p48_be_bottom_xzy",
    "p48_be_bottom_yxz",
    "p48_be_bottom_neg",
    "p48_be_bottom_w2",
    "p48_be_bottom_w2_xzy",
    "p48_be_top",
    "p48_le_bottom",
    "legacy_be_xyz_negw",
)

TRL_TRANSLATION_POLICY_AUTO = "auto"
TRL_TRANSLATION_POLICY_CHARACTER = "character"
TRL_TRANSLATION_POLICY_WEAPON = "weapon"
TRL_TRANSLATION_POLICIES = {
    TRL_TRANSLATION_POLICY_AUTO,
    TRL_TRANSLATION_POLICY_CHARACTER,
    TRL_TRANSLATION_POLICY_WEAPON,
}
WEAPON_TRANSLATION_BONE_MARKERS = {
    "bullet",
    "cartridge",
    "charger",
    "eject",
    "fxcanon",
    "fxeject",
    "fxgun",
    "hammer",
    "magazine",
    "reload",
    "realod",
    "slide",
    "trigger",
}
WORLD_TEXTURE_HINT_GENERIC_TOKENS = {
    "ach",
    "brk",
    "co",
    "di",
    "fa",
    "fur",
    "gen",
    "gl",
    "grd",
    "hub",
    "la",
    "low",
    "lou",
    "lt",
    "mat",
    "me",
    "pap",
    "pfb",
    "pl",
    "pr",
    "sh",
    "sig",
    "sta",
    "u",
    "vfx",
    "wo",
}


@dataclass(frozen=True)
class GeoMaterialInfo:
    path: str | None = None
    match_method: str = ""
    keys: Tuple[int, ...] = ()

    @property
    def stem(self) -> str:
        if not self.path:
            return ""
        return _world_asset_stem(self.path)


def build_mesh_object(
    object_name: str,
    points: Sequence[Tuple[float, float, float]],
    uvs: Sequence[Tuple[float, float]],
    normals: Sequence[Tuple[float, float, float]],
    part: MeshPart,
    header: GeoHeader,
    skin_weights: Sequence[SkinWeightList] | None = None,
    bone_names_by_hash: Dict[int, str] | None = None,
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

    assigned_groups = 0
    assigned_weights = 0
    weighted_vertices = 0
    if skin_weights and bone_names_by_hash:
        used_lookup = index_map
        weights_by_bone: Dict[str, Dict[int, float]] = {}
        totals_by_vertex: Dict[int, float] = {}

        for weight_list in skin_weights:
            bone_name = bone_names_by_hash.get(weight_list.bone_hash)
            if not bone_name:
                continue

            bone_weights = weights_by_bone.setdefault(bone_name, {})
            for entry in weight_list.entries:
                if entry.vertex_index not in used_lookup or entry.weight <= 0.0:
                    continue
                local_index = used_lookup[entry.vertex_index]
                bone_weights[local_index] = bone_weights.get(local_index, 0.0) + entry.weight
                totals_by_vertex[local_index] = totals_by_vertex.get(local_index, 0.0) + entry.weight

        weighted_vertices = len(totals_by_vertex)
        for bone_name, bone_weights in weights_by_bone.items():
            local_entries = []
            for local_index, weight in bone_weights.items():
                total = totals_by_vertex.get(local_index, 0.0)
                if total > 0.000001:
                    local_entries.append((local_index, max(0.0, min(1.0, weight / total))))
            if not local_entries:
                continue

            group = obj.vertex_groups.new(name=bone_name)
            for local_index, weight in local_entries:
                group.add([local_index], weight, "REPLACE")
            assigned_groups += 1
            assigned_weights += len(local_entries)

    mesh["geo_skin_source_groups"] = len(skin_weights or [])
    mesh["geo_skin_groups"] = assigned_groups
    mesh["geo_skin_weights"] = assigned_weights
    mesh["geo_skin_weighted_vertices"] = weighted_vertices
    mesh["geo_skin_unweighted_vertices"] = max(0, len(vertices) - weighted_vertices)
    mesh["geo_skin_weights_normalized"] = bool(assigned_weights)

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


def game_to_blender_point(point: Sequence[float], scale: float) -> Tuple[float, float, float]:
    x, y, z = point
    return (x * scale, -z * scale, y * scale)


def game_to_blender_vector(vector: Sequence[float]) -> Tuple[float, float, float]:
    x, y, z = vector
    return (x, -z, y)


def game_to_blender_matrix(rows: Sequence[Sequence[float]]) -> Matrix:
    # Jade/LyN matrices store I, J, K axis vectors, not row-major transform
    # rows. The engine applies them as I*x + J*y + K*z.
    game_matrix = Matrix(
        (
            (rows[0][0], rows[1][0], rows[2][0]),
            (rows[0][1], rows[1][1], rows[2][1]),
            (rows[0][2], rows[1][2], rows[2][2]),
        )
    )
    convert = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    blender_matrix = convert @ game_matrix @ convert.inverted()
    result = Matrix.Identity(4)
    for row in range(3):
        for column in range(3):
            result[row][column] = blender_matrix[row][column]
    return result


def _vec_add(left: Sequence[float], right: Sequence[float]) -> Tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _vec_sub(left: Sequence[float], right: Sequence[float]) -> Tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _vec_scale(value: Sequence[float], amount: float) -> Tuple[float, float, float]:
    return (value[0] * amount, value[1] * amount, value[2] * amount)


def _vec_lerp(left: Sequence[float], right: Sequence[float], amount: float) -> Tuple[float, float, float]:
    return (
        left[0] + (right[0] - left[0]) * amount,
        left[1] + (right[1] - left[1]) * amount,
        left[2] + (right[2] - left[2]) * amount,
    )


def _vec_len(value: Sequence[float]) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def _vec_normalize(value: Sequence[float], fallback: Sequence[float]) -> Tuple[float, float, float]:
    length = _vec_len(value)
    if length <= 0.000001:
        return (fallback[0], fallback[1], fallback[2])
    return (value[0] / length, value[1] / length, value[2] / length)


def _quat_normalize(value: Sequence[float]) -> Tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 0.000001:
        return (0.0, 0.0, 0.0, 1.0)
    return (value[0] / length, value[1] / length, value[2] / length, value[3] / length)


def _quat_mul(left: Sequence[float], right: Sequence[float]) -> Tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_conjugate(value: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = value
    return (-x, -y, -z, w)


def _quat_slerp(
    left: Sequence[float],
    right: Sequence[float],
    amount: float,
) -> Tuple[float, float, float, float]:
    left_q = _quat_normalize(left)
    right_q = _quat_normalize(right)
    dot = sum(left_q[index] * right_q[index] for index in range(4))
    if dot < 0.0:
        right_q = tuple(-component for component in right_q)
        dot = -dot

    if dot > 0.9995:
        return _quat_normalize(
            tuple(left_q[index] + (right_q[index] - left_q[index]) * amount for index in range(4))
        )

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * amount
    sin_theta = math.sin(theta)
    scale_left = math.cos(theta) - dot * sin_theta / sin_theta_0
    scale_right = sin_theta / sin_theta_0
    return _quat_normalize(
        tuple(left_q[index] * scale_left + right_q[index] * scale_right for index in range(4))
    )


def _quat_rotate(rotation: Sequence[float], vector: Sequence[float]) -> Tuple[float, float, float]:
    quat = _quat_normalize(rotation)
    rotated = _quat_mul(_quat_mul(quat, (vector[0], vector[1], vector[2], 0.0)), _quat_conjugate(quat))
    return (rotated[0], rotated[1], rotated[2])


def _clean_skn_matrix(matrix: Sequence[float] | None) -> Tuple[float, ...]:
    if matrix is None:
        return (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    cleaned = list(matrix)
    cleaned[3] = 0.0
    cleaned[7] = 0.0
    cleaned[11] = 0.0
    cleaned[15] = 1.0
    return tuple(cleaned)


def _mat_mul_row(left: Sequence[float], right: Sequence[float]) -> Tuple[float, ...]:
    return tuple(
        sum(left[row * 4 + index] * right[index * 4 + column] for index in range(4))
        for row in range(4)
        for column in range(4)
    )


def _bone_positions_from_matrices(bones: Sequence[SknBone]) -> Dict[int, Tuple[float, float, float]]:
    by_index = {bone.index: bone for bone in bones}
    matrices: Dict[int, Tuple[float, ...]] = {}

    def resolve(bone: SknBone) -> Tuple[float, ...]:
        if bone.index in matrices:
            return matrices[bone.index]

        local = _clean_skn_matrix(bone.matrix)
        if bone.parent_index in by_index:
            matrix = _mat_mul_row(local, resolve(by_index[bone.parent_index]))
        else:
            matrix = local

        matrices[bone.index] = matrix
        return matrix

    for bone in bones:
        resolve(bone)

    return {
        bone_index: (matrix[12], matrix[13], matrix[14])
        for bone_index, matrix in matrices.items()
    }


def _bone_positions_from_pose_block(skn: SknFile) -> Dict[int, Tuple[float, float, float]]:
    if not skn.pose_blocks:
        return {}

    transforms = skn.pose_blocks[0].transforms
    if len(transforms) < skn.bone_count:
        return {}

    by_index = {bone.index: bone for bone in skn.bones}
    local = {transform.index: transform for transform in transforms}
    positions: Dict[int, Tuple[float, float, float]] = {}
    rotations: Dict[int, Tuple[float, float, float, float]] = {}

    def resolve(bone: SknBone) -> None:
        if bone.index in positions:
            return

        transform = local.get(bone.index)
        if transform is None:
            return

        local_rotation = _quat_normalize(transform.rotation)
        local_translation = transform.translation
        if bone.parent_index in by_index:
            resolve(by_index[bone.parent_index])
            parent_position = positions[bone.parent_index]
            parent_rotation = rotations[bone.parent_index]
            positions[bone.index] = _vec_add(parent_position, _quat_rotate(parent_rotation, local_translation))
            rotations[bone.index] = _quat_normalize(_quat_mul(parent_rotation, local_rotation))
        else:
            positions[bone.index] = local_translation
            rotations[bone.index] = local_rotation

    for bone in skn.bones:
        resolve(bone)

    return positions


def _bone_pose_globals(
    skn: SknFile,
) -> Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    if not skn.pose_blocks:
        return {}

    transforms = skn.pose_blocks[0].transforms
    if len(transforms) < skn.bone_count:
        return {}

    by_index = {bone.index: bone for bone in skn.bones}
    local = {transform.index: transform for transform in transforms}
    globals_by_index: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = {}

    def resolve(bone: SknBone) -> None:
        if bone.index in globals_by_index:
            return

        transform = local.get(bone.index)
        if transform is None:
            return

        local_rotation = _quat_normalize(transform.rotation)
        local_translation = transform.translation
        if bone.parent_index in by_index:
            resolve(by_index[bone.parent_index])
            parent = globals_by_index.get(bone.parent_index)
            if parent is None:
                globals_by_index[bone.index] = (local_translation, local_rotation)
                return

            parent_position, parent_rotation = parent
            position = _vec_add(parent_position, _quat_rotate(parent_rotation, local_translation))
            rotation = _quat_normalize(_quat_mul(parent_rotation, local_rotation))
            globals_by_index[bone.index] = (position, rotation)
        else:
            globals_by_index[bone.index] = (local_translation, local_rotation)

    for bone in skn.bones:
        resolve(bone)

    return globals_by_index


def _bone_positions_game(skn: SknFile) -> Dict[int, Tuple[float, float, float]]:
    pose_globals = _bone_pose_globals(skn)
    if len(pose_globals) >= skn.bone_count:
        return {
            bone_index: transform[0]
            for bone_index, transform in pose_globals.items()
        }

    pose_positions = _bone_positions_from_pose_block(skn)
    if len(pose_positions) >= skn.bone_count:
        return pose_positions
    return _bone_positions_from_matrices(skn.bones)


def _bone_tail(
    bone: SknBone,
    bones: Sequence[SknBone],
    positions: Dict[int, Tuple[float, float, float]],
    scale: float,
    bone_length: float,
) -> Tuple[float, float, float]:
    head = positions[bone.index]
    children = [child for child in bones if child.parent_index == bone.index and child.index in positions]
    useful_children = [positions[child.index] for child in children if _vec_len(_vec_sub(positions[child.index], head)) > 0.000001]

    if useful_children:
        average = (
            sum(point[0] for point in useful_children) / len(useful_children),
            sum(point[1] for point in useful_children) / len(useful_children),
            sum(point[2] for point in useful_children) / len(useful_children),
        )
        return average

    fallback_len = max(0.01, bone_length)
    if bone.parent_index in positions:
        direction = _vec_normalize(_vec_sub(head, positions[bone.parent_index]), (0.1, 0.0, 0.0))
    else:
        direction = (0.1, 0.0, 0.0)
    return _vec_add(head, _vec_scale(direction, fallback_len / max(scale, 0.000001)))


def _bone_tail_from_pose_axis(
    bone: SknBone,
    bones: Sequence[SknBone],
    pose_globals: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]],
    scale: float,
    bone_length: float,
) -> Tuple[float, float, float]:
    head, rotation = pose_globals[bone.index]
    children = [child for child in bones if child.parent_index == bone.index and child.index in pose_globals]
    child_distances = [
        _vec_len(_vec_sub(pose_globals[child.index][0], head))
        for child in children
        if _vec_len(_vec_sub(pose_globals[child.index][0], head)) > 0.000001
    ]
    if child_distances:
        length = sum(child_distances) / len(child_distances)
    else:
        length = max(0.01, bone_length) / max(scale, 0.000001)

    # In LyN/Jade skeletons, the local x axis is the bone-length axis. Blender
    # bones use local y for length, so point the edit bone tail along game x.
    direction = _quat_rotate(rotation, (1.0, 0.0, 0.0))
    direction = _vec_normalize(direction, (1.0, 0.0, 0.0))
    return _vec_add(head, _vec_scale(direction, length))


def _set_active_object(obj: bpy.types.Object) -> None:
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def link_objects_to_armature(
    objects: Iterable[bpy.types.Object],
    armature_object: bpy.types.Object | None,
    add_modifier: bool,
) -> None:
    if armature_object is None or armature_object.type != "ARMATURE":
        return

    for obj in objects:
        obj.parent = armature_object
        try:
            obj.matrix_parent_inverse = armature_object.matrix_world.inverted()
        except Exception:
            pass
        obj["geo_armature"] = armature_object.name

        if add_modifier and obj.type == "MESH":
            modifier = obj.modifiers.get("Armature") or obj.modifiers.new("Armature", "ARMATURE")
            modifier.object = armature_object
            if hasattr(modifier, "use_vertex_groups"):
                modifier.use_vertex_groups = True
            if hasattr(modifier, "use_bone_envelopes"):
                modifier.use_bone_envelopes = False
            if len(obj.vertex_groups) == 0:
                obj["geo_armature_warning"] = "linked to armature, but no matching GEO skin weights were found"


def active_armature(context) -> bpy.types.Object | None:
    obj = getattr(context, "object", None)
    if obj is not None and obj.type == "ARMATURE":
        return obj

    for selected in getattr(context, "selected_objects", []):
        if selected.type == "ARMATURE":
            return selected

    return None


def scene_armature(context) -> bpy.types.Object | None:
    armatures = [obj for obj in context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) == 1:
        return armatures[0]
    return None


def armature_hash_sequence(armature_object: bpy.types.Object) -> List[int]:
    bones = list(armature_object.data.bones)
    bones.sort(key=lambda bone: int(bone.get("skn_index", 999999)))

    hashes: List[int] = []
    for bone in bones:
        value = bone.get("skn_hash")
        if isinstance(value, int):
            hashes.append(value)
        elif isinstance(value, str):
            try:
                hashes.append(int(value, 16))
            except ValueError:
                continue
    if not hashes:
        skn_path = _armature_skn_path(armature_object)
        if skn_path:
            try:
                hashes = bone_hash_sequence(parse_skn_file(skn_path))
            except Exception:
                pass
    return hashes


def armature_bone_names_by_hash(armature_object: bpy.types.Object) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for bone in armature_object.data.bones:
        value = bone.get("skn_hash")
        if isinstance(value, int):
            names[value] = bone.name
        elif isinstance(value, str):
            try:
                names[int(value, 16)] = bone.name
            except ValueError:
                continue
    if not names:
        skn_path = _armature_skn_path(armature_object)
        if skn_path:
            try:
                skn = parse_skn_file(skn_path)
                bones_by_index = {int(bone.get("skn_index", 999999)): bone for bone in armature_object.data.bones}
                bones_by_name = {bone.name: bone for bone in armature_object.data.bones}
                for skn_bone in skn.bones:
                    armature_bone = bones_by_index.get(skn_bone.index) or bones_by_name.get(skn_bone.name)
                    if armature_bone is not None:
                        names[skn_bone.hash_value] = armature_bone.name
            except Exception:
                pass
    return names


def import_skn(filepath: str, scale: float = 1.0, bone_length: float = 0.05) -> bpy.types.Object:
    skn = parse_skn_file(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    armature_data = bpy.data.armatures.new(base_name)
    armature_object = bpy.data.objects.new(base_name, armature_data)
    bpy.context.collection.objects.link(armature_object)

    armature_data["skn_path"] = os.path.abspath(filepath)
    armature_data["skn_version"] = skn.version
    armature_data["skn_stored_size"] = skn.stored_size
    armature_data["skn_bone_count"] = skn.bone_count
    armature_data["skn_root_matrix"] = ",".join(f"{value:.9g}" for value in skn.root_matrix)
    armature_data["skn_hashes"] = ",".join(hash_hex(value) for value in bone_hash_sequence(skn))
    armature_data["skn_masks"] = ",".join(mask.name for mask in skn.masks)
    armature_data["skn_pose_blocks"] = ",".join(f"{block.tag}@0x{block.offset:x}" for block in skn.pose_blocks)
    armature_data["skn_pose_source"] = "30SE compact transforms" if skn.pose_blocks and skn.pose_blocks[0].transforms else "bone matrices"
    armature_data["skn_scale"] = scale
    for mask in skn.masks:
        armature_data[f"skn_mask_{mask.name}"] = mask_summary(mask)

    pose_globals = _bone_pose_globals(skn)
    positions_game = {
        bone_index: transform[0]
        for bone_index, transform in pose_globals.items()
    } if len(pose_globals) >= skn.bone_count else _bone_positions_game(skn)
    tails_game = {}
    for bone in skn.bones:
        if bone.index in pose_globals:
            tails_game[bone.index] = _bone_tail_from_pose_axis(bone, skn.bones, pose_globals, scale, bone_length)
        else:
            tails_game[bone.index] = _bone_tail(bone, skn.bones, positions_game, scale, bone_length)

    _set_active_object(armature_object)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones: Dict[int, bpy.types.EditBone] = {}
    for bone in skn.bones:
        edit_bone = armature_data.edit_bones.new(bone.name)
        edit_bone.head = game_to_blender_point(positions_game[bone.index], scale)
        edit_bone.tail = game_to_blender_point(tails_game[bone.index], scale)
        if _vec_len(_vec_sub(edit_bone.tail, edit_bone.head)) <= 0.000001:
            edit_bone.tail = _vec_add(edit_bone.head, (0.0, 0.0, max(0.01, bone_length * scale)))
        edit_bones[bone.index] = edit_bone

    for bone in skn.bones:
        edit_bone = edit_bones.get(bone.index)
        pose_global = pose_globals.get(bone.index)
        if edit_bone is None or pose_global is None:
            continue

        _position, rotation = pose_global
        roll_axis = Vector(game_to_blender_vector(_quat_rotate(rotation, (0.0, 0.0, -1.0))))
        if roll_axis.length <= 0.000001:
            continue
        try:
            edit_bone.align_roll(roll_axis.normalized())
        except Exception as exc:
            armature_data["skn_roll_error"] = f"{bone.name}: {exc}"

    for bone in skn.bones:
        parent = edit_bones.get(bone.parent_index)
        if parent is not None:
            edit_bones[bone.index].parent = parent
            edit_bones[bone.index].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    pose_transforms = {}
    if skn.pose_blocks and skn.pose_blocks[0].transforms:
        pose_transforms = {transform.index: transform for transform in skn.pose_blocks[0].transforms}

    for bone in skn.bones:
        data_bone = armature_data.bones.get(bone.name)
        if not data_bone:
            continue
        data_bone["skn_index"] = bone.index
        data_bone["skn_parent_index"] = bone.parent_index
        data_bone["skn_hash"] = hash_hex(bone.hash_value)
        data_bone["skn_record_offset"] = f"0x{bone.record_offset:x}"
        data_bone["skn_flags"] = ",".join(str(value) for value in bone.flags)
        if bone.matrix is not None:
            data_bone["skn_local_matrix"] = ",".join(f"{value:.9g}" for value in bone.matrix)
        transform = pose_transforms.get(bone.index)
        if transform is not None:
            data_bone["skn_pose_rotation"] = ",".join(f"{value:.9g}" for value in transform.rotation)
            data_bone["skn_pose_translation"] = ",".join(f"{value:.9g}" for value in transform.translation)

    _set_active_object(armature_object)
    return armature_object


def _set_action_range(action: bpy.types.Action, frame_start: int, frame_end: int) -> None:
    if hasattr(action, "use_frame_range"):
        action.use_frame_range = True
        action.frame_start = frame_start
        action.frame_end = frame_end


def _add_action_range_marker(
    action: bpy.types.Action,
    armature_object: bpy.types.Object,
    frame_start: int,
    frame_end: int,
) -> bool:
    armature_object["trl_import_marker"] = 0.0
    try:
        if hasattr(action, "id_root"):
            action.id_root = "OBJECT"
        fcurve = action.fcurves.new(data_path='["trl_import_marker"]')
        fcurve.keyframe_points.add(1)
        fcurve.keyframe_points[0].co = (frame_start, 0.0)
        fcurve.keyframe_points[1].co = (frame_end, 0.0)
        for point in fcurve.keyframe_points:
            point.interpolation = "CONSTANT"
        return True
    except Exception as exc:
        action["trl_marker_error"] = str(exc)
        return False


def _trl_group_summary(trl) -> str:
    return ",".join(f"{group.kind}:{group.length}" for group in trl.channel_groups)


def _trl_section_summary(trl) -> str:
    return ",".join(f"{section.name}@0x{section.offset:x}+0x{section.length:x}" for section in trl.sections)


def _armature_skn_path(armature_object: bpy.types.Object) -> str | None:
    value = armature_object.data.get("skn_path")
    if isinstance(value, str) and os.path.exists(value):
        return value
    return None


def _armature_skn_scale(armature_object: bpy.types.Object) -> float:
    value = armature_object.data.get("skn_scale", 1.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _matrix_from_game_transform(
    position: Sequence[float],
    rotation: Sequence[float],
    scale: float,
) -> Matrix:
    x, y, z, w = _quat_normalize(rotation)
    game_rotation = Quaternion((w, x, y, z)).to_matrix().to_4x4()
    basis = Matrix(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    # game points are imported as (x, -z, y). The skeleton also needs a local
    # bone-axis remap: LyN/Jade uses x as the length axis, Blender uses y.
    bone_axis = Matrix(((0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    blender_rotation = basis @ game_rotation @ bone_axis
    return Matrix.Translation(Vector(game_to_blender_point(position, scale))) @ blender_rotation


def _pose_matrix_basis(
    pose_bone: bpy.types.PoseBone,
    pose_matrix: Matrix,
    matrices_by_name: Dict[str, Matrix],
) -> Matrix:
    rest_matrix = pose_bone.bone.matrix_local
    parent = pose_bone.parent
    if parent is None:
        try:
            return pose_bone.bone.convert_local_to_pose(
                pose_matrix,
                rest_matrix,
                invert=True,
            )
        except Exception:
            return rest_matrix.inverted() @ pose_matrix

    parent_pose = matrices_by_name.get(parent.name, parent.bone.matrix_local)
    parent_rest = parent.bone.matrix_local
    try:
        return pose_bone.bone.convert_local_to_pose(
            pose_matrix,
            rest_matrix,
            parent_matrix=parent_pose,
            parent_matrix_local=parent_rest,
            invert=True,
        )
    except Exception:
        return rest_matrix.inverted() @ parent_rest @ parent_pose.inverted() @ pose_matrix


def _is_main_translation_bone(bone: SknBone, by_index: Dict[int, SknBone]) -> bool:
    return bone.parent_index not in by_index


def _is_supported_translation_bone(
    bone: SknBone,
    by_index: Dict[int, SknBone],
    translations: Dict[int, Tuple[float, float, float]],
    rest_translations: Dict[int, Tuple[float, float, float]],
) -> bool:
    if bone.index not in translations:
        return False
    return _is_main_translation_bone(bone, by_index)


def _is_weapon_translation_skn(
    skn: SknFile,
    armature_object: bpy.types.Object | None = None,
    trl_path: str = "",
) -> bool:
    if len(skn.bones) > 40:
        return False

    names = [bone.name.lower() for bone in skn.bones]
    joined = " ".join(names)
    marker_hits = sum(1 for marker in WEAPON_TRANSLATION_BONE_MARKERS if marker in joined)
    if marker_hits >= 2:
        return True

    context = " ".join(
        value.lower()
        for value in (
            os.path.basename(skn.path or ""),
            os.path.basename(trl_path or ""),
            armature_object.name if armature_object else "",
        )
        if value
    )
    if marker_hits >= 1 and re.search(r"\bw[0-9]{2}|weapon|wpn", context):
        return True

    return False


def _resolve_translation_policy(
    skn: SknFile,
    armature_object: bpy.types.Object | None,
    trl_path: str,
    requested_policy: str,
) -> str:
    policy = (requested_policy or TRL_TRANSLATION_POLICY_AUTO).lower()
    if policy not in TRL_TRANSLATION_POLICIES:
        raise ValueError(f"Unknown TRL translation policy: {requested_policy}")
    if policy != TRL_TRANSLATION_POLICY_AUTO:
        return policy
    if _is_weapon_translation_skn(skn, armature_object, trl_path):
        return TRL_TRANSLATION_POLICY_WEAPON
    return TRL_TRANSLATION_POLICY_CHARACTER


def _accepted_translation_indices(
    skn: SknFile,
    translations: Dict[int, Tuple[float, float, float]],
    translation_policy: str = TRL_TRANSLATION_POLICY_CHARACTER,
) -> set[int]:
    by_index = {bone.index: bone for bone in skn.bones}
    if translation_policy == TRL_TRANSLATION_POLICY_WEAPON:
        return {bone_index for bone_index in translations if bone_index in by_index}

    rest_translations = _rest_translations_by_index(skn)
    return {
        bone_index
        for bone_index in translations
        if bone_index in by_index and _is_supported_translation_bone(
            by_index[bone_index],
            by_index,
            translations,
            rest_translations,
        )
    }


def _pose_local_rotation(
    bone_index: int,
    rest_rotation: Tuple[float, float, float, float],
    base_pose: TrlBasePose,
) -> Tuple[float, float, float, float]:
    rotation = base_pose.rotations.get(bone_index)
    if rotation is None:
        return _quat_normalize(rest_rotation)

    mode = base_pose.rotation_modes.get(bone_index, ROTATION_MODE_ABSOLUTE)
    if mode == ROTATION_MODE_REST_DELTA:
        # Static TRL rotations behave like local offsets from the SKN bind pose.
        # Treating them as full replacements tears fingers, toes, and shoulders.
        return _quat_normalize(_quat_mul(rest_rotation, rotation))

    return _quat_normalize(rotation)


def _compose_pose_transforms(
    skn: SknFile,
    base_pose: TrlBasePose,
    translation_policy: str = TRL_TRANSLATION_POLICY_CHARACTER,
) -> Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    if not skn.pose_blocks or not skn.pose_blocks[0].transforms:
        return {}

    by_index = {bone.index: bone for bone in skn.bones}
    rest = {transform.index: transform for transform in skn.pose_blocks[0].transforms}
    rest_translations = {index: transform.translation for index, transform in rest.items()}
    globals_by_index: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = {}

    def resolve(bone: SknBone) -> None:
        if bone.index in globals_by_index:
            return
        transform = rest.get(bone.index)
        if transform is None:
            return

        local_translation = transform.translation
        if (
            translation_policy == TRL_TRANSLATION_POLICY_WEAPON
            and bone.index in base_pose.translations
        ) or (
            translation_policy == TRL_TRANSLATION_POLICY_CHARACTER
            and _is_supported_translation_bone(bone, by_index, base_pose.translations, rest_translations)
        ):
            local_translation = base_pose.translations[bone.index]

        if bone.parent_index in by_index:
            resolve(by_index[bone.parent_index])
            parent = globals_by_index.get(bone.parent_index)
            local_rotation = _pose_local_rotation(bone.index, transform.rotation, base_pose)
            if parent is None:
                globals_by_index[bone.index] = (local_translation, local_rotation)
                return
            parent_position, parent_rotation = parent
            world_position = _vec_add(parent_position, _quat_rotate(parent_rotation, local_translation))
            world_rotation = _quat_normalize(_quat_mul(parent_rotation, local_rotation))
            globals_by_index[bone.index] = (world_position, world_rotation)
        else:
            rotation = _pose_local_rotation(bone.index, transform.rotation, base_pose)
            globals_by_index[bone.index] = (local_translation, rotation)

    for bone in skn.bones:
        resolve(bone)

    return globals_by_index


def _rest_translations_by_index(skn: SknFile) -> Dict[int, Tuple[float, float, float]]:
    if not skn.pose_blocks or not skn.pose_blocks[0].transforms:
        return {}
    return {
        transform.index: transform.translation
        for transform in skn.pose_blocks[0].transforms
    }


def _linearize_action_curves(action: bpy.types.Action) -> None:
    for fcurve in action.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = "LINEAR"


def _fix_quaternion_curve_signs(action: bpy.types.Action) -> None:
    curves_by_path: Dict[str, Dict[int, bpy.types.FCurve]] = {}
    for fcurve in action.fcurves:
        if fcurve.data_path.endswith("rotation_quaternion"):
            curves_by_path.setdefault(fcurve.data_path, {})[fcurve.array_index] = fcurve

    for curves in curves_by_path.values():
        if any(index not in curves for index in range(4)):
            continue

        points_by_index = {
            index: {point.co[0]: point for point in curves[index].keyframe_points}
            for index in range(4)
        }
        frames = sorted(set().union(*(set(points) for points in points_by_index.values())))
        previous: Tuple[float, float, float, float] | None = None
        for frame in frames:
            points = [points_by_index[index].get(frame) for index in range(4)]
            if any(point is None for point in points):
                continue

            current = tuple(point.co[1] for point in points if point is not None)
            if previous is not None and sum(previous[index] * current[index] for index in range(4)) < 0.0:
                for point in points:
                    if point is None:
                        continue
                    point.co[1] = -point.co[1]
                    point.handle_left[1] = -point.handle_left[1]
                    point.handle_right[1] = -point.handle_right[1]
                current = tuple(-value for value in current)
            previous = current


def _key_pose_matrices(
    action: bpy.types.Action,
    armature_object: bpy.types.Object,
    skn: SknFile,
    transforms: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]],
    keyed_indices: set[int],
    translation_indices: set[int],
    frame: int,
    scale: float,
) -> int:
    matrices_by_index = {
        bone_index: _matrix_from_game_transform(position, rotation, scale)
        for bone_index, (position, rotation) in transforms.items()
    }
    matrices_by_name = {
        skn.bones[bone_index].name: matrix
        for bone_index, matrix in matrices_by_index.items()
        if 0 <= bone_index < len(skn.bones)
    }

    keyed = 0
    for bone in skn.bones:
        if bone.index not in keyed_indices:
            continue
        pose_bone = armature_object.pose.bones.get(bone.name)
        pose_matrix = matrices_by_index.get(bone.index)
        if pose_bone is None or pose_matrix is None:
            continue

        try:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = _pose_matrix_basis(pose_bone, pose_matrix, matrices_by_name)
            pose_bone.scale = (1.0, 1.0, 1.0)
            if bone.index not in translation_indices:
                # face/finger/toe records are usually rotation-only on character rigs.
                # keeping matrix-derived locations here tears small helpers away from rest.
                pose_bone.location = (0.0, 0.0, 0.0)
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            if bone.index in translation_indices:
                pose_bone.keyframe_insert(data_path="location", frame=frame)
            keyed += 1
        except Exception as exc:
            action["trl_pose_key_error"] = f"{bone.name}: {exc}"
            break

    return keyed


def _key_base_pose_action(
    action: bpy.types.Action,
    armature_object: bpy.types.Object,
    skn: SknFile,
    base_pose: TrlBasePose,
    frame_start: int,
    frame_end: int,
    restore_action: bool,
    translation_policy: str,
) -> int:
    if not base_pose.rotations and not base_pose.translations:
        return 0

    transforms = _compose_pose_transforms(skn, base_pose, translation_policy)
    if not transforms:
        return 0

    translation_indices = _accepted_translation_indices(
        skn,
        base_pose.translations,
        translation_policy,
    )
    keyed_indices = set(base_pose.rotations) | translation_indices
    if not keyed_indices:
        return 0

    scale = _armature_skn_scale(armature_object)
    previous_action = armature_object.animation_data.action if armature_object.animation_data else None
    armature_object.animation_data_create()
    armature_object.animation_data.action = action

    keyed = 0
    frame_current = bpy.context.scene.frame_current
    for frame in (frame_start, frame_end):
        bpy.context.scene.frame_set(frame)
        keyed += _key_pose_matrices(
            action,
            armature_object,
            skn,
            transforms,
            keyed_indices,
            translation_indices,
            frame,
            scale,
        )

    bpy.context.scene.frame_set(frame_current)
    if restore_action:
        armature_object.animation_data.action = previous_action
    return keyed


def _combined_sample_pose(skn: SknFile, base_pose: TrlBasePose, sample: TrlKeySample) -> TrlBasePose:
    rotations = dict(base_pose.rotations)
    rotations.update(sample.rotations)
    rotation_modes = dict(base_pose.rotation_modes)
    for bone_index in sample.rotations:
        rotation_modes[bone_index] = ROTATION_MODE_ABSOLUTE
    translations = dict(base_pose.translations)
    translations.update(sample.translations)
    return TrlBasePose(
        rotations=rotations,
        translations=translations,
        notes=[],
        rotation_modes=rotation_modes,
    )


def _interpolated_sample_pose(
    skn: SknFile,
    base_pose: TrlBasePose,
    samples: Sequence[TrlKeySample],
    frame: int,
) -> TrlBasePose:
    if not samples:
        return base_pose
    if len(samples) == 1 or frame <= samples[0].frame:
        return _combined_sample_pose(skn, base_pose, samples[0])
    if frame >= samples[-1].frame:
        return _combined_sample_pose(skn, base_pose, samples[-1])

    left = samples[0]
    right = samples[-1]
    for index in range(len(samples) - 1):
        if samples[index].frame <= frame <= samples[index + 1].frame:
            left = samples[index]
            right = samples[index + 1]
            break

    frame_span = max(1, right.frame - left.frame)
    amount = (frame - left.frame) / frame_span
    left_pose = _combined_sample_pose(skn, base_pose, left)
    right_pose = _combined_sample_pose(skn, base_pose, right)

    rotations = {}
    rotation_modes = {}
    for bone_index in set(left_pose.rotations) | set(right_pose.rotations):
        left_rotation = left_pose.rotations.get(bone_index)
        right_rotation = right_pose.rotations.get(bone_index)
        if left_rotation is not None and right_rotation is not None:
            rotations[bone_index] = _quat_slerp(left_rotation, right_rotation, amount)
        elif left_rotation is not None:
            rotations[bone_index] = left_rotation
        elif right_rotation is not None:
            rotations[bone_index] = right_rotation
        rotation_modes[bone_index] = right_pose.rotation_modes.get(
            bone_index,
            left_pose.rotation_modes.get(bone_index, ROTATION_MODE_ABSOLUTE),
        )

    translations = {}
    for bone_index in set(left_pose.translations) | set(right_pose.translations):
        left_translation = left_pose.translations.get(bone_index)
        right_translation = right_pose.translations.get(bone_index)
        if left_translation is not None and right_translation is not None:
            translations[bone_index] = _vec_lerp(left_translation, right_translation, amount)
        elif left_translation is not None:
            translations[bone_index] = left_translation
        elif right_translation is not None:
            translations[bone_index] = right_translation

    return TrlBasePose(
        rotations=rotations,
        translations=translations,
        notes=[],
        rotation_modes=rotation_modes,
    )


def _sample_key_frames(samples: Sequence[TrlKeySample], bake_frames: bool) -> List[int]:
    if not samples:
        return []
    ordered_frames = sorted({sample.frame for sample in samples})
    if not bake_frames:
        return ordered_frames
    return list(range(ordered_frames[0], ordered_frames[-1] + 1))


def _sample_rotation_keys(sample: TrlKeySample) -> Dict[int, Tuple[float, float, float, float]]:
    if sample.dense_key:
        updates = getattr(sample, "rotation_updates", None)
        if updates is not None:
            return updates
    return sample.rotations


def _sample_translation_keys(sample: TrlKeySample) -> Dict[int, Tuple[float, float, float]]:
    if sample.dense_key:
        updates = getattr(sample, "translation_updates", None)
        if updates is not None:
            return updates
    return sample.translations


def _rotation_at_frame(
    keys_by_frame: Dict[int, Tuple[float, float, float, float]],
    fallback: Tuple[float, float, float, float] | None,
    frame: int,
) -> Tuple[float, float, float, float] | None:
    if frame in keys_by_frame:
        return keys_by_frame[frame]
    if not keys_by_frame:
        return fallback

    frames = sorted(keys_by_frame)
    previous_frame = None
    next_frame = None
    for key_frame in frames:
        if key_frame < frame:
            previous_frame = key_frame
        elif key_frame > frame:
            next_frame = key_frame
            break

    if previous_frame is None:
        return keys_by_frame[next_frame] if next_frame is not None else fallback
    if next_frame is None:
        return keys_by_frame[previous_frame]

    amount = (frame - previous_frame) / max(1, next_frame - previous_frame)
    return _quat_slerp(keys_by_frame[previous_frame], keys_by_frame[next_frame], amount)


def _translation_at_frame(
    keys_by_frame: Dict[int, Tuple[float, float, float]],
    fallback: Tuple[float, float, float] | None,
    frame: int,
) -> Tuple[float, float, float] | None:
    if frame in keys_by_frame:
        return keys_by_frame[frame]
    if not keys_by_frame:
        return fallback

    frames = sorted(keys_by_frame)
    previous_frame = None
    next_frame = None
    for key_frame in frames:
        if key_frame < frame:
            previous_frame = key_frame
        elif key_frame > frame:
            next_frame = key_frame
            break

    if previous_frame is None:
        return keys_by_frame[next_frame] if next_frame is not None else fallback
    if next_frame is None:
        return keys_by_frame[previous_frame]

    amount = (frame - previous_frame) / max(1, next_frame - previous_frame)
    return _vec_lerp(keys_by_frame[previous_frame], keys_by_frame[next_frame], amount)


def _dense_channel_pose(
    skn: SknFile,
    base_pose: TrlBasePose,
    rotation_keys: Dict[int, Dict[int, Tuple[float, float, float, float]]],
    translation_keys: Dict[int, Dict[int, Tuple[float, float, float]]],
    frame: int,
) -> TrlBasePose:
    rotations: Dict[int, Tuple[float, float, float, float]] = {}
    rotation_modes: Dict[int, str] = {}
    for bone_index in set(base_pose.rotations) | set(rotation_keys):
        rotation = _rotation_at_frame(rotation_keys.get(bone_index, {}), base_pose.rotations.get(bone_index), frame)
        if rotation is not None:
            rotations[bone_index] = rotation
            rotation_modes[bone_index] = (
                ROTATION_MODE_ABSOLUTE
                if bone_index in rotation_keys
                else base_pose.rotation_modes.get(bone_index, ROTATION_MODE_ABSOLUTE)
            )

    translations: Dict[int, Tuple[float, float, float]] = {}
    for bone_index in set(base_pose.translations) | set(translation_keys):
        translation = _translation_at_frame(
            translation_keys.get(bone_index, {}),
            base_pose.translations.get(bone_index),
            frame,
        )
        if translation is not None:
            translations[bone_index] = translation

    return TrlBasePose(
        rotations=rotations,
        translations=translations,
        notes=[],
        rotation_modes=rotation_modes,
    )


def _key_dense_channel_action(
    action: bpy.types.Action,
    armature_object: bpy.types.Object,
    skn: SknFile,
    base_pose: TrlBasePose,
    samples: Sequence[TrlKeySample],
    restore_action: bool,
    translation_policy: str,
) -> int:
    if not samples:
        return 0

    rotation_keys: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    translation_keys: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
    translation_indices = _accepted_translation_indices(
        skn,
        base_pose.translations,
        translation_policy,
    )
    keyed_indices = set(base_pose.rotations) | translation_indices

    for sample in samples:
        for bone_index, rotation in _sample_rotation_keys(sample).items():
            rotation_keys.setdefault(bone_index, {})[sample.frame] = rotation
            keyed_indices.add(bone_index)

        translations = _sample_translation_keys(sample)
        for bone_index in _accepted_translation_indices(skn, translations, translation_policy):
            translation_keys.setdefault(bone_index, {})[sample.frame] = translations[bone_index]
            translation_indices.add(bone_index)
            keyed_indices.add(bone_index)

    frames = sorted(
        set(frame for keys in rotation_keys.values() for frame in keys)
        | set(frame for keys in translation_keys.values() for frame in keys)
    )
    if not frames or not keyed_indices:
        return 0

    scale = _armature_skn_scale(armature_object)
    previous_action = armature_object.animation_data.action if armature_object.animation_data else None
    armature_object.animation_data_create()
    armature_object.animation_data.action = action

    keyed = 0
    frame_current = bpy.context.scene.frame_current
    for frame in frames:
        pose = _dense_channel_pose(skn, base_pose, rotation_keys, translation_keys, frame)
        transforms = _compose_pose_transforms(skn, pose, translation_policy)

        bpy.context.scene.frame_set(frame)
        keyed += _key_pose_matrices(
            action,
            armature_object,
            skn,
            transforms,
            keyed_indices,
            translation_indices,
            frame,
            scale,
        )

    _linearize_action_curves(action)
    _fix_quaternion_curve_signs(action)
    bpy.context.scene.frame_set(frame_current)
    if restore_action:
        armature_object.animation_data.action = previous_action
    action["trl_dense_channel_frames"] = len(frames)
    action["trl_dense_channel_policy"] = "sparse channel keys are interpolated before pose baking"
    return keyed


def _key_sampled_pose_action(
    action: bpy.types.Action,
    armature_object: bpy.types.Object,
    skn: SknFile,
    base_pose: TrlBasePose,
    samples: Sequence[TrlKeySample],
    restore_action: bool,
    bake_frames: bool,
    translation_policy: str,
) -> int:
    if not samples:
        return 0

    scale = _armature_skn_scale(armature_object)
    previous_action = armature_object.animation_data.action if armature_object.animation_data else None
    armature_object.animation_data_create()
    armature_object.animation_data.action = action

    translation_indices = _accepted_translation_indices(skn, base_pose.translations, translation_policy)
    keyed_indices = set(base_pose.rotations) | translation_indices
    for sample in samples:
        keyed_indices.update(sample.rotations)
        accepted_sample_translations = _accepted_translation_indices(skn, sample.translations, translation_policy)
        translation_indices.update(accepted_sample_translations)
        keyed_indices.update(accepted_sample_translations)

    keyed = 0
    frame_current = bpy.context.scene.frame_current
    for frame in _sample_key_frames(samples, bake_frames):
        pose = _interpolated_sample_pose(skn, base_pose, samples, frame) if bake_frames else _combined_sample_pose(
            skn,
            base_pose,
            next(sample for sample in samples if sample.frame == frame),
        )
        transforms = _compose_pose_transforms(skn, pose, translation_policy)

        bpy.context.scene.frame_set(frame)
        keyed += _key_pose_matrices(
            action,
            armature_object,
            skn,
            transforms,
            keyed_indices,
            translation_indices,
            frame,
            scale,
        )

    _linearize_action_curves(action)
    _fix_quaternion_curve_signs(action)
    bpy.context.scene.frame_set(frame_current)
    if restore_action:
        armature_object.animation_data.action = previous_action
    return keyed


def import_trl(
    filepath: str,
    armature_object: bpy.types.Object,
    assign_action: bool = True,
    set_scene_fps: bool = False,
    rotation_variant: str = DEFAULT_ROTATION_VARIANT,
    action_suffix: str = "",
    bake_sampled_frames: bool = False,
    translation_policy: str = TRL_TRANSLATION_POLICY_AUTO,
) -> bpy.types.Action:
    if armature_object is None or armature_object.type != "ARMATURE":
        raise ValueError("Select an imported SKN armature before importing TRL.")
    if rotation_variant not in ROTATION_VARIANTS:
        raise ValueError(f"Unknown TRL rotation variant: {rotation_variant}")

    target_hashes = armature_hash_sequence(armature_object)
    if not target_hashes:
        raise ValueError("Selected armature has no SKN bone hashes.")

    with open(filepath, "rb") as handle:
        trl_data = handle.read()

    trl = parse_trl_file(filepath, target_hashes)
    skn_path = _armature_skn_path(armature_object)
    skn: SknFile | None = None
    skn_error = ""
    resolved_translation_policy = TRL_TRANSLATION_POLICY_CHARACTER
    if skn_path:
        try:
            skn = parse_skn_file(skn_path)
            resolved_translation_policy = _resolve_translation_policy(
                skn,
                armature_object,
                filepath,
                translation_policy,
            )
        except Exception as exc:
            skn_error = str(exc)

    base_pose = decode_trl_base_pose(
        trl_data,
        trl,
        rotation_variant,
    )
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    if action_suffix:
        base_name = f"{base_name}_{action_suffix}"
    action = bpy.data.actions.new(base_name)
    try:
        if hasattr(action, "id_root"):
            action.id_root = "OBJECT"
    except Exception:
        pass

    action["trl_path"] = os.path.abspath(filepath)
    action["trl_version"] = trl.version
    action["trl_stored_size"] = trl.stored_size
    action["trl_duration"] = trl.duration
    action["trl_fps"] = trl.fps
    action["trl_section_size"] = trl.section_size
    action["trl_frame_count"] = trl.frame_count
    action["trl_bone_count"] = trl.bone_count
    action["trl_unknown_count"] = trl.unknown_count
    action["trl_group_count"] = len(trl.channel_groups)
    action["trl_group_lengths"] = ",".join(str(group.length) for group in trl.channel_groups)
    action["trl_group_slots"] = _trl_group_summary(trl)
    action["trl_section_offsets"] = ",".join(f"0x{offset:x}" for offset in trl.section_offsets)
    action["trl_sections"] = _trl_section_summary(trl)
    action["trl_base_pose"] = "; ".join(base_pose.notes)
    action["trl_decoded_base_rotations"] = len(base_pose.rotations)
    action["trl_decoded_base_translations"] = len(base_pose.translations)
    action["trl_flags"] = f"0x{trl.flags:04x}"
    action["trl_block_offset"] = f"0x{trl.block_offset:x}"
    action["trl_bone_hash_offset"] = f"0x{trl.bone_hash_offset:x}" if trl.bone_hash_offset is not None else ""
    action["trl_payload_offset"] = f"0x{trl.payload_offset:x}" if trl.payload_offset is not None else ""
    action["trl_payload_size"] = trl.payload_size
    action["trl_bone_hashes"] = ",".join(hash_hex(value) for value in trl.bone_hashes)
    action["trl_rotation_variant"] = rotation_variant
    action["trl_static_rotation_variant"] = DEFAULT_STATIC_ROTATION_VARIANT
    action["trl_static_rotation_policy"] = "static rotations are local rest-pose deltas"
    action["trl_leaf_translation_policy"] = (
        "weapon rig: leaf and child translations allowed"
        if resolved_translation_policy == TRL_TRANSLATION_POLICY_WEAPON
        else "character rig: only root/main translations are keyed"
    )
    action["trl_translation_policy_requested"] = translation_policy
    action["trl_translation_policy_resolved"] = resolved_translation_policy
    action["trl_bake_sampled_frames"] = bake_sampled_frames
    action["zombiu_importer_version"] = "0.9.67"
    action["trl_dense_gap_layout"] = "mask prefix bits, channel-major masks, records, tail padding"
    action["trl_decode_status"] = "base pose decoded"

    target_set = set(target_hashes)
    trl_set = set(trl.bone_hashes)
    action["trl_unmapped_hashes"] = ",".join(hash_hex(value) for value in trl.bone_hashes if value not in target_set)
    action["trl_missing_target_hashes"] = ",".join(hash_hex(value) for value in target_hashes if value not in trl_set)

    _set_action_range(action, trl.frame_start, trl.frame_end)
    keyed_bones = 0
    sampled_key_count = 0
    if skn:
        try:
            accepted_base_translations = _accepted_translation_indices(
                skn,
                base_pose.translations,
                resolved_translation_policy,
            )
            action["trl_rotation_policy"] = "local bone rotations converted to Blender pose keys"
            action["trl_translation_policy"] = (
                "weapon rig: all keyed part translations"
                if resolved_translation_policy == TRL_TRANSLATION_POLICY_WEAPON
                else "character rig: root/main translations only"
            )
            action["trl_location_key_policy"] = "location curves are only written for accepted translation channels"
            action["trl_keyed_base_translations"] = len(accepted_base_translations)
            action["trl_skipped_base_translations"] = len(base_pose.translations) - len(accepted_base_translations)
            sampled_animation = decode_trl_dense_animation(
                trl_data,
                trl,
                _rest_translations_by_index(skn),
                rotation_variant,
            )
            action["trl_sampled_animation"] = "; ".join(sampled_animation.notes)
            action["trl_sample_count"] = len(sampled_animation.samples)
            action["trl_frame_windows"] = ",".join(
                f"{window.start_frame}:{window.frame_count}"
                for window in sampled_animation.frame_windows
            )
            action["trl_sample_offsets"] = ",".join(
                f"0x{sample.source_offset:x}"
                for sample in sampled_animation.samples
                if not sample.dense_key
            )
            action["trl_translation_offsets"] = ",".join(
                f"0x{sample.translation_offset:x}"
                for sample in sampled_animation.samples
                if not sample.dense_key
            )
            action["trl_dense_frame_keys"] = sampled_animation.dense_frame_keys
            sample_translation_total = sum(len(sample.translations) for sample in sampled_animation.samples)
            sample_translation_keyed = sum(
                len(_accepted_translation_indices(skn, sample.translations, resolved_translation_policy))
                for sample in sampled_animation.samples
            )
            action["trl_keyed_sample_translations"] = sample_translation_keyed
            action["trl_skipped_sample_translations"] = sample_translation_total - sample_translation_keyed

            if sampled_animation.samples:
                sampled_key_count = len(sampled_animation.samples)
                key_exact_frames = sampled_animation.dense_frame_keys or not bake_sampled_frames
                if sampled_animation.dense_frame_keys:
                    keyed_bones = _key_dense_channel_action(
                        action,
                        armature_object,
                        skn,
                        base_pose,
                        sampled_animation.samples,
                        restore_action=not assign_action,
                        translation_policy=resolved_translation_policy,
                    )
                else:
                    keyed_bones = _key_sampled_pose_action(
                        action,
                        armature_object,
                        skn,
                        base_pose,
                        sampled_animation.samples,
                        restore_action=not assign_action,
                        bake_frames=not key_exact_frames,
                        translation_policy=resolved_translation_policy,
                    )
                action["trl_baked_frame_count"] = len(_sample_key_frames(sampled_animation.samples, not key_exact_frames))
                action["trl_decode_status"] = (
                    f"sampled pose keys decoded ({sampled_key_count}); "
                    f"{action['trl_baked_frame_count']} frames keyed"
                )
                if sampled_animation.dense_frame_keys:
                    action["trl_decode_status"] += "; packed in-window keys decoded"
                else:
                    action["trl_decode_status"] += "; packed in-window keys not decoded yet"
            else:
                keyed_bones = _key_base_pose_action(
                    action,
                    armature_object,
                    skn,
                    base_pose,
                    trl.frame_start,
                    trl.frame_end,
                    restore_action=not assign_action,
                    translation_policy=resolved_translation_policy,
                )
        except Exception as exc:
            action["trl_base_pose_error"] = str(exc)
    else:
        action["trl_base_pose_error"] = skn_error or "selected armature has no readable skn_path"

    action["trl_has_fcurves"] = keyed_bones > 0
    action["trl_keyed_pose_keys"] = keyed_bones
    action["trl_keyed_base_pose_bones"] = keyed_bones // 2 if keyed_bones and not sampled_key_count else 0
    action["trl_keyed_sample_count"] = sampled_key_count

    if assign_action:
        armature_object.animation_data_create()
        armature_object.animation_data.action = action

    if set_scene_fps and trl.fps > 0:
        bpy.context.scene.render.fps = int(round(trl.fps))
        bpy.context.scene.frame_start = trl.frame_start
        bpy.context.scene.frame_end = trl.frame_end

    return action


def import_trl_debug_variants(
    filepath: str,
    armature_object: bpy.types.Object,
    assign_action: bool = True,
    set_scene_fps: bool = False,
    bake_sampled_frames: bool = False,
) -> List[bpy.types.Action]:
    """Import the same TRL with several 48-bit quaternion mappings.

    This is intentionally a debug helper. The correct component order is still
    the last open TRL problem, so keeping each candidate as its own action makes
    it easy to compare in Blender without changing the file or the armature.
    """

    actions: List[bpy.types.Action] = []
    for index, variant in enumerate(DEBUG_ROTATION_VARIANTS):
        action = import_trl(
            filepath,
            armature_object,
            assign_action=False,
            set_scene_fps=set_scene_fps and index == 0,
            rotation_variant=variant,
            action_suffix=f"rot_{variant}",
            bake_sampled_frames=bake_sampled_frames,
        )
        action["trl_debug_rotation_candidate"] = True
        actions.append(action)

    if assign_action and actions:
        armature_object.animation_data_create()
        armature_object.animation_data.action = actions[0]

    return actions


def import_geo(
    filepath: str,
    scale: float,
    flip_uv_v: bool,
    resolve_textures: bool,
    convert_tdt_textures: bool,
    texture_alpha_mode: str = "opaque",
    armature_object: bpy.types.Object | None = None,
    add_armature_modifier: bool = True,
    texture_query_hints: Sequence[str] | None = None,
    material_key_hints: Sequence[int] | None = None,
    material_info_override: GeoMaterialInfo | None = None,
    texture_resolver: TextureResolver | None = None,
    texture_search_dirs: Sequence[str] | None = None,
    mta_stems_by_key: Dict[int, str] | None = None,
) -> List[bpy.types.Object]:
    with open(filepath, "rb") as handle:
        data = handle.read()

    header = parse_header(data)
    points = read_positions(data, header, scale)
    uvs = read_primary_uvs(data, header, flip_uv_v)
    normals = read_normals(data, header)
    parts = parse_mesh_parts(data, header, points)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    material_info = material_info_override or material_info_for_geo(filepath)
    mat_keys = material_info.keys
    material_key_hints = tuple((key & 0xFFFFFFFF) for key in (material_key_hints or ()))
    material_query_hints = list(texture_query_hints or ())
    if material_info.stem:
        material_query_hints.append(material_info.stem)
    resolver = texture_resolver if resolve_textures else None
    if resolver is None and resolve_textures:
        resolver = TextureResolver(
            filepath,
            convert_tdt_textures,
            texture_alpha_mode,
            search_dirs=texture_search_dirs,
            mta_stems_by_key=mta_stems_by_key,
        )
    bone_names_by_hash = armature_bone_names_by_hash(armature_object) if armature_object else {}
    skin_weights = parse_skin_weight_lists(data, header, list(bone_names_by_hash)) if bone_names_by_hash else []
    objects: List[bpy.types.Object] = []

    for part in parts:
        material_index = _material_index_for_part(part, len(parts), mat_keys)
        obj = build_mesh_object(
            blender_object_name(base_name, part),
            points,
            uvs,
            normals,
            part,
            header,
            skin_weights,
            bone_names_by_hash,
        )
        if material_info.path:
            obj.data["geo_material_source"] = os.path.basename(material_info.path)
            obj.data["geo_material_source_path"] = material_info.path
            obj.data["geo_material_match"] = material_info.match_method
        if part.element_index is not None:
            obj.data["geo_element_index"] = part.element_index
        if material_index is not None:
            obj.data["geo_material_index"] = material_index
            if 0 <= material_index < len(mat_keys):
                obj.data["geo_material_key"] = f"{mat_keys[material_index] & 0xFFFFFFFF:08X}"
        if resolver:
            material_file_key = None
            material_key_source = ""
            if material_index is not None and 0 <= material_index < len(mat_keys):
                material_file_key = mat_keys[material_index]
                material_key_source = "mat-slot"
            elif material_key_hints:
                if part.element_index is not None and 0 <= part.element_index < len(material_key_hints):
                    material_file_key = material_key_hints[part.element_index]
                    material_key_source = "sidecar-slot"
                elif len(material_key_hints) == 1:
                    material_file_key = material_key_hints[0]
                    material_key_source = "sidecar"
            if material_file_key is not None:
                obj.data["geo_material_key"] = f"{material_file_key & 0xFFFFFFFF:08X}"
                if material_key_source:
                    obj.data["geo_material_key_source"] = material_key_source
            resolved = resolver.resolve(
                base_name,
                part.name,
                material_query_hints,
                material_file_key=material_file_key,
            )
            if resolved:
                material = resolver.material_for_resolution(resolved)
                if material:
                    obj.data.materials.append(material)
                    obj.data["geo_texture_group"] = resolved.group.key
                    obj.data["geo_texture_key"] = resolved.material_key
                    obj.data["geo_texture_match"] = resolved.method
                    obj.data["geo_texture_score"] = resolved.score
        objects.append(obj)

    link_objects_to_armature(objects, armature_object, add_armature_modifier)

    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]

    return objects


def _world_asset_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    return stem


def _world_asset_stem_variants(path: str) -> Tuple[str, ...]:
    stem = _world_asset_stem(path)
    variants = [stem]
    no_variant = re.sub(r"__variant_[0-9]+$", "", stem, flags=re.IGNORECASE)
    if no_variant != stem:
        variants.append(no_variant)
    stripped = re.sub(r"_[0-9]+$", "", stem)
    if stripped != stem:
        variants.append(stripped)
    stripped = re.sub(r"_[0-9]+$", "", no_variant)
    if stripped and stripped not in variants:
        variants.append(stripped)
    stripped = re.sub(r"\(\$[0-9A-Fa-f]+\)$", "", stripped)
    if stripped and stripped not in variants:
        variants.append(stripped)
    return tuple(variants)


def _parse_geo_material_info(path: str, match_method: str) -> GeoMaterialInfo:
    try:
        descriptor = material_format.parse_mat_file(path)
    except Exception:
        return GeoMaterialInfo()
    return GeoMaterialInfo(path=path, match_method=match_method, keys=descriptor.submaterial_keys)


def _is_world_material_prefix_match(geo_stem: str, mat_stem: str) -> bool:
    if len(mat_stem) < 6:
        return False
    if "default" in mat_stem.lower():
        return False
    if not geo_stem.lower().startswith(mat_stem.lower()):
        return False

    suffix = geo_stem[len(mat_stem):]
    if not suffix:
        return False

    first = suffix[0]
    return first.isdigit() or first.isupper() or first in {"_", " ", "-", "(", "$"}


def _shared_material_path_for_geo(filepath: str) -> Tuple[str | None, str]:
    directory = os.path.dirname(os.path.abspath(filepath))
    geo_stem = _world_asset_stem(filepath)
    best: Tuple[int, int, str] | None = None
    try:
        names = os.listdir(directory)
    except OSError:
        return None, ""

    for name in names:
        if not name.lower().endswith(".mat"):
            continue
        mat_stem = _world_asset_stem(name)
        if not _is_world_material_prefix_match(geo_stem, mat_stem):
            continue
        suffix_len = len(geo_stem) - len(mat_stem)
        score = (len(mat_stem), -suffix_len, os.path.join(directory, name))
        if best is None or score > best:
            best = score

    if best is None:
        return None, ""
    return best[2], "mat-shared-prefix"


def material_info_for_geo(filepath: str) -> GeoMaterialInfo:
    directory = os.path.dirname(os.path.abspath(filepath))
    variants = _world_asset_stem_variants(filepath)
    for index, stem in enumerate(variants):
        mat_path = os.path.join(directory, stem + ".mat")
        if not os.path.exists(mat_path):
            continue
        return _parse_geo_material_info(mat_path, "mat-exact" if index == 0 else "mat-variant")

    shared_path, method = _shared_material_path_for_geo(filepath)
    if shared_path:
        return _parse_geo_material_info(shared_path, method)
    return GeoMaterialInfo()


def material_keys_for_geo(filepath: str) -> Tuple[int, ...]:
    return material_info_for_geo(filepath).keys


def _material_index_for_part(part: MeshPart, part_count: int, mat_keys: Sequence[int]) -> int | None:
    if part.element_index is not None:
        return part.element_index
    if mat_keys and (part_count == 1 or len(mat_keys) == 1):
        return 0
    return None


def _unique_hint(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    result: List[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return tuple(result)


def _world_texture_hint_tokens(text: str) -> set[str]:
    return set(texture.texture_tokens(text)) - WORLD_TEXTURE_HINT_GENERIC_TOKENS


def import_wor_refs(
    filepath: str,
    create_ref_empties: bool = False,
    empty_limit: int = 250,
    import_meshes: bool = False,
    object_limit: int = 0,
    orient_upright: bool = True,
    scale: float = 1.0,
    flip_uv_v: bool = True,
    resolve_textures: bool = True,
    convert_tdt_textures: bool = True,
    texture_alpha_mode: str = "opaque",
    source_mode: str = "exported-folder",
    source_archive_path: str | None = None,
    deprecated_folder: bool = False,
) -> bpy.types.Object:
    with open(filepath, "rb") as handle:
        world = wor_format.parse_wor(handle.read(), os.path.basename(filepath))

    manifest_entries = wor_format.load_nearby_manifest_entries(filepath)
    resolved_objects, object_resolve_method = wor_format.object_entries_for_world(world, manifest_entries)
    resolved_count = len(resolved_objects)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    world_dir = os.path.dirname(os.path.abspath(filepath))
    geo_name_index = wor_format.build_geo_name_index(world_dir)
    manifest_mat_paths = [entry.path for entry in manifest_entries if entry.extension == ".mat"]
    if not manifest_mat_paths:
        manifest_mat_paths = [
            os.path.join(world_dir, name)
            for name in os.listdir(world_dir)
            if name.lower().endswith(".mat")
        ]
    material_link_keys = material_format.mat_link_keys(manifest_mat_paths)
    world_sidecar_keys = set()
    visual_geo_keys: Dict[int, List[int]] = {}
    for name in os.listdir(world_dir):
        if not name.lower().endswith((".mtn", ".vii")):
            continue
        path = os.path.join(world_dir, name)
        try:
            sidecar = obj_format.parse_sidecar_file(path)
        except Exception:
            continue
        world_sidecar_keys.update(sidecar.resource_keys)
        if not name.lower().endswith(".mtn"):
            continue
        vii_path = obj_format.find_sidecar(path, ".vii")
        if not vii_path:
            continue
        try:
            vii_sidecar = obj_format.parse_sidecar_file(vii_path)
        except Exception:
            continue
        visual_keys = set(sidecar.resource_keys) & set(vii_sidecar.resource_keys)
        if not visual_keys:
            continue
        geo_keys = [key for key in sidecar.resource_keys if key not in visual_keys]
        for visual_key in visual_keys:
            target = visual_geo_keys.setdefault(visual_key, [])
            for geo_key in geo_keys:
                if geo_key not in target:
                    target.append(geo_key)
    geo_key_index = wor_format.build_geo_key_index(geo_name_index, world_sidecar_keys)
    world_material_stems: List[Tuple[str, str]] = []
    for name in os.listdir(world_dir):
        lower = name.lower()
        if lower.endswith(".mta") or lower.endswith(".mat"):
            world_material_stems.append((_world_asset_stem(name), os.path.splitext(lower)[1]))

    collection = bpy.data.collections.new(base_name)
    bpy.context.scene.collection.children.link(collection)

    root = bpy.data.objects.new(base_name, None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 1.0
    collection.objects.link(root)

    root["wor_path"] = os.path.abspath(filepath)
    root["wor_source_mode"] = source_mode
    root["wor_source_archive"] = os.path.abspath(source_archive_path) if source_archive_path else ""
    root["wor_deprecated_folder_import"] = bool(deprecated_folder)
    root["wor_version"] = world.version
    root["wor_world_chunk_size"] = world.world_chunk_size
    root["wor_object_group_size"] = world.object_group_size
    root["wor_object_refs"] = world.object_count
    root["wor_resolved_refs"] = resolved_count
    root["wor_resolved_objects"] = resolved_count
    root["wor_manifest"] = wor_format.find_bfz_manifest(filepath) or ""
    root["wor_orient_upright"] = bool(orient_upright)
    root["wor_object_resolve"] = object_resolve_method
    root["wor_material_link_keys"] = len(material_link_keys)

    mdf_path = os.path.join(world_dir, base_name + ".mdf")
    mdf = None
    if os.path.exists(mdf_path):
        try:
            mdf = mdf_format.parse_mdf_file(mdf_path)
            root["mdf_path"] = mdf_path
            root["mdf_version"] = mdf.version
            root["mdf_link_count"] = mdf.link_count
            root["mdf_resource_keys"] = ",".join(f"{key & 0xFFFFFFFF:08X}" for key in mdf.resource_keys[:24])
            root["mdf_modifier_types"] = ",".join(
                f"{link.type_name}:{link.key_hex}" for link in mdf.modifier_links[:24]
            )
        except Exception as exc:
            root["mdf_path"] = mdf_path
            root["mdf_error"] = str(exc)
    if deprecated_folder:
        root["wor_import_note"] = "deprecated folder import; use BFZ World for duplicate-safe archive context"
    elif source_mode == "bfz-archive-cache":
        root["wor_import_note"] = "world imported from BFZ archive cache; duplicate same-name files are preserved"
    elif object_resolve_method.startswith("manifest-key"):
        root["wor_import_note"] = "object files resolved from BFZ resource keys in the manifest"
    else:
        root["wor_import_note"] = "object files resolved by export order because manifest keys did not match WOR refs"

    lines = [
        f"{base_name}",
        "",
        f"version: {world.version}",
        f"object refs: {world.object_count}",
        f"object files resolved: {resolved_count}",
        f"object resolve: {object_resolve_method}",
        f"source: {source_mode}",
        f"deprecated folder import: {'yes' if deprecated_folder else 'no'}",
        f"archive: {root['wor_source_archive'] or 'n/a'}",
        f"manifest: {root['wor_manifest'] or 'not found'}",
        f"mdf links: {mdf.link_count if mdf else 'n/a'}",
        f"meshes: {'on' if import_meshes else 'off'}",
        f"upright root: {'on' if orient_upright else 'off'}",
        "",
    ]

    def link_to_collection(obj: bpy.types.Object) -> None:
        if not any(existing == obj for existing in collection.objects):
            collection.objects.link(obj)
        for user_collection in list(obj.users_collection):
            if user_collection != collection:
                user_collection.objects.unlink(obj)

    def object_matrix(game_object: obj_format.ObjFile) -> Matrix:
        matrix = game_to_blender_matrix(game_object.matrix.rows)
        matrix.translation = Vector(game_to_blender_point(game_object.translation, scale))
        return matrix

    def sidecar_key_summary(path: str) -> str:
        if not path:
            return ""
        try:
            sidecar = obj_format.parse_sidecar_file(path)
        except Exception:
            return ""
        if not sidecar.key_hex:
            return ""
        return ",".join(sidecar.key_hex[:8])

    texture_hint_cache: Dict[Tuple[str, str], Tuple[str, ...]] = {}

    def texture_hints_for_object(object_path: str, geo_path: str) -> Tuple[str, ...]:
        cache_key = (object_path, geo_path)
        cached = texture_hint_cache.get(cache_key)
        if cached is not None:
            return cached

        object_stem = _world_asset_stem(object_path)
        geo_stem = _world_asset_stem(geo_path)
        hints: List[str] = [object_stem, geo_stem]
        exact_stems = {
            stem.lower()
            for source_path in (object_path, geo_path)
            for stem in _world_asset_stem_variants(source_path)
        }

        material_scores: List[Tuple[int, float, str]] = []
        base_query = " ".join(hints)
        query_tokens = _world_texture_hint_tokens(base_query)
        for material_stem, extension in world_material_stems:
            if extension == ".mta" and material_stem.lower() in exact_stems:
                hints.append(material_stem)
                continue
            if extension == ".mat" and material_stem.lower() == geo_stem.lower():
                hints.append(material_stem)
                continue
            if extension == ".mat" and material_stem.lower() in exact_stems:
                hints.append(material_stem)
                continue
            material_tokens = _world_texture_hint_tokens(material_stem)
            overlap = query_tokens & material_tokens
            if query_tokens and material_tokens and not overlap:
                continue
            score = max(
                texture.texture_name_score(base_query, material_stem),
                texture.texture_name_score(object_stem, material_stem),
                texture.texture_name_score(geo_stem, material_stem),
            )
            if extension == ".mta" and score >= 0.48:
                material_scores.append((len(overlap), score, material_stem))

        material_scores.sort(reverse=True)
        hints.extend(stem for _overlap, _score, stem in material_scores[:2])
        resolved = _unique_hint(hints)
        texture_hint_cache[cache_key] = resolved
        return resolved

    geo_cache: Dict[Tuple[str, Tuple[str, ...], Tuple[int, ...]], List[bpy.types.Object]] = {}
    imported_geo = 0
    unresolved_geo = 0
    created_empties = 0
    limit = object_limit if object_limit > 0 else world.object_count

    lines.append("objects:")
    for ref in world.object_refs[:limit]:
        entry = resolved_objects.get(ref.index)
        object_path = entry.path if entry else ""
        object_name = os.path.basename(object_path) if object_path else ref.key_hex
        vii_path = obj_format.find_sidecar(object_path, ".vii") if object_path else None
        mtn_path = obj_format.find_sidecar(object_path, ".mtn") if object_path else None
        vii_keys: Tuple[int, ...] = ()
        mtn_keys: Tuple[int, ...] = ()
        if vii_path:
            try:
                vii_keys = obj_format.parse_sidecar_file(vii_path).resource_keys
            except Exception:
                vii_keys = ()
        if mtn_path:
            try:
                mtn_keys = obj_format.parse_sidecar_file(mtn_path).resource_keys
            except Exception:
                mtn_keys = ()
        vii_key_set = set(vii_keys)
        raw_geo_sidecar_keys = [key for key in mtn_keys if key not in vii_key_set] or list(mtn_keys)
        material_sidecar_keys = [key for key in raw_geo_sidecar_keys if key in material_link_keys]
        geo_sidecar_keys = list(raw_geo_sidecar_keys)
        if not geo_sidecar_keys and vii_keys:
            for visual_key in vii_keys:
                geo_sidecar_keys.extend(visual_geo_keys.get(visual_key, ()))
        geo_path, geo_match = (
            wor_format.resolve_geo_for_object_path_info(
                object_path,
                geo_name_index,
                geo_key_index,
                geo_sidecar_keys,
            )
            if object_path
            else (None, "")
        )

        game_object = None
        matrix = Matrix.Identity(4)
        object_error = ""
        if object_path:
            try:
                game_object = obj_format.parse_obj_file(object_path)
                matrix = object_matrix(game_object)
            except Exception as exc:
                object_error = str(exc)

        should_create_empty = create_ref_empties or bool(import_meshes and geo_path)
        empty = None
        if should_create_empty and created_empties < max(0, empty_limit if not import_meshes else limit):
            empty_name = os.path.splitext(object_name)[0]
            empty = bpy.data.objects.new(empty_name, None)
            empty.empty_display_type = "PLAIN_AXES"
            empty.empty_display_size = 0.25
            empty.parent = root
            empty.matrix_world = matrix
            empty["wor_key"] = ref.key_hex
            empty["wor_ref_index"] = ref.index
            empty["wor_flags"] = f"0x{ref.type_flags:08X}"
            empty["wor_extra"] = f"0x{ref.extra:08X}"
            empty["wor_object_path"] = object_path
            empty["wor_geo_path"] = geo_path or ""
            empty["wor_geo_match"] = geo_match
            empty["wor_vii_path"] = vii_path or ""
            empty["wor_mtn_path"] = mtn_path or ""
            if game_object:
                empty["obj_translation"] = ",".join(f"{value:.6g}" for value in game_object.translation)
                empty["obj_matrix_offset"] = f"0x{game_object.matrix.offset:x}"
                empty["obj_translation_offset"] = f"0x{game_object.translation_offset:x}"
                empty["obj_keys"] = ",".join(game_object.key_hex[:12])
            if vii_keys:
                empty["vii_keys"] = ",".join(f"{key & 0xFFFFFFFF:08X}" for key in vii_keys[:8])
            if mtn_keys:
                empty["mtn_keys"] = ",".join(f"{key & 0xFFFFFFFF:08X}" for key in mtn_keys[:8])
            if material_sidecar_keys:
                empty["mtn_material_keys"] = ",".join(f"{key & 0xFFFFFFFF:08X}" for key in material_sidecar_keys[:8])
            collection.objects.link(empty)
            created_empties += 1

        if import_meshes and geo_path and empty is not None:
            texture_hints = texture_hints_for_object(object_path, geo_path) if resolve_textures else ()
            material_key_hints = tuple((key - 1) & 0xFFFFFFFF for key in material_sidecar_keys)
            geo_cache_key = (geo_path, texture_hints, material_key_hints)
            if geo_cache_key in geo_cache:
                objects = []
                for template in geo_cache[geo_cache_key]:
                    duplicate = template.copy()
                    duplicate.data = template.data
                    duplicate.animation_data_clear()
                    duplicate.name = f"{empty.name}_{template.name}"
                    collection.objects.link(duplicate)
                    objects.append(duplicate)
            else:
                objects = import_geo(
                    geo_path,
                    scale,
                    flip_uv_v,
                    resolve_textures,
                    convert_tdt_textures,
                    texture_alpha_mode,
                    None,
                    False,
                    texture_hints,
                    material_key_hints,
                )
                for obj in objects:
                    link_to_collection(obj)
                geo_cache[geo_cache_key] = list(objects)

            for obj in objects:
                obj.parent = empty
                obj.matrix_parent_inverse = Matrix.Identity(4)
                obj.matrix_basis = Matrix.Identity(4)
                obj["wor_key"] = ref.key_hex
                obj["wor_ref_index"] = ref.index
                obj["wor_object_path"] = object_path
                obj["wor_geo_path"] = geo_path
                obj["wor_geo_match"] = geo_match
            imported_geo += 1
        elif import_meshes and object_path and not geo_path:
            unresolved_geo += 1

        target = object_name
        if geo_path:
            target += f" -> {os.path.basename(geo_path)} ({geo_match})"
        elif object_path:
            target += " -> no geo match"
        else:
            target = "unresolved object"
        flags = ""
        if ref.has_metadata:
            flags = f" flags=0x{ref.type_flags:08X} extra=0x{ref.extra:08X}"
        if object_error:
            flags += f" obj_error={object_error}"
        lines.append(f"- [{ref.index:03d}] {ref.key_hex} -> {target}{flags}")

    if len(world.object_refs) > limit:
        lines.append(f"... {len(world.object_refs) - limit:,} more refs skipped by import limit")

    root["wor_created_empties"] = created_empties
    root["wor_imported_geo_instances"] = imported_geo
    root["wor_unresolved_geo"] = unresolved_geo
    root["wor_unique_geo_imports"] = len(geo_cache)
    if orient_upright:
        # World GEOs are Z-up in file space. The standalone GEO importer rotates
        # vertices for Blender, so apply the opposite only on the world root.
        root.rotation_euler[0] = -math.pi / 2.0
        root["wor_root_rotation"] = "x -90"

    text = bpy.data.texts.new(f"{base_name}_wor_refs")
    text.write("\n".join(lines))

    _set_active_object(root)
    return root


def _bfz_cache_base_dir() -> str:
    blender_temp = getattr(bpy.app, "tempdir", "") or ""
    return blender_temp if blender_temp else tempfile.gettempdir()


def _duplicate_name_group_count(entries: Sequence[bfz_archive.BfzEntry]) -> int:
    counts: Dict[str, int] = {}
    for entry in entries:
        key = bfz_archive.normalized_archive_path(entry.name).lower()
        counts[key] = counts.get(key, 0) + 1
    return sum(1 for count in counts.values() if count > 1)


def import_bfz_world_archive(
    filepath: str,
    create_ref_empties: bool = False,
    empty_limit: int = 250,
    import_meshes: bool = True,
    object_limit: int = 0,
    orient_upright: bool = True,
    scale: float = 1.0,
    flip_uv_v: bool = True,
    resolve_textures: bool = True,
    convert_tdt_textures: bool = True,
    texture_alpha_mode: str = "opaque",
    refresh_cache: bool = False,
) -> List[bpy.types.Object]:
    archive = bfz_archive.BfzArchive(filepath)
    archive.parse(decompress=False)
    cache_dir = bfz_archive.archive_cache_dir(filepath, _bfz_cache_base_dir())
    manifest_path, exported_paths = archive.export_all(cache_dir, refresh=refresh_cache)
    wor_entries = archive.wor_entries()
    if not wor_entries:
        raise ValueError("BFZ archive does not contain a .wor file")

    roots: List[bpy.types.Object] = []
    for entry in wor_entries:
        wor_path = os.path.join(cache_dir, exported_paths[entry.index])
        root = import_wor_refs(
            wor_path,
            create_ref_empties=create_ref_empties,
            empty_limit=empty_limit,
            import_meshes=import_meshes,
            object_limit=object_limit,
            orient_upright=orient_upright,
            scale=scale,
            flip_uv_v=flip_uv_v,
            resolve_textures=resolve_textures,
            convert_tdt_textures=convert_tdt_textures,
            texture_alpha_mode=texture_alpha_mode,
            source_mode="bfz-archive-cache",
            source_archive_path=filepath,
            deprecated_folder=False,
        )
        root["bfz_path"] = os.path.abspath(filepath)
        root["bfz_cache_dir"] = cache_dir
        root["bfz_manifest"] = manifest_path
        root["bfz_file_count"] = len(archive.file_entries)
        root["bfz_duplicate_name_groups"] = _duplicate_name_group_count(archive.file_entries)
        root["bfz_lzo_backend"] = archive.lzo_backend
        root["bfz_wor_archive_path"] = bfz_archive.normalized_archive_path(entry.name)
        root["bfz_wor_export_path"] = exported_paths[entry.index]
        roots.append(root)

    return roots
