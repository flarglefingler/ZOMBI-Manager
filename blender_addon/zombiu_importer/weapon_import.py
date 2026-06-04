from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import bpy
from mathutils import Matrix

from . import bfz_archive, game_index, material_format, resource_index
from .blender_import import import_geo, import_skn
from .texture import TextureResolver


WEAPON_WORLD_NAME = "_kit_PC_weapons.wor"
WEAPON_ARCHIVE_STEM = "wor_f6000aac"


@dataclass(frozen=True)
class WeaponArchiveScan:
    archive_path: str
    archive_name: str
    world_name: str
    weapons: Tuple["WeaponSummary", ...]
    error: str = ""


@dataclass(frozen=True)
class WeaponSummary:
    archive_path: str
    archive_name: str
    world_name: str
    label: str
    geo_names: Tuple[str, ...]
    skn_name: str = ""
    material_count: int = 0
    texture_count: int = 0
    recommended_bone: str = "RightHand"
    error: str = ""


def _clean_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    stem = re.sub(r"__variant_[0-9]+$", "", stem, flags=re.IGNORECASE)
    return stem


def _natural_key(text: str) -> Tuple[object, ...]:
    parts = re.split(r"([0-9]+)", text.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _display_label(name: str) -> str:
    stem = _clean_stem(name)
    mp_prefix = stem.lower().startswith("mp_")
    if stem.lower().startswith("autoturret_"):
        return "Auto Turret"
    stem = re.sub(r"^(?:m_|mp_|gpe_)", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^m[0-9]+_", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^w[0-9]+_?", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^weapon", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    stem = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", stem)
    stem = re.sub(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])", " ", stem)
    stem = re.sub(r"[_\\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    replacements = (
        ("HKUMP 45", "HK UMP45"),
        ("SA 80 A 2", "SA80A2"),
        ("AK 47", "AK47"),
        ("AW 50", "AW50"),
        ("MP 5", "MP5"),
        ("P 226", "P226"),
        ("M 4", "M4"),
        ("DBS", "Double Barrel Shotgun"),
    )
    for old, new in replacements:
        stem = stem.replace(old, new)
    if mp_prefix and not stem.upper().startswith("MP "):
        stem = "MP " + stem
    return stem or _clean_stem(name)


def _tokens(name: str) -> Tuple[str, ...]:
    stem = _clean_stem(name)
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", stem)
    stem = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).lower()
    stop = {
        "m",
        "mp",
        "gpe",
        "pc",
        "geo",
        "skn",
        "weapon",
        "wpn",
        "puppet",
        "first",
        "third",
        "entity",
        "w",
        "m01",
        "w01",
        "w02",
        "w03",
        "w04",
        "w05",
    }
    result: List[str] = []
    for token in stem.split("_"):
        for part in re.findall(r"[a-z]+|[0-9]+", token):
            if not part or part in stop:
                continue
            result.append(part)
    return tuple(result)


def _is_weapon_geo(name: str) -> bool:
    stem = _clean_stem(name)
    lower = stem.lower()
    if lower.startswith(("m_weapon", "m_w", "mp_weapon")):
        return True
    if lower.startswith("m01_cricketbat"):
        return True
    if lower in {"m_grenade", "molotov", "nailedbaseballbat", "shovel", "gpe_landmine"}:
        return True
    if lower.startswith("autoturret_"):
        return True
    return False


def _group_key(name: str) -> str:
    stem = _clean_stem(name).lower()
    if stem.startswith("autoturret_"):
        return "autoturret"
    return stem


def _recommended_bone(names: Sequence[str]) -> str:
    tokens = set()
    for name in names:
        tokens.update(_tokens(name))
    left_tokens = {"remington", "870", "770", "ak", "47"}
    if tokens & left_tokens and ("remington" in tokens or "ak" in tokens):
        return "LeftHand"
    return "RightHand"


def _weapon_sort_key(summary: WeaponSummary) -> Tuple[object, ...]:
    return _natural_key(summary.label)


def find_weapon_archive(game_dir: str) -> str:
    data_dir = game_index.data_dir_for_game_dir(game_dir)
    if not data_dir:
        raise ValueError("pick the ZOMBI folder or its Data folder")

    preferred = os.path.join(data_dir, "Wor_F6000AAC_0.lin.bfz")
    candidates = [preferred] if os.path.exists(preferred) else []
    candidates.extend(path for path in game_index.iter_bfz_paths(data_dir) if path not in candidates)

    fallback = ""
    for path in candidates:
        name = os.path.basename(path).lower()
        if WEAPON_ARCHIVE_STEM in name:
            fallback = path
        try:
            archive = bfz_archive.BfzArchive(path)
            archive.parse(decompress=False)
        except Exception:
            continue
        export_paths = archive.export_path_map()
        world_names = [
            os.path.basename(export_paths.get(entry.index, bfz_archive.normalized_archive_path(entry.name))).lower()
            for entry in archive.file_entries
            if entry.extension == ".wor"
        ]
        has_weapon_world = WEAPON_WORLD_NAME.lower() in world_names
        has_geo = any(entry.extension == ".geo" for entry in archive.file_entries)
        if has_weapon_world and has_geo:
            return path

    if fallback:
        return fallback
    raise ValueError("could not find _kit_PC_weapons.wor / Wor_F6000AAC")


def _build_weapon_index(game_dir: str, archive_path: str, include_common: bool) -> resource_index.GameResourceIndex:
    index = resource_index.GameResourceIndex()
    index.mount_archive(archive_path, "Weapons", 0)
    data_dir = game_index.data_dir_for_game_dir(game_dir)
    if include_common and data_dir:
        priority = 20
        for path, kind, _old_priority in resource_index.iter_startup_archives(data_dir, include_common=True):
            if os.path.abspath(path) in index.archives:
                continue
            index.mount_archive(path, kind, priority)
            priority += 1
    return index


def _resource_sort_key(resource: resource_index.ResourceFile) -> Tuple[int, str, int]:
    return (resource.archive_priority, resource.archive_name.lower(), resource.entry_index)


def _resource_stem(resource: resource_index.ResourceFile) -> str:
    return _clean_stem(resource.name)


def _match_skn(geo_names: Sequence[str], skn_resources: Sequence[resource_index.ResourceFile]) -> resource_index.ResourceFile | None:
    wanted = set()
    for name in geo_names:
        wanted.update(_tokens(name))
    if not wanted:
        return None

    best: Tuple[int, Tuple[object, ...], resource_index.ResourceFile] | None = None
    for resource in skn_resources:
        candidate_tokens = set(_tokens(resource.name))
        overlap = wanted & candidate_tokens
        score = sum(4 if token.isdigit() else 2 for token in overlap)
        if "bat" in wanted and "bat" in candidate_tokens:
            score += 3
        if not score:
            continue
        item = (-score, _natural_key(resource.name), resource)
        if best is None or item < best:
            best = item
    if best and -best[0] >= 2:
        return best[2]
    return None


def _build_mta_stems_by_mat_key(
    index: resource_index.GameResourceIndex,
    resources: Sequence[resource_index.ResourceFile],
) -> Dict[int, str]:
    mat_resources = sorted((resource for resource in resources if resource.extension == ".mat"), key=_resource_sort_key)
    mta_resources = sorted((resource for resource in resources if resource.extension == ".mta"), key=_resource_sort_key)

    material_keys: List[int] = []
    seen: set[int] = set()
    for resource in mat_resources:
        try:
            descriptor = material_format.parse_mat(index.read(resource), resource.name)
        except Exception:
            continue
        for key in descriptor.submaterial_keys:
            normalized = key & 0xFFFFFFFF
            if normalized in seen:
                continue
            seen.add(normalized)
            material_keys.append(normalized)

    mapping: Dict[int, str] = {}
    for index_value, key in enumerate(material_keys):
        if index_value >= len(mta_resources):
            break
        mapping[key] = _resource_stem(mta_resources[index_value])
    return mapping


def _extract_resources(
    index: resource_index.GameResourceIndex,
    resources: Iterable[resource_index.ResourceFile],
    cache_root: str,
) -> Tuple[List[str], List[str]]:
    paths = []
    dirs = []
    seen_dirs = set()
    for resource in resources:
        path = index.extract(resource, cache_root)
        paths.append(path)
        directory = os.path.dirname(os.path.abspath(path))
        if directory not in seen_dirs:
            seen_dirs.add(directory)
            dirs.append(directory)
    return paths, dirs


def scan_weapon_archive(game_dir: str) -> WeaponArchiveScan:
    archive_path = find_weapon_archive(game_dir)
    index = _build_weapon_index(game_dir, archive_path, include_common=False)
    archive_resources = index.entries_for_archive(archive_path)
    archive_name = os.path.basename(archive_path)
    world_names = [resource.name for resource in archive_resources if resource.extension == ".wor"]
    world_name = next((name for name in world_names if name.lower() == WEAPON_WORLD_NAME.lower()), world_names[0] if world_names else "")

    skn_resources = sorted((resource for resource in archive_resources if resource.extension == ".skn"), key=lambda item: _natural_key(item.name))
    grouped: Dict[str, List[resource_index.ResourceFile]] = {}
    for resource in archive_resources:
        if resource.extension != ".geo" or not _is_weapon_geo(resource.name):
            continue
        grouped.setdefault(_group_key(resource.name), []).append(resource)

    material_resources = [resource for resource in archive_resources if resource.extension in {".mat", ".mta"}]
    texture_resources = [resource for resource in archive_resources if resource.extension in {".tex", ".tdt", ".png"}]
    weapons: List[WeaponSummary] = []
    for resources in grouped.values():
        resources.sort(key=lambda item: _natural_key(item.name))
        geo_names = tuple(resource.name for resource in resources)
        skn = _match_skn(geo_names, skn_resources)
        label = _display_label(resources[0].name)
        weapons.append(
            WeaponSummary(
                archive_path=archive_path,
                archive_name=archive_name,
                world_name=world_name,
                label=label,
                geo_names=geo_names,
                skn_name=skn.name if skn else "",
                material_count=len(material_resources),
                texture_count=len(texture_resources),
                recommended_bone=_recommended_bone(geo_names),
            )
        )

    return WeaponArchiveScan(
        archive_path=archive_path,
        archive_name=archive_name,
        world_name=world_name,
        weapons=tuple(sorted(weapons, key=_weapon_sort_key)),
    )


def _link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if not any(existing == obj for existing in collection.objects):
        collection.objects.link(obj)
    for user_collection in list(obj.users_collection):
        if user_collection != collection:
            user_collection.objects.unlink(obj)


def _bone_world_matrix(armature: bpy.types.Object, bone_name: str) -> Matrix:
    return armature.matrix_world @ armature.data.bones[bone_name].matrix_local


def _find_resource_by_name(
    resources: Sequence[resource_index.ResourceFile],
    name: str,
    extension: str | None = None,
) -> resource_index.ResourceFile | None:
    wanted = name.lower()
    for resource in resources:
        if extension and resource.extension != extension:
            continue
        if resource.name.lower() == wanted:
            return resource
    return None


def _find_bone_name(armature: bpy.types.Object, requested: str, fallback: str) -> str:
    requested = requested.strip()
    if requested.lower() == "auto":
        requested = fallback
    candidates = [requested, fallback, "RightHand", "LeftHand", "RightProp1", "LeftProp1"]
    by_normalized = {
        re.sub(r"[^a-z0-9]", "", bone.name.lower()): bone.name
        for bone in armature.data.bones
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in by_normalized:
            return by_normalized[key]
    return ""


def _main_weapon_bone_name(weapon_armature: bpy.types.Object) -> str:
    by_index = {
        int(bone.get("skn_index", index)): bone.name
        for index, bone in enumerate(weapon_armature.data.bones)
    }
    for index in sorted(by_index):
        name = by_index[index]
        if name.lower() != "magicbox":
            return name
    if len(weapon_armature.data.bones) > 0:
        return weapon_armature.data.bones[0].name
    return ""


def _weapon_socket_bone_name(weapon_armature: bpy.types.Object, target_bone_name: str) -> str:
    wanted = re.sub(r"[^a-z0-9]", "", target_bone_name.lower())
    by_normalized = {
        re.sub(r"[^a-z0-9]", "", bone.name.lower()): bone.name
        for bone in weapon_armature.data.bones
    }
    for normalized, bone_name in by_normalized.items():
        if normalized == wanted or normalized.endswith(wanted):
            return bone_name

    target_lower = target_bone_name.lower()
    if "left" in target_lower:
        for normalized, bone_name in by_normalized.items():
            if normalized.endswith("lefthand") or "lefthand" in normalized:
                return bone_name
    if "right" in target_lower:
        for normalized, bone_name in by_normalized.items():
            if normalized.endswith("righthand") or "righthand" in normalized:
                return bone_name

    return _main_weapon_bone_name(weapon_armature)


def _snap_weapon_armature_to_bone(
    weapon_armature: bpy.types.Object,
    character_armature: bpy.types.Object,
    target_bone_name: str,
) -> str:
    socket_name = _weapon_socket_bone_name(weapon_armature, target_bone_name)
    target_world = _bone_world_matrix(character_armature, target_bone_name)
    if socket_name:
        socket_local = weapon_armature.data.bones[socket_name].matrix_local
        try:
            weapon_armature.matrix_world = target_world @ socket_local.inverted()
        except Exception:
            weapon_armature.matrix_world = target_world
    else:
        weapon_armature.matrix_world = target_world
    weapon_armature["weapon_append_socket_bone"] = socket_name
    return socket_name


def _parent_object_to_bone(obj: bpy.types.Object, armature: bpy.types.Object, bone_name: str) -> None:
    world = obj.matrix_world.copy()
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    try:
        obj.matrix_parent_inverse = Matrix.Identity(4)
    except Exception:
        pass
    obj.matrix_world = world


def _append_object_to_bone(
    obj: bpy.types.Object,
    armature: bpy.types.Object,
    requested_bone: str,
    fallback_bone: str,
) -> str:
    if armature is None or armature.type != "ARMATURE":
        raise ValueError("append needs an armature object")
    bone_name = _find_bone_name(armature, requested_bone, fallback_bone)
    if not bone_name:
        raise ValueError(f"could not find append bone '{requested_bone}'")

    if obj.type == "ARMATURE":
        socket_name = _snap_weapon_armature_to_bone(obj, armature, bone_name)
        obj["weapon_append_offset"] = f"weapon socket {socket_name or 'origin'}"
    else:
        obj.matrix_world = _bone_world_matrix(armature, bone_name)
        obj["weapon_append_offset"] = "object origin"

    _parent_object_to_bone(obj, armature, bone_name)
    obj["weapon_append_armature"] = armature.name
    obj["weapon_append_bone"] = bone_name
    return bone_name


def _append_unrigged_objects_to_bone(
    objects: Sequence[bpy.types.Object],
    armature: bpy.types.Object,
    requested_bone: str,
    fallback_bone: str,
) -> str:
    if armature is None or armature.type != "ARMATURE":
        raise ValueError("append needs an armature object")
    bone_name = _find_bone_name(armature, requested_bone, fallback_bone)
    if not bone_name:
        raise ValueError(f"could not find append bone '{requested_bone}'")
    if not objects:
        return bone_name

    target_world = _bone_world_matrix(armature, bone_name)
    try:
        pivot_inverse = objects[0].matrix_world.inverted()
    except Exception:
        pivot_inverse = Matrix.Identity(4)

    for obj in objects:
        obj.matrix_world = target_world @ pivot_inverse @ obj.matrix_world
        _parent_object_to_bone(obj, armature, bone_name)
        obj["weapon_append_armature"] = armature.name
        obj["weapon_append_bone"] = bone_name
        obj["weapon_append_offset"] = "group pivot"
    return bone_name


def import_weapon(
    game_dir: str,
    archive_path: str,
    geo_names: Sequence[str],
    *,
    skn_name: str = "",
    label: str = "Weapon",
    include_common: bool = False,
    scale: float = 1.0,
    resolve_textures: bool = True,
    convert_tdt_textures: bool = True,
    texture_alpha_mode: str = "opaque",
    append_armature: bpy.types.Object | None = None,
    append_bone: str = "Auto",
    recommended_bone: str = "RightHand",
) -> List[bpy.types.Object]:
    if not geo_names:
        raise ValueError("selected weapon has no GEO model")

    index = _build_weapon_index(game_dir, archive_path, include_common)
    archive_path = os.path.abspath(archive_path)
    archive_resources = index.entries_for_archive(archive_path)
    geo_resources = []
    for name in geo_names:
        resource = _find_resource_by_name(archive_resources, name, ".geo")
        if resource:
            geo_resources.append(resource)
    if not geo_resources:
        raise ValueError("selected weapon GEOs were not found in the archive")

    skn_resource = _find_resource_by_name(archive_resources, skn_name, ".skn") if skn_name else None
    base_name = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_") or "Weapon"
    collection = bpy.data.collections.new(f"Weapon - {label}")
    bpy.context.scene.collection.children.link(collection)
    collection["weapon_archive"] = os.path.basename(archive_path)
    collection["weapon_label"] = label
    collection["weapon_recommended_bone"] = recommended_bone

    cache_root = resource_index.default_resource_cache_dir()
    material_texture_resources = [
        resource
        for resource in index.resources
        if resource.extension in {".tex", ".tdt", ".png", ".mta", ".mat"}
    ]
    _texture_paths, search_dirs = _extract_resources(index, material_texture_resources, cache_root)
    mta_stems_by_key = _build_mta_stems_by_mat_key(index, material_texture_resources)
    resolver = None
    if resolve_textures:
        resolver = TextureResolver(
            index.extract(geo_resources[0], cache_root),
            convert_tdt_textures,
            texture_alpha_mode,
            search_dirs=search_dirs,
            mta_stems_by_key=mta_stems_by_key,
            cache_dir=os.path.join(cache_root, "converted_textures"),
        )

    imported: List[bpy.types.Object] = []
    weapon_armature = None
    if skn_resource:
        skn_path = index.extract(skn_resource, cache_root)
        weapon_armature = import_skn(skn_path, scale, 0.05)
        weapon_armature.name = f"{base_name}_Skeleton"
        weapon_armature.data.name = weapon_armature.name
        weapon_armature["weapon_archive"] = os.path.basename(archive_path)
        weapon_armature["weapon_skn"] = skn_resource.name
        _link_to_collection(weapon_armature, collection)
        imported.append(weapon_armature)

    for resource in geo_resources:
        geo_path = index.extract(resource, cache_root)
        hints = (_clean_stem(resource.name), label, *_tokens(resource.name))
        objects = import_geo(
            geo_path,
            scale,
            True,
            resolve_textures,
            convert_tdt_textures,
            texture_alpha_mode,
            weapon_armature,
            True,
            texture_query_hints=hints,
            texture_resolver=resolver,
            texture_search_dirs=search_dirs,
            mta_stems_by_key=mta_stems_by_key,
        )
        for obj in objects:
            obj["weapon_label"] = label
            obj["weapon_resource"] = resource.normalized_path
            obj.name = f"{base_name}_{obj.name}"
            _link_to_collection(obj, collection)
        imported.extend(objects)

    if append_armature is not None:
        if weapon_armature is not None:
            used_bone = _append_object_to_bone(weapon_armature, append_armature, append_bone, recommended_bone)
        else:
            append_targets = [obj for obj in imported if obj.parent is None]
            used_bone = _append_unrigged_objects_to_bone(
                append_targets,
                append_armature,
                append_bone,
                recommended_bone,
            )
        for obj in imported:
            obj["weapon_append_bone"] = used_bone
        collection["weapon_append_bone"] = used_bone

    collection["weapon_archive"] = os.path.basename(archive_path)
    collection["weapon_label"] = label
    collection["weapon_geo_count"] = len(geo_resources)
    collection["weapon_skn"] = skn_resource.name if skn_resource else ""
    return imported
