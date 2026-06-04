from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import bpy

from . import bfz_archive, game_index, material_format, resource_index, texture
from .blender_import import import_geo, import_skn
from .texture import TextureResolver


CHARACTER_GEO_KINDS = ("head", "fullbody", "upbody", "lowbody", "arms", "left_eye", "right_eye", "accessory")


@dataclass(frozen=True)
class CharacterArchiveSummary:
    path: str
    name: str
    first_world_name: str
    skeleton_name: str
    geo_count: int
    head_count: int
    fullbody_count: int
    upbody_count: int
    lowbody_count: int
    arms_count: int
    eye_count: int
    accessory_count: int
    texture_count: int
    profile_summary: str = ""
    error: str = ""


@dataclass(frozen=True)
class CharacterPartProfile:
    species: str = ""
    sex: str = ""
    body_type: str = ""
    part_number: int = -1
    skin: str = ""

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.species, self.sex, self.body_type)

    @property
    def short_label(self) -> str:
        pieces = []
        if self.species:
            pieces.append("human" if self.species == "human" else "zombie")
        if self.sex:
            pieces.append("male" if self.sex == "male" else "female")
        if self.body_type:
            pieces.append(
                {
                    "regular": "reg",
                    "fat": "fat",
                    "thin": "thin",
                }.get(self.body_type, self.body_type)
            )
        return " ".join(pieces) or "unknown"


def _clean_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    stem = re.sub(r"__variant_[0-9]+$", "", stem, flags=re.IGNORECASE)
    return stem


def _natural_key(text: str) -> Tuple[object, ...]:
    parts = re.split(r"([0-9]+)", text.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _skeleton_score(name: str) -> Tuple[int, Tuple[object, ...]]:
    lower = os.path.basename(name).lower()
    score = 0
    if "skelet" in lower or "skeleton" in lower:
        score += 100
    if re.search(r"(?:^|_)h_[mf]_|(?:^|_)z_[mf]_", lower):
        score += 20
    if "cam" in lower or lower.startswith("gpe_"):
        score -= 80
    return (-score, _natural_key(lower))


def _geo_kind(name: str) -> str:
    lower = _clean_stem(name).lower()
    if "lefteye" in lower or "left_eye" in lower:
        return "left_eye"
    if "righteye" in lower or "right_eye" in lower:
        return "right_eye"
    if "fullbody" in lower:
        return "fullbody"
    if "upbody" in lower:
        return "upbody"
    if "lobody" in lower or "lowbody" in lower:
        return "lowbody"
    if "arms" in lower:
        return "arms"
    if "head" in lower and lower.startswith("ch"):
        return "head"
    if "accs" in lower or any(token in lower for token in ("backpack", "helmet", "hat", "hair", "mask", "glasses")):
        return "accessory"
    return ""


def _normal_body_type(token: str) -> str:
    lower = token.lower()
    if lower in {"reg", "regular"}:
        return "regular"
    if lower in {"fat", "thk", "thick"}:
        return "fat"
    if lower in {"thn", "thin"}:
        return "thin"
    return lower


def _part_number_from_name(name: str, kind: str) -> int:
    stem = _clean_stem(name)
    patterns = {
        "head": r"head([0-9]+)",
        "fullbody": r"fullbody([0-9]+)",
        "upbody": r"upbody([0-9]+)",
        "lowbody": r"(?:lo|low)body([0-9]+)",
        "arms": r"arms([0-9]+)",
    }
    pattern = patterns.get(kind)
    if not pattern:
        return -1
    match = re.search(pattern, stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _profile_for_geo(name: str, kind: str = "") -> CharacterPartProfile:
    stem = _clean_stem(name)
    lower = stem.lower()

    species = ""
    sex = ""
    body_type = ""
    match = re.search(
        r"(?:^|_)(?P<species>[hz])_(?P<sex>[mf])_(?P<body>reg|regular|fat|thk|thick|thn|thin)(?:_|$)",
        lower,
        flags=re.IGNORECASE,
    )
    if match:
        species = "human" if match.group("species").lower() == "h" else "zombie"
        sex = "male" if match.group("sex").lower() == "m" else "female"
        body_type = _normal_body_type(match.group("body"))

    skin = ""
    skin_match = re.search(r"_ca_(h|z[0-9]*)", lower, flags=re.IGNORECASE)
    if skin_match:
        skin = "human" if skin_match.group(1).lower() == "h" else "zombie"
        if not species:
            species = skin

    if not sex and re.search(r"(?:^|_)chcom_f_", lower):
        sex = "female"
    elif not sex and re.search(r"(?:^|_)chcom_m_", lower):
        sex = "male"

    if not kind:
        kind = _geo_kind(name)
    return CharacterPartProfile(
        species=species,
        sex=sex,
        body_type=body_type,
        part_number=_part_number_from_name(name, kind),
        skin=skin,
    )


def _is_character_geo(name: str) -> bool:
    lower = _clean_stem(name).lower()
    if _geo_kind(name):
        return True
    return lower.startswith(("chgen_", "chcom_"))


def _skin_tag_for_name(name: str, texture_type: str) -> str:
    if texture_type == "human":
        return "H"
    if texture_type == "zombie":
        return "Z1"
    lower = _clean_stem(name).lower()
    return "Z1" if re.search(r"(?:^|_)z(?:_|[0-9])", lower) else "H"


def _part_code(name: str, kind: str) -> str:
    stem = _clean_stem(name)
    patterns = {
        "head": r"head([0-9]+)",
        "fullbody": r"fullbody([0-9]+)",
        "upbody": r"upbody([0-9]+)",
        "lowbody": r"(?:lo|low)body([0-9]+)",
        "arms": r"arms([0-9]+)",
    }
    pattern = patterns.get(kind)
    if not pattern:
        return ""
    match = re.search(pattern, stem, flags=re.IGNORECASE)
    if not match:
        return ""
    if kind == "lowbody":
        prefix = "LoBody"
    elif kind == "fullbody":
        prefix = "FullBody"
    else:
        prefix = kind[0].upper() + kind[1:]
    return f"{prefix}{match.group(1)}"


def _texture_group_stems(resources: Sequence[resource_index.ResourceFile]) -> Tuple[str, ...]:
    result: List[str] = []
    seen: set[str] = set()
    for resource in resources:
        if resource.extension not in {".tex", ".tdt", ".png"}:
            continue
        stem = texture.texture_stem_from_path(resource.name)
        group, _role = texture.split_texture_role(stem)
        key = group.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(group)
    return tuple(result)


def _texture_hints_for_geo(
    geo_name: str,
    kind: str,
    texture_groups: Sequence[str],
    texture_type: str,
    selected_profile: CharacterPartProfile,
) -> Tuple[str, ...]:
    hints: List[str] = []
    stem = _clean_stem(geo_name)
    skin_tag = _skin_tag_for_name(geo_name, texture_type)
    code = _part_code(geo_name, kind)
    if kind == "head" and code:
        number = re.search(r"([0-9]+)$", code)
        if number:
            index = number.group(1)
            hints.extend(
                (
                    f"ChCom_M_Head{index}_Skin_CA_{skin_tag}__D",
                    f"ChCom_M_Head{index}_Skin_CA_{skin_tag}",
                    f"ChCom_M_Head{index}_Skin_CA_{skin_tag}_",
                    f"ChCom_M_head{index}_Skin_CA_{skin_tag}__D",
                    f"ChCom_M_head{index}_Skin_CA_{skin_tag}",
                    f"ChCom_M_head{index}_Skin_CA_{skin_tag}_",
                    f"ChCom_M_Head{index}_Skin_AF_{skin_tag}__D",
                    f"ChCom_M_Head{index}_Skin_CA_{skin_tag}_Thin",
                    f"ChCom_M_Head{index}_Skin_CA_{skin_tag}_Thin_D",
                )
            )
    elif kind == "arms":
        sex_prefix = "F" if selected_profile.sex == "female" else "M"
        other_prefix = "M" if sex_prefix == "F" else "F"
        hints.extend(
            (
                f"ChCom_{sex_prefix}_Arms00_Misc_Drt1_FPS_D",
                f"ChCom_{sex_prefix}_Arms00_Misc_Drt1_FPS",
                f"ChCom_{other_prefix}_Arms00_Misc_Drt1_FPS_D",
                f"ChCom_M_Body00_Skin_CA_{skin_tag}__D",
                f"ChCom_M_Body00_Skin_CA_{skin_tag}",
                f"ChCom_M_Body00_Skin_CA_{skin_tag}_",
            )
        )
    elif kind in {"fullbody", "upbody", "lowbody"}:
        hints.extend(
            (
                f"ChCom_M_Body00_Skin_CA_{skin_tag}__D",
                f"ChCom_M_Body00_Skin_CA_{skin_tag}",
                f"ChCom_M_Body00_Skin_CA_{skin_tag}_",
            )
        )
    elif kind in {"left_eye", "right_eye"}:
        hints.extend(("ChCom_Com_Head00_Orga_CA_H_Eye", "ChCom_Com_Head00_Orga_CA_H_EyeCornea"))

    lower_code = code.lower()
    lower_stem = stem.lower()
    for group in texture_groups:
        group_lower = group.lower()
        if lower_code and lower_code in group_lower:
            hints.append(group)
        elif kind in {"left_eye", "right_eye"} and "eye" in group_lower:
            hints.append(group)
        elif kind == "accessory" and any(token in lower_stem and token in group_lower for token in ("backpack", "hair", "helmet", "mask")):
            hints.append(group)

    hints.append(stem)
    return tuple(dict.fromkeys(hint for hint in hints if hint))


def _profile_summary_for_parts(parts: Sequence[Tuple[str, CharacterPartProfile]]) -> str:
    counts: Dict[Tuple[str, str, str], set[str]] = {}
    for kind, profile in parts:
        if not any(profile.key):
            continue
        counts.setdefault(profile.key, set()).add(kind)
    if not counts:
        return ""

    labels = []
    for key, kinds in sorted(counts.items(), key=lambda item: (-len(item[1]), item[0])):
        profile = CharacterPartProfile(*key)
        label = profile.short_label
        if kinds:
            label = f"{label} ({'/'.join(sorted(kinds))})"
        labels.append(label)
    return ", ".join(labels[:4])


def _archive_resources(path: str) -> Tuple[List[bfz_archive.BfzEntry], Dict[int, str], str]:
    archive = bfz_archive.BfzArchive(path)
    archive.parse(decompress=False)
    return archive.file_entries, archive.export_path_map(), ""


def summarize_character_archive(path: str) -> CharacterArchiveSummary | None:
    name = os.path.basename(path)
    try:
        entries, exported_paths, _error = _archive_resources(path)
    except Exception as exc:
        return CharacterArchiveSummary(
            path=path,
            name=name,
            first_world_name="",
            skeleton_name="",
            geo_count=0,
            head_count=0,
            fullbody_count=0,
            upbody_count=0,
            lowbody_count=0,
            arms_count=0,
            eye_count=0,
            accessory_count=0,
            texture_count=0,
            error=str(exc),
        )

    skeletons = []
    worlds = []
    kinds = {kind: 0 for kind in CHARACTER_GEO_KINDS}
    profiled_parts: List[Tuple[str, CharacterPartProfile]] = []
    texture_count = 0
    geo_count = 0
    for entry in entries:
        path_name = exported_paths.get(entry.index, bfz_archive.normalized_archive_path(entry.name))
        if entry.extension == ".wor":
            worlds.append(path_name)
        elif entry.extension == ".skn":
            skeletons.append(path_name)
        elif entry.extension == ".geo" and _is_character_geo(path_name):
            kind = _geo_kind(path_name)
            if kind:
                kinds[kind] += 1
                profiled_parts.append((kind, _profile_for_geo(path_name, kind)))
            geo_count += 1
        elif entry.extension in {".tex", ".tdt", ".png"}:
            texture_count += 1

    if not skeletons or geo_count <= 0:
        return None

    skeleton_name = sorted(skeletons, key=_skeleton_score)[0]
    return CharacterArchiveSummary(
        path=path,
        name=name,
        first_world_name=os.path.basename(sorted(worlds, key=_natural_key)[0]) if worlds else "",
        skeleton_name=os.path.basename(skeleton_name),
        geo_count=geo_count,
        head_count=kinds["head"],
        fullbody_count=kinds["fullbody"],
        upbody_count=kinds["upbody"],
        lowbody_count=kinds["lowbody"],
        arms_count=kinds["arms"],
        eye_count=kinds["left_eye"] + kinds["right_eye"],
        accessory_count=kinds["accessory"],
        texture_count=texture_count,
        profile_summary=_profile_summary_for_parts(profiled_parts),
    )


def scan_character_archives(game_dir: str) -> List[CharacterArchiveSummary]:
    data_dir = game_index.data_dir_for_game_dir(game_dir)
    if not data_dir:
        raise ValueError("pick the ZOMBI folder or its Data folder")

    summaries = []
    for path in game_index.iter_bfz_paths(data_dir):
        summary = summarize_character_archive(path)
        if summary is not None:
            summaries.append(summary)
    return sorted(summaries, key=lambda item: (item.error != "", item.name.lower()))


def _build_character_index(
    game_dir: str,
    archive_path: str,
    include_common: bool,
) -> resource_index.GameResourceIndex:
    index = resource_index.GameResourceIndex()
    index.mount_archive(archive_path, "Character", 0)
    data_dir = game_index.data_dir_for_game_dir(game_dir)
    if include_common and data_dir:
        priority = 20
        for path, kind, _old_priority in resource_index.iter_startup_archives(data_dir, include_common=True):
            if os.path.abspath(path) in index.archives:
                continue
            index.mount_archive(path, kind, priority)
            priority += 1
    return index


def _resources_by_kind(resources: Iterable[resource_index.ResourceFile]) -> Dict[str, List[resource_index.ResourceFile]]:
    result = {kind: [] for kind in CHARACTER_GEO_KINDS}
    for resource in resources:
        kind = _geo_kind(resource.name)
        if kind:
            result[kind].append(resource)
    for values in result.values():
        values.sort(key=lambda item: _natural_key(item.name))
    return result


def _resource_sort_key(resource: resource_index.ResourceFile) -> Tuple[int, str, int]:
    return (resource.archive_priority, resource.archive_name.lower(), resource.entry_index)


def _resource_stem(resource: resource_index.ResourceFile) -> str:
    stem = os.path.splitext(os.path.basename(resource.name))[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    return stem


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


def _pick(values: Sequence[resource_index.ResourceFile], index: int, all_variants: bool = False) -> List[resource_index.ResourceFile]:
    if all_variants:
        return list(values)
    if not values:
        return []
    return [values[max(0, min(index, len(values) - 1))]]


def _profile_matches_requested(
    profile: CharacterPartProfile,
    *,
    species: str,
    sex: str,
    body_type: str,
) -> bool:
    if species != "auto" and profile.species and profile.species != species:
        return False
    if sex != "auto" and profile.sex and profile.sex != sex:
        return False
    if body_type != "auto" and profile.body_type and profile.body_type != body_type:
        return False
    return True


def _choose_character_profile(
    geo_by_kind: Dict[str, List[resource_index.ResourceFile]],
    *,
    species: str,
    sex: str,
    body_type: str,
) -> CharacterPartProfile:
    scores: Dict[Tuple[str, str, str], int] = {}
    for kind, weight in (("head", 70), ("fullbody", 220), ("upbody", 110), ("lowbody", 110), ("arms", 35)):
        seen_for_kind: set[Tuple[str, str, str]] = set()
        for resource in geo_by_kind.get(kind, []):
            profile = _profile_for_geo(resource.name, kind)
            if not any(profile.key) or not _profile_matches_requested(profile, species=species, sex=sex, body_type=body_type):
                continue
            if profile.key in seen_for_kind:
                continue
            seen_for_kind.add(profile.key)
            scores[profile.key] = scores.get(profile.key, 0) + weight
            if profile.body_type == "regular":
                scores[profile.key] += 8
            if profile.species == "human":
                scores[profile.key] += 3

    if scores:
        key = sorted(scores.items(), key=lambda item: (-item[1], _natural_key(" ".join(item[0]))))[0][0]
        return CharacterPartProfile(*key)

    return CharacterPartProfile(
        species="" if species == "auto" else species,
        sex="" if sex == "auto" else sex,
        body_type="" if body_type == "auto" else body_type,
    )


def _profile_score(
    resource: resource_index.ResourceFile,
    kind: str,
    target: CharacterPartProfile,
    *,
    requested_species: str,
    requested_sex: str,
    requested_body_type: str,
) -> Tuple[int, int, Tuple[object, ...]]:
    profile = _profile_for_geo(resource.name, kind)
    score = 0
    if target.species and profile.species == target.species:
        score += 80
    elif target.species and profile.species and profile.species != target.species:
        score -= 140
    elif requested_species != "auto" and profile.species and profile.species != requested_species:
        score -= 1000
    if target.sex and profile.sex == target.sex:
        score += 120
    elif requested_sex != "auto" and profile.sex and profile.sex != requested_sex:
        score -= 1000
    elif target.sex and profile.sex and profile.sex != target.sex:
        score -= 160
    if target.body_type and profile.body_type == target.body_type:
        score += 90
    elif requested_body_type != "auto" and profile.body_type and profile.body_type != requested_body_type:
        score -= 900
    elif target.body_type and profile.body_type and profile.body_type != target.body_type:
        score -= 130
    if profile.body_type == "regular":
        score += 8
    if profile.part_number >= 0:
        score -= profile.part_number
    return (-score, max(profile.part_number, 9999), _natural_key(resource.name))


def _compatible_with_profile(resource: resource_index.ResourceFile, kind: str, target: CharacterPartProfile) -> bool:
    profile = _profile_for_geo(resource.name, kind)
    if target.species and profile.species and profile.species != target.species:
        return False
    if target.sex and profile.sex and profile.sex != target.sex:
        return False
    if target.body_type and profile.body_type and profile.body_type != target.body_type:
        return False
    return True


def _pick_by_profile(
    values: Sequence[resource_index.ResourceFile],
    kind: str,
    target: CharacterPartProfile,
    variant_index: int,
    all_variants: bool,
    *,
    requested_species: str,
    requested_sex: str,
    requested_body_type: str,
) -> List[resource_index.ResourceFile]:
    if not values:
        return []
    compatible_values = [resource for resource in values if _compatible_with_profile(resource, kind, target)]
    if compatible_values:
        values = compatible_values
    elif any(target.key):
        return []
    sorted_values = sorted(
        values,
        key=lambda resource: _profile_score(
            resource,
            kind,
            target,
            requested_species=requested_species,
            requested_sex=requested_sex,
            requested_body_type=requested_body_type,
        ),
    )
    if all_variants:
        return sorted_values
    index = max(0, min(variant_index, len(sorted_values) - 1))
    return [sorted_values[index]]


def _link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if not any(existing == obj for existing in collection.objects):
        collection.objects.link(obj)
    for user_collection in list(obj.users_collection):
        if user_collection != collection:
            user_collection.objects.unlink(obj)


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


def _first_bone(armature: bpy.types.Object, candidates: Sequence[str]) -> str:
    by_lower = {bone.name.lower(): bone.name for bone in armature.data.bones}
    for candidate in candidates:
        name = by_lower.get(candidate.lower())
        if name:
            return name
    return ""


def _object_center_in_armature_space(obj: bpy.types.Object, armature: bpy.types.Object):
    try:
        from mathutils import Vector
    except Exception:
        return None

    if obj.type != "MESH" or not obj.bound_box:
        return None
    center = Vector((0.0, 0.0, 0.0))
    for corner in obj.bound_box:
        center += obj.matrix_world @ Vector(corner)
    center /= 8.0
    return armature.matrix_world.inverted() @ center


def _nearest_bone(armature: bpy.types.Object, obj: bpy.types.Object, candidates: Sequence[str]) -> str:
    center = _object_center_in_armature_space(obj, armature)
    if center is None:
        return _first_bone(armature, candidates)

    by_lower = {bone.name.lower(): bone for bone in armature.data.bones}
    best_name = ""
    best_distance = None
    for candidate in candidates:
        bone = by_lower.get(candidate.lower())
        if bone is None:
            continue
        distance = (bone.head_local - center).length
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_name = bone.name
    return best_name


def _clear_vertex_groups(obj: bpy.types.Object) -> None:
    while len(obj.vertex_groups) > 0:
        obj.vertex_groups.remove(obj.vertex_groups[0])


def _rigid_bind_mesh(obj: bpy.types.Object, armature: bpy.types.Object, bone_name: str, replace: bool) -> bool:
    if obj.type != "MESH" or not bone_name or len(obj.data.vertices) == 0:
        return False
    if replace:
        _clear_vertex_groups(obj)

    group = obj.vertex_groups.get(bone_name) or obj.vertex_groups.new(name=bone_name)
    group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")

    modifier = obj.modifiers.get("Armature") or obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = armature
    if hasattr(modifier, "use_vertex_groups"):
        modifier.use_vertex_groups = True
    if hasattr(modifier, "use_bone_envelopes"):
        modifier.use_bone_envelopes = False

    obj.parent = armature
    try:
        obj.matrix_parent_inverse = armature.matrix_world.inverted()
    except Exception:
        pass
    obj["character_rigid_bind_bone"] = bone_name
    if "geo_armature_warning" in obj:
        del obj["geo_armature_warning"]
    return True


def _bind_character_static_part(obj: bpy.types.Object, armature: bpy.types.Object, kind: str, resource_name: str) -> None:
    lower = resource_name.lower()
    if kind == "left_eye":
        bone = _first_bone(armature, ("LeftEye_FACE", "LeftEye", "Head"))
        _rigid_bind_mesh(obj, armature, bone, replace=True)
        return
    if kind == "right_eye":
        bone = _first_bone(armature, ("RightEye_FACE", "RightEye", "Head"))
        _rigid_bind_mesh(obj, armature, bone, replace=True)
        return
    if kind != "accessory":
        return

    weighted_vertices = int(obj.data.get("geo_skin_weighted_vertices", 0))
    if len(obj.vertex_groups) > 0 and weighted_vertices > 0:
        return

    if any(token in lower for token in ("backpack", "bag", "pack")):
        candidates = ("Spine2", "Spine1", "Spine", "Hips")
    elif any(token in lower for token in ("helmet", "hat", "mask", "glasses", "hair")):
        candidates = ("Head", "HeadDeform1", "Neck")
    elif any(token in lower for token in ("left", "lhand")):
        candidates = ("LeftProp1", "LeftProp2", "LeftHand", "LeftForeArm")
    elif any(token in lower for token in ("right", "rhand", "hammer", "syringe", "weapon")):
        candidates = ("RightProp1", "RightProp2", "RightHand", "RightForeArm")
    else:
        candidates = ("Head", "Spine2", "Spine1", "RightProp1", "LeftProp1", "RightHand", "LeftHand", "Hips")

    bone = _nearest_bone(armature, obj, candidates)
    _rigid_bind_mesh(obj, armature, bone, replace=True)


def import_character_archive(
    game_dir: str,
    archive_path: str,
    *,
    include_common: bool = False,
    scale: float = 1.0,
    bone_length: float = 0.05,
    resolve_textures: bool = True,
    convert_tdt_textures: bool = True,
    texture_alpha_mode: str = "opaque",
    profile_species: str = "auto",
    profile_sex: str = "auto",
    profile_body_type: str = "auto",
    head_variant: int = 0,
    body_variant: int = 0,
    first_person: bool = False,
    part_override: bool = False,
    head_index: int = 0,
    upbody_index: int = 0,
    lowbody_index: int = 0,
    arms_index: int = 0,
    include_eyes: bool = True,
    include_accessories: bool = True,
    import_all_variants: bool = False,
) -> List[bpy.types.Object]:
    index = _build_character_index(game_dir, archive_path, include_common)
    archive_path = os.path.abspath(archive_path)
    archive_resources = index.entries_for_archive(archive_path)
    skeletons = [resource for resource in archive_resources if resource.extension == ".skn"]
    if not skeletons:
        raise ValueError("selected archive has no SKN skeleton")
    skeleton = sorted(skeletons, key=lambda item: _skeleton_score(item.name))[0]

    geo_by_kind = _resources_by_kind(resource for resource in archive_resources if resource.extension == ".geo")
    selected_geos: List[Tuple[str, resource_index.ResourceFile]] = []
    requested_species = "human" if first_person else profile_species
    selected_profile = _choose_character_profile(
        geo_by_kind,
        species=requested_species,
        sex=profile_sex,
        body_type=profile_body_type,
    )
    if first_person:
        if part_override:
            human_arms = [
                resource
                for resource in geo_by_kind["arms"]
                if _profile_for_geo(resource.name, "arms").species in {"", "human"}
            ]
            selected_geos.extend(("arms", resource) for resource in _pick(human_arms, arms_index, import_all_variants))
        else:
            arms = _pick_by_profile(
                geo_by_kind["arms"],
                "arms",
                selected_profile,
                body_variant,
                import_all_variants,
                requested_species="human",
                requested_sex=profile_sex,
                requested_body_type=profile_body_type,
            )
            if not arms:
                relaxed_profile = CharacterPartProfile(species="human")
                arms = _pick_by_profile(
                    geo_by_kind["arms"],
                    "arms",
                    relaxed_profile,
                    body_variant,
                    import_all_variants,
                    requested_species="human",
                    requested_sex="auto",
                    requested_body_type="auto",
                )
            selected_geos.extend(("arms", resource) for resource in arms)
    elif part_override:
        for kind, part_index in (
            ("head", head_index),
            ("fullbody", body_variant),
            ("upbody", upbody_index),
            ("lowbody", lowbody_index),
            ("arms", arms_index),
        ):
            selected_geos.extend((kind, resource) for resource in _pick(geo_by_kind[kind], part_index, import_all_variants))
    else:
        head_parts = _pick_by_profile(
            geo_by_kind["head"],
            "head",
            selected_profile,
            head_variant,
            import_all_variants,
            requested_species=requested_species,
            requested_sex=profile_sex,
            requested_body_type=profile_body_type,
        )
        fullbody_parts = _pick_by_profile(
            geo_by_kind["fullbody"],
            "fullbody",
            selected_profile,
            body_variant,
            import_all_variants,
            requested_species=requested_species,
            requested_sex=profile_sex,
            requested_body_type=profile_body_type,
        )
        upbody_parts = _pick_by_profile(
            geo_by_kind["upbody"],
            "upbody",
            selected_profile,
            body_variant,
            import_all_variants,
            requested_species=requested_species,
            requested_sex=profile_sex,
            requested_body_type=profile_body_type,
        )
        lowbody_parts = _pick_by_profile(
            geo_by_kind["lowbody"],
            "lowbody",
            selected_profile,
            body_variant,
            import_all_variants,
            requested_species=requested_species,
            requested_sex=profile_sex,
            requested_body_type=profile_body_type,
        )
        arms_parts = _pick_by_profile(
            geo_by_kind["arms"],
            "arms",
            selected_profile,
            body_variant,
            import_all_variants,
            requested_species=requested_species,
            requested_sex=profile_sex,
            requested_body_type=profile_body_type,
        )

        selected_geos.extend(("head", resource) for resource in head_parts)
        if fullbody_parts and (not upbody_parts or not lowbody_parts):
            selected_geos.extend(("fullbody", resource) for resource in fullbody_parts)
        else:
            selected_geos.extend(("upbody", resource) for resource in upbody_parts)
            selected_geos.extend(("lowbody", resource) for resource in lowbody_parts)
            selected_geos.extend(("arms", resource) for resource in arms_parts)
    if include_eyes and not first_person:
        selected_geos.extend(("left_eye", resource) for resource in geo_by_kind["left_eye"])
        selected_geos.extend(("right_eye", resource) for resource in geo_by_kind["right_eye"])
    if include_accessories and not first_person:
        selected_geos.extend(("accessory", resource) for resource in geo_by_kind["accessory"])
    if not selected_geos:
        raise ValueError("selected archive has no character GEO parts")

    base_name = os.path.splitext(os.path.basename(archive_path))[0]
    collection = bpy.data.collections.new(f"Character - {base_name}")
    bpy.context.scene.collection.children.link(collection)

    cache_root = resource_index.default_resource_cache_dir()
    skeleton_path = index.extract(skeleton, cache_root)
    texture_resources = [
        resource
        for resource in archive_resources
        if resource.extension in {".tex", ".tdt", ".png", ".mta", ".mat"}
    ]
    _texture_paths, search_dirs = _extract_resources(index, texture_resources, cache_root)
    texture_groups = _texture_group_stems(texture_resources)
    mta_stems_by_key = _build_mta_stems_by_mat_key(index, archive_resources)
    resolver = None
    if resolve_textures:
        resolver = TextureResolver(
            skeleton_path,
            convert_tdt_textures,
            texture_alpha_mode,
            search_dirs=search_dirs,
            mta_stems_by_key=mta_stems_by_key,
            cache_dir=os.path.join(cache_root, "converted_textures"),
        )

    imported: List[bpy.types.Object] = []
    armature = import_skn(skeleton_path, scale, bone_length)
    armature.name = f"{base_name}_Skeleton"
    armature.data.name = armature.name
    armature["character_archive"] = os.path.basename(archive_path)
    armature["character_profile"] = selected_profile.short_label
    armature["character_first_person"] = bool(first_person)
    _link_to_collection(armature, collection)
    imported.append(armature)

    texture_type = requested_species if requested_species in {"human", "zombie"} else selected_profile.species or "auto"
    for kind, resource in selected_geos:
        geo_path = index.extract(resource, cache_root)
        hints = _texture_hints_for_geo(resource.name, kind, texture_groups, texture_type, selected_profile)
        objects = import_geo(
            geo_path,
            scale,
            True,
            resolve_textures,
            convert_tdt_textures,
            texture_alpha_mode,
            armature,
            True,
            texture_query_hints=hints,
            texture_resolver=resolver,
            texture_search_dirs=search_dirs,
            mta_stems_by_key=mta_stems_by_key,
        )
        for obj in objects:
            obj["character_part"] = kind
            obj["character_resource"] = resource.normalized_path
            obj["character_profile"] = _profile_for_geo(resource.name, kind).short_label
            obj.name = f"{kind}_{obj.name}"
            _bind_character_static_part(obj, armature, kind, resource.name)
            _link_to_collection(obj, collection)
        imported.extend(objects)

    collection["character_archive"] = os.path.basename(archive_path)
    collection["character_skeleton"] = skeleton.name
    collection["character_geo_parts"] = len(selected_geos)
    collection["character_profile"] = selected_profile.short_label
    collection["character_first_person"] = bool(first_person)
    return imported
