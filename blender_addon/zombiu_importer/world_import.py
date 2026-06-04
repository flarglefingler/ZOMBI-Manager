"""clean BFZ world import path built around mounted resource keys."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import bpy
from mathutils import Matrix, Vector

from . import material_format, obj_format, resource_index, texture, wor_format
from .blender_import import (
    GeoMaterialInfo,
    _is_world_material_prefix_match,
    _world_asset_stem,
    _world_asset_stem_variants,
    game_to_blender_matrix,
    game_to_blender_point,
    import_geo,
)
from .texture import TextureResolver


@dataclass
class ObjectGraph:
    ref: wor_format.WorObjectRef
    object_file: resource_index.ResourceFile | None
    object_data: obj_format.ObjFile | None
    sidecar_files: Tuple[resource_index.ResourceFile, ...]
    sidecar_keys: Tuple[int, ...]
    geo_sidecar_keys: Tuple[int, ...]
    material_sidecar_keys: Tuple[int, ...]
    geo_file: resource_index.ResourceFile | None = None
    geo_key: int | None = None
    geo_state: str = "missing"
    geo_candidates: int = 0
    obj_error: str = ""


@dataclass(frozen=True)
class GeoNameIndex:
    by_literal: Dict[str, Tuple[resource_index.ResourceFile, ...]]
    by_match: Dict[str, Tuple[resource_index.ResourceFile, ...]]
    by_model: Dict[str, Tuple[resource_index.ResourceFile, ...]]


@dataclass(frozen=True)
class WorldMaterialContext:
    search_dirs: Tuple[str, ...]
    mta_stems_by_key: Dict[int, str]
    material_link_keys: frozenset[int]
    mat_resources: Tuple[resource_index.ResourceFile, ...]
    mat_descriptors: Dict[Tuple[str, int], material_format.MatDescriptor]
    cache_dir: str


def _unique_resources(resources: Sequence[resource_index.ResourceFile]) -> List[resource_index.ResourceFile]:
    seen = set()
    result: List[resource_index.ResourceFile] = []
    for resource in resources:
        identity = (resource.archive_path, resource.entry_index)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(resource)
    return result


def _resource_sort_key(resource: resource_index.ResourceFile) -> Tuple[int, str, int]:
    return (resource.archive_priority, resource.archive_name.lower(), resource.entry_index)


def _resource_identity(resource: resource_index.ResourceFile) -> Tuple[str, int]:
    return (resource.archive_path, resource.entry_index)


def _resource_export_stem(resource: resource_index.ResourceFile) -> str:
    stem = os.path.splitext(resource.name)[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    return stem


def _safe_extract_resource(
    index: resource_index.GameResourceIndex,
    resource: resource_index.ResourceFile,
    cache_root: str,
    directories: set[str],
) -> str | None:
    try:
        path = index.extract(resource, cache_root)
    except Exception:
        return None
    directories.add(os.path.dirname(os.path.abspath(path)))
    return path


def _mat_descriptors_for_resources(
    index: resource_index.GameResourceIndex,
    mat_resources: Sequence[resource_index.ResourceFile],
) -> Dict[Tuple[str, int], material_format.MatDescriptor]:
    descriptors: Dict[Tuple[str, int], material_format.MatDescriptor] = {}
    for resource in mat_resources:
        try:
            descriptors[_resource_identity(resource)] = material_format.parse_mat(index.read(resource), resource.name)
        except Exception:
            continue
    return descriptors


def _build_mta_stems_by_mat_key(
    mat_resources: Sequence[resource_index.ResourceFile],
    mat_descriptors: Dict[Tuple[str, int], material_format.MatDescriptor],
    mta_resources: Sequence[resource_index.ResourceFile],
) -> Dict[int, str]:
    material_keys: List[int] = []
    seen: set[int] = set()
    for resource in mat_resources:
        descriptor = mat_descriptors.get(_resource_identity(resource))
        if not descriptor:
            continue
        for key in descriptor.submaterial_keys:
            normalized = key & 0xFFFFFFFF
            if normalized in seen:
                continue
            seen.add(normalized)
            material_keys.append(normalized)

    mapping: Dict[int, str] = {}
    for index, key in enumerate(material_keys):
        if index >= len(mta_resources):
            break
        mapping[key] = _resource_export_stem(mta_resources[index])
    return mapping


def _build_world_material_context(
    index: resource_index.GameResourceIndex,
    cache_root: str,
) -> WorldMaterialContext:
    mat_resources = tuple(sorted(index.resources_with_extensions((".mat",)), key=_resource_sort_key))
    mta_resources = tuple(sorted(index.resources_with_extensions((".mta",)), key=_resource_sort_key))
    tex_resources = tuple(sorted(index.resources_with_extensions((".tex",)), key=_resource_sort_key))
    tdt_resources = tuple(sorted(index.resources_with_extensions((".tdt",)), key=_resource_sort_key))

    mat_descriptors = _mat_descriptors_for_resources(index, mat_resources)
    mta_stems_by_key = _build_mta_stems_by_mat_key(mat_resources, mat_descriptors, mta_resources)
    material_link_keys = frozenset(((key + 1) & 0xFFFFFFFF) for key in mta_stems_by_key)

    directories: set[str] = set()
    for resource in (*mat_resources, *mta_resources, *tex_resources):
        _safe_extract_resource(index, resource, cache_root, directories)

    # TEX descriptors carry the internal TDT key, but the archive table key is
    # not always that value. Stage payloads by matching stem instead.
    needed_texture_stems = {
        texture.texture_stem_from_path(resource.name).lower()
        for resource in tex_resources
    }
    for resource in tdt_resources:
        if texture.texture_stem_from_path(resource.name).lower() not in needed_texture_stems:
            continue
        _safe_extract_resource(index, resource, cache_root, directories)

    cache_dir = os.path.join(cache_root, "converted_textures")
    os.makedirs(cache_dir, exist_ok=True)
    return WorldMaterialContext(
        search_dirs=tuple(sorted(directories, key=str.lower)),
        mta_stems_by_key=mta_stems_by_key,
        material_link_keys=material_link_keys,
        mat_resources=mat_resources,
        mat_descriptors=mat_descriptors,
        cache_dir=cache_dir,
    )


def _material_sidecar_keys(raw_keys: Sequence[int], material_link_keys: frozenset[int]) -> Tuple[int, ...]:
    result: List[int] = []
    seen: set[int] = set()
    for key in raw_keys:
        normalized = key & 0xFFFFFFFF
        if normalized not in material_link_keys:
            continue
        material_key = (normalized - 1) & 0xFFFFFFFF
        if material_key in seen:
            continue
        seen.add(material_key)
        result.append(material_key)
    return tuple(result)


def _material_info_for_geo_resource(
    geo_file: resource_index.ResourceFile,
    context: WorldMaterialContext | None,
) -> GeoMaterialInfo | None:
    if context is None:
        return None

    exact_stems = {stem.lower() for stem in _world_asset_stem_variants(geo_file.name)}
    geo_stem = _world_asset_stem(geo_file.name)
    archive_groups = (
        [resource for resource in context.mat_resources if resource.archive_path == geo_file.archive_path],
        [resource for resource in context.mat_resources if resource.archive_path != geo_file.archive_path],
    )

    for group_index, resources in enumerate(archive_groups):
        for resource in resources:
            mat_stem = _world_asset_stem(resource.name)
            if mat_stem.lower() not in exact_stems:
                continue
            descriptor = context.mat_descriptors.get(_resource_identity(resource))
            if not descriptor:
                continue
            method = "bfz-mat-exact" if group_index == 0 else "bfz-mat-external"
            return GeoMaterialInfo(path=resource.normalized_path, match_method=method, keys=descriptor.submaterial_keys)

    best: Tuple[int, int, str, resource_index.ResourceFile] | None = None
    for group_index, resources in enumerate(archive_groups):
        for resource in resources:
            mat_stem = _world_asset_stem(resource.name)
            if not _is_world_material_prefix_match(geo_stem, mat_stem):
                continue
            score = (len(mat_stem), 0 if group_index == 0 else -1, resource.normalized_path.lower(), resource)
            if best is None or score[:3] > best[:3]:
                best = score

    if best is None:
        return None
    resource = best[3]
    descriptor = context.mat_descriptors.get(_resource_identity(resource))
    if not descriptor:
        return None
    return GeoMaterialInfo(path=resource.normalized_path, match_method="bfz-mat-shared-prefix", keys=descriptor.submaterial_keys)


def _object_matrix(game_object: obj_format.ObjFile, scale: float) -> Matrix:
    matrix = game_to_blender_matrix(game_object.matrix.rows)
    matrix.translation = Vector(game_to_blender_point(game_object.translation, scale))
    return matrix


def _link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if not any(existing == obj for existing in collection.objects):
        collection.objects.link(obj)
    for user_collection in list(obj.users_collection):
        if user_collection != collection:
            user_collection.objects.unlink(obj)


def _clean_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    stem = re.sub(r"__variant_[0-9]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(\$[0-9A-Fa-f]+\)", "", stem)
    return stem


def _match_key(name: str) -> str:
    stem = _clean_stem(name)
    if stem.lower().startswith("pfb_"):
        stem = stem[4:]
    stem = re.sub(r"_[0-9]+$", "", stem)
    return re.sub(r"[^A-Za-z0-9]+", "", stem).lower()


def _literal_key(name: str) -> str:
    stem = _clean_stem(name)
    if stem.lower().startswith("pfb_"):
        stem = stem[4:]
    return re.sub(r"[^A-Za-z0-9]+", "", stem).lower()


def _variant_code(name: str) -> str:
    match = re.search(r"\(\$([0-9A-Fa-f]+)\)", os.path.basename(name))
    return match.group(1).lower() if match else ""


def _context_keys(node: ObjectGraph) -> Tuple[int, ...]:
    keys: List[int] = []
    keys.extend(node.sidecar_keys)
    if node.object_data:
        keys.extend(ref.key for ref in node.object_data.resource_refs)
    seen: set[int] = set()
    result: List[int] = []
    for key in keys:
        normalized = key & 0xFFFFFFFF
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _tokens(name: str) -> set[str]:
    stem = _clean_stem(name)
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem)
    return {
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", stem)
        if len(token) >= 3 and not token.isdigit()
    }


def _without_lod_token(text: str) -> str:
    text = re.sub(r"lod[0-9]+(?:_[0-9]+)?$", "", text)
    text = re.sub(r"_?lod[0-9]+(?:_[0-9]+)?$", "", text)
    return text


def _without_instance_suffix(text: str) -> str:
    return re.sub(r"_[0-9]+$", "", text)


def _swap_side_suffix(text: str) -> str:
    replacements = (
        ("left", "right"),
        ("right", "left"),
        ("wheell", "wheelr"),
        ("wheelr", "wheell"),
        ("tyrel", "tyrer"),
        ("tyrer", "tyrel"),
    )
    for old, new in replacements:
        if text.endswith(old):
            return text[:-len(old)] + new
    if text.endswith("l") and len(text) > 4:
        return text[:-1] + "r"
    if text.endswith("r") and len(text) > 4:
        return text[:-1] + "l"
    return text


def _model_stem_parts(name: str) -> tuple[str, str, str]:
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    stem = re.sub(r"__variant_[0-9]+$", "", stem, flags=re.IGNORECASE)
    if stem.lower().startswith("pfb_"):
        stem = stem[4:]
    stem = stem.replace(" ", "_")

    match = re.search(r"\(\$[0-9A-Fa-f]+\)", stem)
    if not match:
        return stem, "", ""
    return stem[:match.start()], match.group(0), stem[match.end():]


def _compact_model_key(before_variant: str, after_variant: str = "") -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", before_variant + after_variant).lower()


def _add_model_form(forms: List[Tuple[str, int]], before_variant: str, after_variant: str, cost: int) -> None:
    key = _compact_model_key(before_variant, after_variant)
    if not key:
        return
    if any(existing == key and existing_cost <= cost for existing, existing_cost in forms):
        return
    forms[:] = [(existing, existing_cost) for existing, existing_cost in forms if existing != key]
    forms.append((key, cost))


def _model_key_forms(name: str, *, for_geo: bool = False) -> Tuple[Tuple[str, int], ...]:
    before_variant, _variant, after_variant = _model_stem_parts(name)
    if for_geo:
        before_variant = _without_lod_token(before_variant)
        after_variant = _without_lod_token(after_variant)

    forms: List[Tuple[str, int]] = []
    _add_model_form(forms, before_variant, after_variant, 0)

    stripped_after = _without_instance_suffix(after_variant)
    if stripped_after != after_variant:
        _add_model_form(forms, before_variant, stripped_after, 5 if not for_geo else 20)

    stripped_before = _without_instance_suffix(before_variant)
    if stripped_before != before_variant:
        _add_model_form(forms, stripped_before, after_variant, 10 if not for_geo else 30)
        if stripped_after != after_variant:
            _add_model_form(forms, stripped_before, stripped_after, 15 if not for_geo else 35)

    stripped_whole = _without_instance_suffix(before_variant + after_variant)
    if stripped_whole != before_variant + after_variant:
        _add_model_form(forms, stripped_whole, "", 8 if not for_geo else 30)

    mirrored = _swap_side_suffix(forms[0][0]) if forms else ""
    if mirrored and mirrored != forms[0][0]:
        forms.append((mirrored, 12))

    forms.sort(key=lambda item: (item[1], -len(item[0]), item[0]))
    return tuple(forms)


def _model_key_candidates(name: str, *, for_geo: bool = False) -> Tuple[str, ...]:
    return tuple(key for key, _cost in _model_key_forms(name, for_geo=for_geo))


def _geo_instance_rank(name: str) -> int:
    stem = _clean_stem(name)
    stem = re.sub(r"\(\$[0-9A-Fa-f]+\)", "", stem)
    stem = _without_lod_token(stem).strip("_")
    return 1 if re.search(r"_[0-9]+$", stem) else 0


def _variant_rank(object_name: str, geo_name: str, _context_keys: Sequence[int]) -> int:
    object_variant = _variant_code(object_name)
    geo_variant = _variant_code(geo_name)
    if object_variant:
        if geo_variant:
            return 4 if geo_variant == object_variant else 0
        return 1
    return 2 if not geo_variant else 1


def _variant_compatible(object_name: str, geo_name: str) -> bool:
    object_variant = _variant_code(object_name)
    geo_variant = _variant_code(geo_name)
    if object_variant and geo_variant and object_variant != geo_variant:
        return False
    return True


def _name_score(object_name: str, geo_name: str) -> int:
    if not _variant_compatible(object_name, geo_name):
        return 0

    object_forms = _model_key_forms(object_name)
    geo_forms = _model_key_forms(geo_name, for_geo=True)
    best = 0
    for object_key, object_cost in object_forms:
        for geo_key, geo_cost in geo_forms:
            if object_key != geo_key:
                continue
            score = 140 - object_cost - geo_cost
            if score > best:
                best = score
    return best


def _context_name_score(object_name: str, geo_name: str, context_keys: Sequence[int]) -> int:
    score = _name_score(object_name, geo_name)
    if score <= 0:
        return 0
    return score + _variant_rank(object_name, geo_name, context_keys) * 10


def _build_geo_name_index(geos: Sequence[resource_index.ResourceFile]) -> GeoNameIndex:
    literal: Dict[str, List[resource_index.ResourceFile]] = {}
    match: Dict[str, List[resource_index.ResourceFile]] = {}
    model: Dict[str, List[resource_index.ResourceFile]] = {}
    for geo in geos:
        literal.setdefault(_literal_key(geo.name), []).append(geo)
        match.setdefault(_match_key(geo.name), []).append(geo)
        for key in _model_key_candidates(geo.name, for_geo=True):
            model.setdefault(key, []).append(geo)
    return GeoNameIndex(
        by_literal={key: tuple(value) for key, value in literal.items() if key},
        by_match={key: tuple(value) for key, value in match.items() if key},
        by_model={key: tuple(value) for key, value in model.items() if key},
    )


def _best_scored_name_candidate(
    object_name: str,
    candidates: Sequence[resource_index.ResourceFile],
    cursor: int = 0,
    context_keys: Sequence[int] = (),
) -> resource_index.ResourceFile | None:
    candidates = _unique_resources(candidates)
    if not candidates:
        return None
    if len(candidates) == 1:
        if not object_name:
            return candidates[0]
        score = _context_name_score(object_name, candidates[0].name, context_keys)
        return candidates[0] if score >= 100 else None

    scored = [
        (_context_name_score(object_name, resource.name, context_keys), resource.entry_index, resource)
        for resource in candidates
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] < 100:
        return None
    runner_up = scored[1][0] if len(scored) > 1 else -1
    if scored[0][0] <= runner_up:
        tied = [item for item in scored if item[0] == scored[0][0]]
        tied_after = [item for item in tied if item[1] >= cursor]
        chosen = min(tied_after or tied, key=lambda item: item[1])
        return chosen[2]
    return scored[0][2]


def _best_name_index_candidate(
    node: ObjectGraph,
    geo_name_index: GeoNameIndex,
    cursor: int,
) -> resource_index.ResourceFile | None:
    if not node.object_file:
        return None

    object_name = node.object_file.name
    context_keys = _context_keys(node)
    candidates: List[resource_index.ResourceFile] = []
    candidates.extend(geo_name_index.by_literal.get(_literal_key(object_name), ()))
    candidates.extend(geo_name_index.by_match.get(_match_key(object_name), ()))
    for key in _model_key_candidates(object_name):
        candidates.extend(geo_name_index.by_model.get(key, ()))
    return _best_scored_name_candidate(object_name, candidates, cursor, context_keys)


def _ordered_graphic_geos(
    index: resource_index.GameResourceIndex,
    archive_paths: Sequence[str],
) -> List[resource_index.ResourceFile]:
    resources: List[resource_index.ResourceFile] = []
    for archive_path in archive_paths:
        resources.extend(index.entries_for_archive(archive_path, ".geo"))
    resources.sort(key=lambda item: (item.archive_priority, item.archive_name.lower(), item.entry_index))
    return resources


def _best_ordered_key_candidate(
    candidates: Sequence[resource_index.ResourceFile],
    world_archive_path: str,
    cursor: int,
    object_name: str = "",
    context_keys: Sequence[int] = (),
) -> resource_index.ResourceFile | None:
    candidates = _unique_resources(candidates)
    if not candidates:
        return None
    if len(candidates) == 1:
        if not object_name:
            return candidates[0]
        score = _context_name_score(object_name, candidates[0].name, context_keys)
        return candidates[0] if score >= 100 else None

    world_archive_path = os.path.abspath(world_archive_path)
    if object_name:
        scored = [
            (
                _context_name_score(object_name, resource.name, context_keys),
                resource.archive_path == world_archive_path,
                resource.entry_index,
                resource,
            )
            for resource in candidates
        ]
        scored.sort(key=lambda item: (-item[0], not item[1], item[2]))
        if scored and scored[0][0] >= 100:
            runner_up = scored[1][0] if len(scored) > 1 else -1
            if scored[0][0] > runner_up:
                return scored[0][3]

            tied = [item for item in scored if item[0] == scored[0][0]]
            object_keys = set(_model_key_candidates(object_name))
            matched_keys = {
                geo_key
                for item in tied
                for geo_key in _model_key_candidates(item[3].name, for_geo=True)
                if geo_key in object_keys
            }
            if len(matched_keys) == 1:
                tied_after = [item for item in tied if item[2] >= cursor]
                chosen = min(tied_after or tied, key=lambda item: (_geo_instance_rank(item[3].name), item[2]))
                return chosen[3]
    return None


def _candidate_lookup_keys(node: ObjectGraph) -> Tuple[int, ...]:
    result: List[int] = []
    seen: set[int] = set()
    # MTN's first real key is the strongest GEO-family link. Later sidecar
    # refs are mostly visual/setup keys and make the global content scan too
    # expensive for large worlds.
    keys: List[int] = list(node.geo_sidecar_keys)
    for key in keys:
        normalized = key & 0xFFFFFFFF
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _usable_moti_geo_keys(
    moti_keys: Sequence[int],
    visual_keys: set[int],
) -> Tuple[int, ...]:
    # MOTI starts with the packed graphic/GEO family key. Later words are the
    # matching VIS key and setup values; scanning them as GEO ids creates wrong
    # prop matches like barrier objects resolving to buildings.
    for key in moti_keys:
        normalized = key & 0xFFFFFFFF
        if normalized in visual_keys:
            continue
        if normalized in {0, 0xFFFFFFFF, 0x10000001}:
            continue
        return (normalized,)
    return ()


def _find_indexed_geo(
    node: ObjectGraph,
    geo_key_index: Dict[int, List[resource_index.ResourceFile]],
    world_archive_path: str,
    cursor: int,
) -> tuple[resource_index.ResourceFile | None, int | None, int]:
    best_file: resource_index.ResourceFile | None = None
    best_key: int | None = None
    best_count = 0
    for key in _candidate_lookup_keys(node):
        candidates = _unique_resources(geo_key_index.get(key, ()))
        if not candidates:
            continue
        best_count = max(best_count, len(candidates))
        object_name = node.object_file.name if node.object_file else ""
        candidate = _best_ordered_key_candidate(
            candidates,
            world_archive_path,
            cursor,
            object_name,
            _context_keys(node),
        )
        if candidate is None:
            continue
        if best_file is None:
            best_file = candidate
            best_key = key
            continue
        if candidate.archive_path == os.path.abspath(world_archive_path) and candidate.entry_index >= cursor:
            if best_file.archive_path != os.path.abspath(world_archive_path) or candidate.entry_index < best_file.entry_index:
                best_file = candidate
                best_key = key
    return best_file, best_key, best_count


def _build_object_graph(
    index: resource_index.GameResourceIndex,
    world: wor_format.WorFile,
    world_archive_path: str,
    material_link_keys: frozenset[int] = frozenset(),
) -> Tuple[List[ObjectGraph], Dict[int, List[resource_index.ResourceFile]]]:
    world_archive_paths = index.archive_paths_for_kinds(("World", "WorldChunk"))
    if not world_archive_paths:
        world_archive_paths = [world_archive_path]
    object_files = index.entries_for_archive(world_archive_path, ".obj")
    ordered_geos = _ordered_graphic_geos(index, world_archive_paths)
    geo_name_index = _build_geo_name_index(ordered_geos)
    graph: List[ObjectGraph] = []
    all_lookup_keys: List[int] = []

    for ref in world.object_refs:
        object_file = object_files[ref.index] if ref.index < len(object_files) else None
        object_data: obj_format.ObjFile | None = None
        obj_error = ""
        sidecars: Tuple[resource_index.ResourceFile, ...] = ()
        sidecar_keys: Tuple[int, ...] = ()
        geo_sidecar_keys: Tuple[int, ...] = ()
        material_sidecar_keys: Tuple[int, ...] = ()
        if object_file is not None:
            try:
                object_data = obj_format.parse_obj(index.read(object_file), object_file.name)
            except Exception as exc:
                object_data = None
                obj_error = str(exc)
            sidecars = tuple(index.companion_files(object_file, (".vii", ".mtn")))
            keys: List[int] = []
            geo_keys: List[int] = []
            visual_keys: set[int] = set()
            parsed_moti_keys: List[Tuple[int, ...]] = []
            for sidecar in sidecars:
                try:
                    parsed = obj_format.parse_sidecar(index.read(sidecar), sidecar.name)
                except Exception:
                    continue
                keys.extend(parsed.resource_keys)
                if parsed.kind == "VIS":
                    visual_keys.update(key & 0xFFFFFFFF for key in parsed.resource_keys)
                elif parsed.kind == "MOTI" and parsed.resource_keys:
                    parsed_moti_keys.append(parsed.resource_keys)
            for moti_keys in parsed_moti_keys:
                geo_keys.extend(_usable_moti_geo_keys(moti_keys, visual_keys))
            sidecar_keys = tuple(keys)
            geo_sidecar_keys = tuple(geo_keys)
            material_sidecar_keys = _material_sidecar_keys(keys, material_link_keys)
        node = ObjectGraph(ref, object_file, object_data, sidecars, sidecar_keys, geo_sidecar_keys, material_sidecar_keys)
        node.obj_error = obj_error
        graph.append(node)
        all_lookup_keys.extend(_candidate_lookup_keys(node))

    geo_key_index = index.build_content_key_index(
        all_lookup_keys,
        (".geo",),
        world_archive_path,
        archive_paths=world_archive_paths,
    )

    missing_external_keys: List[int] = []
    for node in graph:
        if not _find_indexed_geo(node, geo_key_index, world_archive_path, 0)[0]:
            missing_external_keys.extend(_candidate_lookup_keys(node))

    world_archive_path_set = {os.path.abspath(path) for path in world_archive_paths}
    external_paths = [
        archive_path for archive_path in index.archives
        if os.path.abspath(archive_path) not in world_archive_path_set
    ]
    external_geo_key_index: Dict[int, List[resource_index.ResourceFile]] = {}
    if missing_external_keys and external_paths:
        external_geo_key_index = index.build_content_key_index(
            missing_external_keys,
            (".geo",),
            world_archive_path,
            archive_paths=external_paths,
        )

    cursor = ordered_geos[0].entry_index if ordered_geos else 0
    graphic_key_to_geos: Dict[int, List[resource_index.ResourceFile]] = {}
    for node in graph:
        graphic_ref = node.object_data.primary_graphic_ref if node.object_data else None
        graphic_key = graphic_ref.key & 0xFFFFFFFF if graphic_ref else None

        geo_file, geo_key, candidate_count = _find_indexed_geo(node, geo_key_index, world_archive_path, cursor)
        if geo_file is not None:
            node.geo_file = geo_file
            node.geo_key = geo_key
            node.geo_state = "sidecar-order"
            node.geo_candidates = max(candidate_count, 1)
        else:
            if candidate_count > 0:
                node.geo_state = "sidecar-ambiguous"
                node.geo_candidates = candidate_count
            external_file, external_key, external_count = _find_indexed_geo(
                node,
                external_geo_key_index,
                world_archive_path,
                cursor,
            )
            if external_file is not None and external_count == 1:
                node.geo_file = external_file
                node.geo_key = external_key
                node.geo_state = "common-key"
                node.geo_candidates = 1
            else:
                if external_count > 1:
                    node.geo_state = "common-ambiguous"
                    node.geo_candidates = external_count
                    node.geo_key = external_key
                name_file = _best_name_index_candidate(node, geo_name_index, cursor)
                if name_file is not None:
                    node.geo_file = name_file
                    node.geo_state = "name-match"
                    node.geo_candidates = max(node.geo_candidates, 1)

        if node.geo_file is None and graphic_key is not None:
            object_name = node.object_file.name if node.object_file else ""
            repeat_file = _best_scored_name_candidate(
                object_name,
                graphic_key_to_geos.get(graphic_key, ()),
                cursor,
                _context_keys(node),
            )
            if repeat_file is not None:
                node.geo_file = repeat_file
                node.geo_key = graphic_key
                node.geo_state = "graphic-repeat"
                node.geo_candidates = 1

        if node.geo_file and graphic_key is not None:
            bucket = graphic_key_to_geos.setdefault(graphic_key, [])
            identity = (node.geo_file.archive_path, node.geo_file.entry_index)
            if not any((geo.archive_path, geo.entry_index) == identity for geo in bucket):
                bucket.append(node.geo_file)

        if node.geo_file and node.geo_file.archive_path == os.path.abspath(world_archive_path):
            if node.geo_file.entry_index >= cursor:
                cursor = node.geo_file.entry_index + 1
    return graph, geo_key_index


def import_world_archive(
    game_dir: str,
    world_archive_path: str,
    *,
    include_common: bool = True,
    include_sound: bool = False,
    include_video: bool = False,
    create_ref_empties: bool = False,
    empty_limit: int = 250,
    import_meshes: bool = True,
    object_limit: int = 0,
    orient_upright: bool = True,
    scale: float = 1.0,
    flip_uv_v: bool = True,
    resolve_textures: bool = False,
    convert_tdt_textures: bool = True,
    texture_alpha_mode: str = "opaque",
) -> List[bpy.types.Object]:
    index = resource_index.build_world_resource_index(
        game_dir,
        world_archive_path,
        include_common=include_common,
        include_sound=include_sound,
        include_video=include_video,
    )
    world_files = index.entries_for_archive(world_archive_path, ".wor")
    if not world_files:
        raise ValueError("selected archive has no WOR file")

    roots: List[bpy.types.Object] = []
    cache_root = resource_index.default_resource_cache_dir()
    material_context = _build_world_material_context(index, cache_root) if resolve_textures else None
    shared_texture_resolver = None
    if material_context is not None:
        shared_texture_resolver = TextureResolver(
            os.path.join(cache_root, "world_materials.geo"),
            convert_tdt_textures,
            texture_alpha_mode,
            search_dirs=material_context.search_dirs,
            mta_stems_by_key=material_context.mta_stems_by_key,
            cache_dir=material_context.cache_dir,
        )

    for world_file in world_files:
        world = wor_format.parse_wor(index.read(world_file), world_file.name)
        graph, _geo_key_index = _build_object_graph(
            index,
            world,
            os.path.abspath(world_archive_path),
            material_context.material_link_keys if material_context else frozenset(),
        )

        base_name = os.path.splitext(world_file.name)[0]
        collection = bpy.data.collections.new(base_name)
        bpy.context.scene.collection.children.link(collection)

        root = bpy.data.objects.new(base_name, None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 1.0
        collection.objects.link(root)

        root["zombiu_world_importer"] = "object-resource-name"
        root["bfz_path"] = os.path.abspath(world_archive_path)
        root["mounted_archive_count"] = len(index.archives)
        root["wor_object_refs"] = world.object_count
        root["wor_version"] = world.version
        if material_context is not None:
            root["wor_material_search_dirs"] = len(material_context.search_dirs)
            root["wor_material_key_map"] = len(material_context.mta_stems_by_key)

        if orient_upright:
            root.rotation_euler[0] = -math.pi / 2.0
            root["wor_root_rotation"] = "x -90"

        geo_cache: Dict[Tuple[str, int], List[bpy.types.Object]] = {}
        imported_geo = 0
        resolved_geo = 0
        ambiguous_geo = 0
        missing_geo = 0
        created_empties = 0
        lines = [
            base_name,
            "",
            "importer: object-resource-name",
            f"archive: {os.path.basename(world_archive_path)}",
            f"mounted archives: {len(index.archives)}",
            f"object refs: {world.object_count}",
            "",
            "objects:",
        ]

        limit = object_limit if object_limit > 0 else len(graph)
        for node in graph[:limit]:
            object_file = node.object_file
            object_name = object_file.name if object_file else node.ref.key_hex
            matrix = Matrix.Identity(4)
            if node.object_data is not None:
                matrix = _object_matrix(node.object_data, scale)

            if node.geo_file is not None:
                resolved_geo += 1
            elif "ambiguous" in node.geo_state:
                ambiguous_geo += 1
            else:
                missing_geo += 1

            should_create_empty = create_ref_empties or bool(import_meshes and node.geo_file)
            empty = None
            if should_create_empty and created_empties < max(0, empty_limit if not import_meshes else limit):
                empty = bpy.data.objects.new(os.path.splitext(object_name)[0], None)
                empty.empty_display_type = "PLAIN_AXES"
                empty.empty_display_size = 0.25
                empty.parent = root
                empty.matrix_world = matrix
                empty["wor_ref_index"] = node.ref.index
                empty["wor_ref_key"] = node.ref.key_hex
                empty["object_resource"] = object_file.normalized_path if object_file else ""
                empty["geo_state"] = node.geo_state
                empty["geo_key"] = f"{node.geo_key:08X}" if node.geo_key is not None else ""
                empty["geo_candidates"] = node.geo_candidates
                if node.object_data and node.object_data.primary_graphic_ref:
                    empty["graphic_key"] = node.object_data.primary_graphic_ref.key_hex
                if node.geo_file:
                    empty["geo_resource"] = node.geo_file.normalized_path
                    empty["geo_archive"] = node.geo_file.archive_name
                if node.obj_error:
                    empty["obj_error"] = node.obj_error
                collection.objects.link(empty)
                created_empties += 1

            if import_meshes and node.geo_file and empty is not None:
                geo_cache_key = (node.geo_file.archive_path, node.geo_file.entry_index)
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
                    geo_path = index.extract(node.geo_file, cache_root)
                    material_info = _material_info_for_geo_resource(node.geo_file, material_context)
                    object_material_keys = tuple(ref.key for ref in node.object_data.material_refs) if node.object_data else ()
                    material_key_hints = tuple(dict.fromkeys((*node.material_sidecar_keys, *object_material_keys)))
                    objects = import_geo(
                        geo_path,
                        scale,
                        flip_uv_v,
                        resolve_textures,
                        convert_tdt_textures,
                        texture_alpha_mode,
                        None,
                        False,
                        texture_query_hints=(object_name, node.geo_file.name),
                        material_key_hints=material_key_hints,
                        material_info_override=material_info,
                        texture_resolver=shared_texture_resolver,
                        texture_search_dirs=material_context.search_dirs if material_context else None,
                        mta_stems_by_key=material_context.mta_stems_by_key if material_context else None,
                    )
                    for obj in objects:
                        _link_to_collection(obj, collection)
                    geo_cache[geo_cache_key] = list(objects)

                for obj in objects:
                    obj.parent = empty
                    obj.matrix_parent_inverse = Matrix.Identity(4)
                    obj.matrix_basis = Matrix.Identity(4)
                    obj["wor_ref_index"] = node.ref.index
                    obj["object_resource"] = object_file.normalized_path if object_file else ""
                    obj["geo_resource"] = node.geo_file.normalized_path
                    obj["geo_archive"] = node.geo_file.archive_name
                    if node.material_sidecar_keys:
                        obj["mtn_material_keys"] = ",".join(f"{key & 0xFFFFFFFF:08X}" for key in node.material_sidecar_keys[:8])
                imported_geo += 1

            if node.geo_file:
                target = f"{object_name} -> {node.geo_file.name} ({node.geo_state})"
            else:
                target = f"{object_name} -> {node.geo_state}"
            if node.geo_key is not None:
                target += f" key={node.geo_key:08X}"
            if node.geo_candidates:
                target += f" candidates={node.geo_candidates}"
            if node.object_data and node.object_data.primary_graphic_ref:
                target += f" graphic={node.object_data.primary_graphic_ref.key_hex}"
            if node.obj_error:
                target += f" obj_error={node.obj_error}"
            lines.append(f"- [{node.ref.index:03d}] {node.ref.key_hex} -> {target}")

        if len(graph) > limit:
            lines.append(f"... {len(graph) - limit:,} more refs skipped by import limit")

        root["wor_resolved_geo"] = resolved_geo
        root["wor_ambiguous_geo"] = ambiguous_geo
        root["wor_missing_geo"] = missing_geo
        root["wor_imported_geo_instances"] = imported_geo
        root["wor_unique_geo_imports"] = len(geo_cache)
        root["wor_created_empties"] = created_empties

        text = bpy.data.texts.new(f"{base_name}_resource_graph")
        text.write("\n".join(lines))

        roots.append(root)

    if roots:
        bpy.context.view_layer.objects.active = roots[0]
        roots[0].select_set(True)
    return roots
