from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import bpy

from . import wor_format
from .material_format import MtaDescriptor, mat_key_to_mta_stem_map, parse_mta_file, parse_tex_file
from .tdt import convert_tdt_to_png, tdt_top_mip_has_non_opaque_alpha

TEXTURE_ROLE_SUFFIXES = ("RGBAO", "ENCA", "ENC", "DAT", "NM", "AO", "DA", "D")
TEXTURE_COLOR_ROLES = ("D", "DA", "DAT", "COLOR")
TEXTURE_NORMAL_ROLES = ("NM",)
TEXTURE_MIN_SCORE = 0.62
TEXTURE_MTA_MIN_SCORE = 0.72
TEXTURE_GENERIC_KEYS = {"body", "head", "hair", "eye", "eyes"}
TEXTURE_TOKEN_STOPWORDS = {
    "ch", "gen", "com", "skin", "misc", "drt", "lod", "pc", "h", "ca",
    "af", "as", "z", "reg", "med", "m", "f", "t", "w", "weapon", "material",
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
    "default",
    "ach", "brk", "co", "di", "ele", "fa", "fur", "gl", "grd", "hub",
    "la", "lou", "low", "lt", "mat", "me", "msc", "pap", "pfb", "pl",
    "pr", "sh", "sig", "sta", "u", "vfx", "wo",
    "0", "1", "2", "3", "4", "5",
}
TEXTURE_TOKEN_ALIASES = {
    "nrs": "nurse",
    "trd": "trader",
    "lo": "low",
    "up": "upper",
    "suit": "uniform",
    "pants": "pant",
    "eyes": "eye",
}
TEXTURE_MATERIAL_PREFIXES = {
    "ach",
    "brk",
    "co",
    "di",
    "fa",
    "gl",
    "grd",
    "me",
    "pap",
    "pl",
    "sig",
    "vfx",
    "wo",
}
TEXTURE_TOKEN_WEIGHTS = {
    "cop": 4,
    "nurse": 3,
    "trader": 4,
    "shirt": 5,
    "pant": 5,
    "helmet": 6,
    "hair": 7,
    "eye": 6,
    "cornea": 6,
    "head": 4,
    "body": 3,
    "uniform": 5,
    "female": 3,
    "male": 3,
    "thin": 3,
}


@dataclass
class TextureAsset:
    stem: str
    group_key: str
    role: str | None
    png_path: str | None = None
    tdt_path: str | None = None
    tex_path: str | None = None
    file_key: int | None = None


@dataclass
class TextureGroup:
    key: str
    assets: dict[str, TextureAsset]
    shader_name: str | None = None
    primary_refs: Tuple[int, ...] = ()


@dataclass
class ResolvedTexture:
    group: TextureGroup
    material_key: str
    score: float
    method: str


def texture_stem_from_path(path: str) -> str:
    name = os.path.basename(path)
    lowered = name.lower()
    if lowered.endswith(".pc.tdt"):
        return name[:-7]
    return os.path.splitext(name)[0]


def split_texture_role(stem: str) -> Tuple[str, str | None]:
    upper = stem.upper()
    for role in TEXTURE_ROLE_SUFFIXES:
        suffix = "_" + role
        if upper.endswith(suffix):
            return stem[:-len(suffix)], role
    return stem, None


def normalized_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def texture_tokens(text: str) -> List[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).lower()
    tokens: List[str] = []
    for token in text.split("_"):
        if not token or token in TEXTURE_TOKEN_STOPWORDS:
            continue
        for piece in re.findall(r"[a-z]+|\d+", token):
            if piece and piece not in TEXTURE_TOKEN_STOPWORDS:
                tokens.append(TEXTURE_TOKEN_ALIASES.get(piece, piece))
    return tokens


def texture_token_weight(token: str) -> int:
    return TEXTURE_TOKEN_WEIGHTS.get(token, 1)


def texture_name_score(query: str, candidate: str) -> float:
    query_tokens = set(texture_tokens(query))
    candidate_tokens = set(texture_tokens(candidate))
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap_weight = sum(texture_token_weight(token) for token in query_tokens & candidate_tokens)
    query_weight = sum(texture_token_weight(token) for token in query_tokens)
    candidate_weight = sum(texture_token_weight(token) for token in candidate_tokens)
    overlap = overlap_weight / max(1, query_weight)
    coverage = overlap_weight / max(1, candidate_weight)
    sequence = difflib.SequenceMatcher(None, normalized_name(query), normalized_name(candidate)).ratio()
    score = overlap * 0.65 + coverage * 0.20 + sequence * 0.15

    if "vfx" in candidate_tokens and "vfx" not in query_tokens:
        score -= 0.20
    return score


def extract_material_key(part_name: str) -> str | None:
    match = re.search(r"\(([^()]*)\)\s*$", part_name)
    if not match:
        return None
    key = match.group(1).strip()
    return key or None


def unique_strings(values: Iterable[str | None]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if not value:
            continue
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def texture_query_aliases(query: str) -> List[str]:
    stem, _role = split_texture_role(query)
    aliases = [stem]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].lower() in TEXTURE_MATERIAL_PREFIXES:
        tail = "_".join(parts[1:])
        aliases.append("LA_" + tail)
        aliases.append("Pr_" + tail)
    return unique_strings(aliases)


def is_generic_texture_query(query: str) -> bool:
    key_tokens = set(texture_tokens(query))
    return len(key_tokens) <= 1 and bool(key_tokens & TEXTURE_GENERIC_KEYS)


def is_useful_material_query(query: str) -> bool:
    tokens = [
        token
        for token in texture_tokens(query)
        if not token.isdigit() and token not in {"default", "material"}
    ]
    return bool(tokens)


def unique_existing_dirs(paths: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for path in paths:
        if not path:
            continue
        normalized = os.path.abspath(path)
        if normalized in seen or not os.path.isdir(normalized):
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def texture_search_dirs(geo_path: str) -> List[str]:
    geo_dir = os.path.dirname(os.path.abspath(geo_path))
    parent_dir = os.path.dirname(geo_dir)
    if os.path.exists(os.path.join(geo_dir, "_zombi_bfz_manifest.json")):
        return unique_existing_dirs([
            os.path.join(geo_dir, "converted_textures"),
            geo_dir,
            os.path.join(parent_dir, "converted_textures"),
            parent_dir,
        ])
    return unique_existing_dirs([
        os.path.join(parent_dir, "converted_textures"),
        os.path.join(geo_dir, "converted_textures"),
        geo_dir,
        parent_dir,
    ])


def texture_cache_dir(geo_path: str) -> str:
    geo_dir = os.path.dirname(os.path.abspath(geo_path))
    parent_dir = os.path.dirname(geo_dir)
    if os.path.exists(os.path.join(geo_dir, "_zombi_bfz_manifest.json")):
        return os.path.join(geo_dir, "converted_textures")
    for candidate in (
        os.path.join(parent_dir, "converted_textures"),
        os.path.join(geo_dir, "converted_textures"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(parent_dir, "converted_textures")


# exact geo material keys win; name scoring only fills gaps.
class TextureResolver:
    def __init__(
        self,
        geo_path: str,
        convert_tdt: bool,
        alpha_mode: str = "opaque",
        search_dirs: Sequence[str] | None = None,
        mta_stems_by_key: dict[int, str] | None = None,
        cache_dir: str | None = None,
    ):
        self.geo_path = geo_path
        self.convert_tdt = convert_tdt
        self.alpha_mode = alpha_mode
        self.search_dirs = tuple(search_dirs or ())
        self.external_mta_stems_by_key = {
            key & 0xFFFFFFFF: stem
            for key, stem in (mta_stems_by_key or {}).items()
            if stem
        }
        self.cache_dir = cache_dir or texture_cache_dir(geo_path)
        self.groups: dict[str, TextureGroup] = {}
        self.assets_by_key: dict[int, TextureAsset] = {}
        self.mta_descriptors: dict[str, MtaDescriptor] = {}
        self.mta_stems_by_key: dict[int, str] = {}
        self.mta_material_keys: set[int] = set()
        self.mta_groups: dict[str, TextureGroup] = {}
        self.materials: dict[str, bpy.types.Material] = {}
        self.alpha_probe_cache: dict[str, bool] = {}
        self.scan()

    def scan(self) -> None:
        directories = unique_existing_dirs(self.search_dirs) if self.search_dirs else texture_search_dirs(self.geo_path)
        assets_by_stem: dict[str, TextureAsset] = {}
        for directory in directories:
            for name in os.listdir(directory):
                lowered = name.lower()
                if not (lowered.endswith(".png") or lowered.endswith(".tex") or lowered.endswith(".tdt")):
                    continue

                path = os.path.join(directory, name)
                stem = texture_stem_from_path(path)
                group_key, role = split_texture_role(stem)
                asset = assets_by_stem.get(stem)
                if asset is None:
                    asset = TextureAsset(stem=stem, group_key=group_key, role=role)
                    assets_by_stem[stem] = asset

                if lowered.endswith(".png"):
                    asset.png_path = asset.png_path or path
                elif lowered.endswith(".tdt"):
                    asset.tdt_path = asset.tdt_path or path
                elif lowered.endswith(".tex"):
                    asset.tex_path = asset.tex_path or path
                    if asset.file_key is None:
                        try:
                            asset.file_key = parse_tex_file(path).file_key
                        except Exception:
                            asset.file_key = None
                    companion = os.path.join(directory, stem + ".PC.tdt")
                    if os.path.exists(companion):
                        asset.tdt_path = asset.tdt_path or companion

        for asset in assets_by_stem.values():
            role_key = asset.role or "COLOR"
            group = self.groups.setdefault(asset.group_key.lower(), TextureGroup(asset.group_key, {}))
            existing = group.assets.get(role_key)
            if existing is None or (not existing.png_path and asset.png_path):
                group.assets[role_key] = asset
            if asset.file_key is not None:
                existing_key_asset = self.assets_by_key.get(asset.file_key)
                if existing_key_asset is None or (not existing_key_asset.png_path and asset.png_path):
                    self.assets_by_key[asset.file_key] = asset

        for directory in directories:
            for name in os.listdir(directory):
                if not name.lower().endswith(".mta"):
                    continue
                path = os.path.join(directory, name)
                stem = os.path.splitext(name)[0]
                try:
                    self.mta_descriptors.setdefault(stem.lower(), parse_mta_file(path))
                except Exception:
                    continue

        manifest_entries = wor_format.load_nearby_manifest_entries(self.geo_path)
        for entry in manifest_entries:
            if entry.extension != ".mta":
                continue
            self.mta_stems_by_key.setdefault(entry.key, os.path.splitext(os.path.basename(entry.path))[0])

        mat_paths = [entry.path for entry in manifest_entries if entry.extension == ".mat"]
        mta_paths = [entry.path for entry in manifest_entries if entry.extension == ".mta"]
        for key, stem in mat_key_to_mta_stem_map(mat_paths, mta_paths).items():
            self.mta_stems_by_key.setdefault(key, stem)
            self.mta_material_keys.add(key & 0xFFFFFFFF)

        for key, stem in self.external_mta_stems_by_key.items():
            self.mta_stems_by_key.setdefault(key, stem)
            self.mta_material_keys.add(key & 0xFFFFFFFF)

    def group_from_mta(self, stem: str) -> TextureGroup | None:
        descriptor = self.mta_descriptors.get(stem.lower())
        if not descriptor:
            return None

        cache_key = stem.lower()
        cached = self.mta_groups.get(cache_key)
        if cached:
            return cached

        assets: dict[str, TextureAsset] = {}
        for ref in (*descriptor.primary_texture_refs, *descriptor.extra_texture_refs):
            asset = self.assets_by_key.get((ref + 1) & 0xFFFFFFFF)
            if not asset:
                continue
            role = asset.role or "COLOR"
            if role not in assets:
                assets[role] = asset

        if not assets:
            return None

        group = TextureGroup(
            os.path.splitext(descriptor.name)[0],
            assets,
            shader_name=descriptor.shader_name or None,
            primary_refs=descriptor.primary_texture_refs,
        )
        self.mta_groups[cache_key] = group
        return group

    def resolve_mta_query(self, query: str) -> TextureGroup | None:
        if not is_useful_material_query(query):
            return None
        for alias in texture_query_aliases(query):
            stem_key, _role = split_texture_role(alias)
            group = self.group_from_mta(stem_key)
            if group:
                return group
        return None

    def resolve_mta_key(self, key: int | None) -> TextureGroup | None:
        if key is None:
            return None
        stem = self.mta_stems_by_key.get(key & 0xFFFFFFFF)
        if not stem:
            return None
        return self.group_from_mta(stem)

    def resolve(
        self,
        base_name: str,
        part_name: str,
        extra_queries: Sequence[str] | None = None,
        material_file_key: int | None = None,
    ) -> ResolvedTexture | None:
        extra_queries = extra_queries or ()
        material_key = extract_material_key(part_name)

        group = self.resolve_mta_key(material_file_key)
        if group:
            method = "mta-material-key" if (material_file_key & 0xFFFFFFFF) in self.mta_material_keys else "mta-resource-key"
            return ResolvedTexture(
                group=group,
                material_key=f"{material_file_key & 0xFFFFFFFF:08X}",
                score=1.0,
                method=method,
            )

        mta_queries = unique_strings((material_key, part_name, base_name, *extra_queries))
        for query in mta_queries:
            group = self.resolve_mta_query(query)
            if group:
                return ResolvedTexture(group=group, material_key=query, score=1.0, method="mta-key")

        exact_queries = unique_strings((material_key, part_name, base_name, *extra_queries))

        for query in exact_queries:
            for alias in texture_query_aliases(query):
                stem_key, _role = split_texture_role(alias)
                group = self.groups.get(stem_key.lower())
                if group:
                    method = "exact" if alias == query else "alias"
                    return ResolvedTexture(group=group, material_key=query, score=1.0, method=method)

        scored_queries = []
        if material_key and not is_generic_texture_query(material_key):
            scored_queries.append(material_key)
        if part_name and not is_generic_texture_query(part_name):
            scored_queries.append(part_name)
        if base_name and not is_generic_texture_query(base_name):
            scored_queries.append(base_name)
        scored_queries.append(f"{base_name} {part_name}")
        scored_queries.extend(extra_queries)
        if extra_queries:
            scored_queries.append(" ".join((base_name, part_name, *extra_queries)))
        scored_queries = unique_strings(scored_queries)

        best_mta: Tuple[float, str, str] | None = None
        for query in scored_queries:
            if not is_useful_material_query(query):
                continue
            for descriptor in self.mta_descriptors.values():
                stem = os.path.splitext(descriptor.name)[0]
                score = texture_name_score(query, stem)
                if best_mta is None or score > best_mta[0]:
                    best_mta = (score, stem, query)

        if best_mta and best_mta[0] >= TEXTURE_MTA_MIN_SCORE:
            group = self.group_from_mta(best_mta[1])
            if group:
                return ResolvedTexture(
                    group=group,
                    material_key=best_mta[2],
                    score=best_mta[0],
                    method="mta-scored",
                )

        best: Tuple[float, TextureGroup, str] | None = None
        for query in scored_queries:
            for group in self.groups.values():
                score = texture_name_score(query, group.key)
                if best is None or score > best[0]:
                    best = (score, group, query)

        if best and best[0] >= TEXTURE_MIN_SCORE:
            return ResolvedTexture(group=best[1], material_key=best[2], score=best[0], method="scored")
        return None

    def image_path_for_asset(self, asset: TextureAsset) -> str | None:
        if asset.png_path and os.path.exists(asset.png_path):
            return asset.png_path
        if not self.convert_tdt or not asset.tdt_path or not os.path.exists(asset.tdt_path):
            return None

        output = os.path.join(self.cache_dir, asset.stem + ".png")
        if not os.path.exists(output):
            convert_tdt_to_png(asset.tdt_path, output)
        asset.png_path = output
        return output

    def image_for_asset(self, asset: TextureAsset, colorspace: str) -> bpy.types.Image | None:
        path = self.image_path_for_asset(asset)
        if not path:
            return None
        try:
            image = bpy.data.images.load(path, check_existing=True)
        except Exception:
            return None
        try:
            image.colorspace_settings.name = colorspace
        except Exception:
            pass
        return image

    def asset_has_non_opaque_alpha(self, asset: TextureAsset) -> bool:
        probe_path = asset.tdt_path or asset.png_path or asset.tex_path or asset.stem
        if probe_path in self.alpha_probe_cache:
            return self.alpha_probe_cache[probe_path]

        has_alpha = False
        if asset.tdt_path and os.path.exists(asset.tdt_path):
            try:
                has_alpha = tdt_top_mip_has_non_opaque_alpha(asset.tdt_path)
            except Exception:
                # DA/DAT commonly carry cutout alpha; keep that fallback for
                # loose files where the TDT payload is not available.
                has_alpha = asset.role in {"DA", "DAT"}
        else:
            has_alpha = asset.role in {"DA", "DAT"}

        self.alpha_probe_cache[probe_path] = has_alpha
        return has_alpha

    @staticmethod
    def _try_set_enum(target, attr: str, values: Sequence[str]) -> None:
        if not hasattr(target, attr):
            return
        for value in values:
            try:
                setattr(target, attr, value)
                return
            except Exception:
                continue

    def configure_material_alpha(self, material: bpy.types.Material, mode: str) -> None:
        if mode == "clip":
            self._try_set_enum(material, "blend_method", ("CLIP", "HASHED", "BLEND"))
            self._try_set_enum(material, "surface_render_method", ("DITHERED", "BLENDED"))
            if hasattr(material, "alpha_threshold"):
                material.alpha_threshold = 0.5
        elif mode == "blend":
            self._try_set_enum(material, "blend_method", ("BLEND", "HASHED"))
            self._try_set_enum(material, "surface_render_method", ("BLENDED", "DITHERED"))

        if hasattr(material, "show_transparent_back"):
            material.show_transparent_back = False

    def material_for_resolution(self, resolved: ResolvedTexture) -> bpy.types.Material | None:
        cache_key = resolved.group.key.lower()
        if cache_key in self.materials:
            return self.materials[cache_key]

        material = bpy.data.materials.new(resolved.group.key)
        material.use_nodes = True
        self._try_set_enum(material, "blend_method", ("OPAQUE",))
        material["geo_texture_key"] = resolved.material_key
        material["geo_texture_match"] = resolved.method
        material["geo_texture_score"] = resolved.score
        material["geo_texture_alpha_mode"] = self.alpha_mode
        material["geo_texture_roles"] = ",".join(sorted(resolved.group.assets))
        if resolved.group.shader_name:
            material["geo_mta_shader"] = resolved.group.shader_name
        if resolved.group.primary_refs:
            material["geo_mta_primary_refs"] = ",".join(f"{ref & 0xffffffff:08X}" for ref in resolved.group.primary_refs)
        material["geo_texture_encoded_roles"] = ",".join(
            role for role in ("ENC", "ENCA", "RGBAO", "AO") if role in resolved.group.assets
        )

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            self.materials[cache_key] = material
            return material

        color_asset = self.best_usable_asset(resolved.group, TEXTURE_COLOR_ROLES)
        if color_asset:
            material["geo_texture_color_role"] = color_asset.role or "COLOR"
            color_image = self.image_for_asset(color_asset, "sRGB")
            base_color_input = bsdf.inputs.get("Base Color")
            if color_image and base_color_input:
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.label = color_asset.role or "Color"
                tex_node.image = color_image
                links.new(tex_node.outputs["Color"], base_color_input)
                alpha_output = tex_node.outputs.get("Alpha")
                alpha_input = bsdf.inputs.get("Alpha")
                has_alpha = self.asset_has_non_opaque_alpha(color_asset)
                if (
                    self.alpha_mode != "opaque"
                    and alpha_output
                    and alpha_input
                    and has_alpha
                ):
                    links.new(alpha_output, alpha_input)
                    material["geo_texture_alpha_source"] = color_asset.role or color_asset.stem
                    material["geo_texture_alpha_detected"] = True
                    self.configure_material_alpha(material, self.alpha_mode)

        normal_asset = self.best_usable_asset(resolved.group, TEXTURE_NORMAL_ROLES)
        if normal_asset:
            material["geo_texture_normal_role"] = normal_asset.role or "Normal"
            normal_image = self.image_for_asset(normal_asset, "Non-Color")
            normal_input = bsdf.inputs.get("Normal")
            if normal_image and normal_input:
                normal_tex = nodes.new("ShaderNodeTexImage")
                normal_tex.label = normal_asset.role or "Normal"
                normal_tex.image = normal_image
                normal_node = nodes.new("ShaderNodeNormalMap")
                links.new(normal_tex.outputs["Color"], normal_node.inputs["Color"])
                links.new(normal_node.outputs["Normal"], normal_input)

        self.materials[cache_key] = material
        return material

    def best_usable_asset(self, group: TextureGroup, roles: Sequence[str]) -> TextureAsset | None:
        for role in roles:
            asset = group.assets.get(role)
            if asset and self.image_path_for_asset(asset):
                return asset
        return None
