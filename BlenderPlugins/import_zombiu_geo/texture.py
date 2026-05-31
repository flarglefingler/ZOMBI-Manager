"""loose texture resolving and blender material setup."""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import bpy

from .tdt import convert_tdt_to_png

TEXTURE_ROLE_SUFFIXES = ("RGBAO", "ENCA", "ENC", "NM", "DA", "D")
TEXTURE_COLOR_ROLES = ("D", "DA", "ENC", "ENCA")
TEXTURE_NORMAL_ROLES = ("NM",)
TEXTURE_MIN_SCORE = 0.62
TEXTURE_GENERIC_KEYS = {"body", "head", "hair", "eye", "eyes"}
TEXTURE_TOKEN_STOPWORDS = {
    "ch", "gen", "com", "skin", "misc", "drt", "lod", "pc", "h", "ca",
    "af", "as", "z", "reg", "med", "00", "01", "0", "1", "2", "3", "4", "5",
}
TEXTURE_TOKEN_ALIASES = {
    "nrs": "nurse",
    "trd": "trader",
    "lo": "low",
    "up": "upper",
    "m": "male",
    "f": "female",
    "suit": "uniform",
    "pants": "pant",
    "eyes": "eye",
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


@dataclass
class TextureGroup:
    key: str
    assets: dict[str, TextureAsset]


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
    return unique_existing_dirs([
        os.path.join(parent_dir, "converted_textures"),
        os.path.join(geo_dir, "converted_textures"),
        geo_dir,
        parent_dir,
    ])


def texture_cache_dir(geo_path: str) -> str:
    geo_dir = os.path.dirname(os.path.abspath(geo_path))
    parent_dir = os.path.dirname(geo_dir)
    for candidate in (
        os.path.join(parent_dir, "converted_textures"),
        os.path.join(geo_dir, "converted_textures"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(parent_dir, "converted_textures")


# exact geo material keys win; name scoring only fills gaps.
class TextureResolver:
    def __init__(self, geo_path: str, convert_tdt: bool):
        self.geo_path = geo_path
        self.convert_tdt = convert_tdt
        self.cache_dir = texture_cache_dir(geo_path)
        self.groups: dict[str, TextureGroup] = {}
        self.materials: dict[str, bpy.types.Material] = {}
        self.scan()

    def scan(self) -> None:
        assets_by_stem: dict[str, TextureAsset] = {}
        for directory in texture_search_dirs(self.geo_path):
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
                    companion = os.path.join(directory, stem + ".PC.tdt")
                    if os.path.exists(companion):
                        asset.tdt_path = asset.tdt_path or companion

        for asset in assets_by_stem.values():
            role_key = asset.role or "COLOR"
            group = self.groups.setdefault(asset.group_key.lower(), TextureGroup(asset.group_key, {}))
            existing = group.assets.get(role_key)
            if existing is None or (not existing.png_path and asset.png_path):
                group.assets[role_key] = asset

    def resolve(self, base_name: str, part_name: str) -> ResolvedTexture | None:
        material_key = extract_material_key(part_name)
        exact_queries = [query for query in (material_key, part_name, base_name) if query]

        for query in exact_queries:
            stem_key, _role = split_texture_role(query)
            group = self.groups.get(stem_key.lower())
            if group:
                return ResolvedTexture(group=group, material_key=query, score=1.0, method="exact")

        if material_key:
            key_tokens = set(texture_tokens(material_key))
            if len(key_tokens) <= 1 and key_tokens & TEXTURE_GENERIC_KEYS:
                return None
            query = material_key
        else:
            query = f"{base_name} {part_name}"

        best: Tuple[float, TextureGroup] | None = None
        for group in self.groups.values():
            score = texture_name_score(query, group.key)
            if best is None or score > best[0]:
                best = (score, group)

        if best and best[0] >= TEXTURE_MIN_SCORE:
            return ResolvedTexture(group=best[1], material_key=query, score=best[0], method="scored")
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

    def material_for_resolution(self, resolved: ResolvedTexture) -> bpy.types.Material | None:
        cache_key = resolved.group.key.lower()
        if cache_key in self.materials:
            return self.materials[cache_key]

        material = bpy.data.materials.new(resolved.group.key)
        material.use_nodes = True
        material["geo_texture_key"] = resolved.material_key
        material["geo_texture_match"] = resolved.method
        material["geo_texture_score"] = resolved.score

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            self.materials[cache_key] = material
            return material

        color_asset = self.best_usable_asset(resolved.group, TEXTURE_COLOR_ROLES)
        if color_asset:
            color_image = self.image_for_asset(color_asset, "sRGB")
            base_color_input = bsdf.inputs.get("Base Color")
            if color_image and base_color_input:
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.label = color_asset.role or "Color"
                tex_node.image = color_image
                links.new(tex_node.outputs["Color"], base_color_input)
                alpha_output = tex_node.outputs.get("Alpha")
                alpha_input = bsdf.inputs.get("Alpha")
                if alpha_output and alpha_input and color_asset.role in {"DA", "ENCA"}:
                    links.new(alpha_output, alpha_input)
                    material.blend_method = "BLEND"
                    if hasattr(material, "use_screen_refraction"):
                        material.use_screen_refraction = True

        normal_asset = self.best_usable_asset(resolved.group, TEXTURE_NORMAL_ROLES)
        if normal_asset:
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
