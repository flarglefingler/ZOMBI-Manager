from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


WOR_MAGIC = b"WOR_"
OBJECT_GROUP_HEADER_SIZE = 0x20
OBJECT_REF_SIZE = 12
MANIFEST_NAMES = (
    "_zombi_bfz_manifest.json",
    "zombi_bfz_manifest.json",
    "bfz_manifest.json",
)


@dataclass(frozen=True)
class WorObjectRef:
    index: int
    object_key: int
    type_flags: int
    extra: int

    @property
    def key_hex(self) -> str:
        return f"{self.object_key & 0xffffffff:08X}"

    @property
    def has_metadata(self) -> bool:
        return self.type_flags != 0 or self.extra != 0


@dataclass(frozen=True)
class WorFile:
    name: str
    version: int
    stored_size: int
    world_chunk_size: int
    object_group_size: int
    object_group_offset: int
    object_refs: List[WorObjectRef]
    group_header: bytes

    @property
    def object_count(self) -> int:
        return len(self.object_refs)


@dataclass(frozen=True)
class ManifestEntry:
    index: int
    key: int
    path: str
    relative_path: str
    size: int
    offset: int

    @property
    def extension(self) -> str:
        return os.path.splitext(self.path)[1].lower()

    @property
    def key_hex(self) -> str:
        return f"{self.key & 0xffffffff:08X}"


def _u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError(f"unexpected end of WOR at 0x{offset:x}")
    return struct.unpack_from("<I", data, offset)[0]


def parse_wor(data: bytes, name: str = "") -> WorFile:
    if len(data) < 0x1c:
        raise ValueError("file is too small to be a WOR")

    if data[0x14:0x18] != WOR_MAGIC:
        if data[:4] == WOR_MAGIC:
            return WorFile(name, _u32(data, 4), len(data), len(data), 0, len(data), [], b"")
        raise ValueError("missing WOR_ magic")

    world_chunk_size = _u32(data, 0x00)
    world_chunk_size_again = _u32(data, 0x04)
    object_group_size = _u32(data, 0x08)
    version = _u32(data, 0x18)

    if world_chunk_size != world_chunk_size_again:
        raise ValueError(
            f"WOR chunk size mismatch: 0x{world_chunk_size:x} != 0x{world_chunk_size_again:x}"
        )

    object_group_offset = world_chunk_size + OBJECT_GROUP_HEADER_SIZE
    if object_group_offset > len(data):
        raise ValueError("WOR object group starts past end of file")

    group_header = data[world_chunk_size:object_group_offset]
    group_end = min(len(data), object_group_offset + object_group_size)
    group_data = data[object_group_offset:group_end]
    refs: List[WorObjectRef] = []

    for index, offset in enumerate(range(0, len(group_data) - (OBJECT_REF_SIZE - 1), OBJECT_REF_SIZE)):
        object_key, type_flags, extra = struct.unpack_from("<III", group_data, offset)
        if object_key == 0 and type_flags == 0 and extra == 0:
            continue
        refs.append(WorObjectRef(index, object_key, type_flags, extra))

    return WorFile(
        name=name,
        version=version,
        stored_size=len(data),
        world_chunk_size=world_chunk_size,
        object_group_size=object_group_size,
        object_group_offset=object_group_offset,
        object_refs=refs,
        group_header=group_header,
    )


def manifest_key(key: int | str) -> int:
    if isinstance(key, int):
        return key & 0xffffffff
    text = key.strip()
    if text.lower().startswith("0x"):
        text = text[2:]
    return int(text, 16) & 0xffffffff


def load_bfz_manifest(path: str) -> Dict[int, List[str]]:
    entries = load_bfz_manifest_entries(path)
    result: Dict[int, List[str]] = {}
    for item in entries:
        result.setdefault(item.key, []).append(item.path)
    return result


def load_bfz_manifest_entries(path: str) -> List[ManifestEntry]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    base_dir = os.path.dirname(os.path.abspath(path))
    result: List[ManifestEntry] = []
    for index, item in enumerate(manifest.get("files", [])):
        key_text = item.get("key")
        file_path = item.get("path")
        if not key_text or not file_path:
            continue
        try:
            key = manifest_key(key_text)
        except ValueError:
            continue
        normalized = file_path.replace("\\", "/")
        result.append(
            ManifestEntry(
                index=index,
                key=key,
                path=os.path.join(base_dir, normalized),
                relative_path=normalized,
                size=int(item.get("size", 0) or 0),
                offset=int(item.get("offset", 0) or 0),
            )
        )
    return result


def find_bfz_manifest(start_path: str) -> Optional[str]:
    directory = start_path if os.path.isdir(start_path) else os.path.dirname(start_path)
    directory = os.path.abspath(directory)

    while True:
        for name in MANIFEST_NAMES:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def load_nearby_manifest(start_path: str) -> Dict[int, List[str]]:
    manifest_path = find_bfz_manifest(start_path)
    if not manifest_path:
        return {}
    return load_bfz_manifest(manifest_path)


def load_nearby_manifest_entries(start_path: str) -> List[ManifestEntry]:
    manifest_path = find_bfz_manifest(start_path)
    if not manifest_path:
        return []
    return load_bfz_manifest_entries(manifest_path)


def resolve_refs(wor: WorFile, key_map: Dict[int, List[str]]) -> Dict[int, List[str]]:
    resolved: Dict[int, List[str]] = {}
    for ref in wor.object_refs:
        paths = key_map.get(ref.object_key)
        if paths:
            resolved[ref.object_key] = paths
    return resolved


def ordered_object_entries_for_world(
    wor: WorFile,
    entries: Iterable[ManifestEntry],
) -> Dict[int, ManifestEntry]:
    object_entries = [entry for entry in entries if entry.extension == ".obj"]
    result: Dict[int, ManifestEntry] = {}
    for ref, entry in zip(wor.object_refs, object_entries):
        result[ref.index] = entry
    return result


def object_entries_for_world(
    wor: WorFile,
    entries: Iterable[ManifestEntry],
) -> Tuple[Dict[int, ManifestEntry], str]:
    entries = list(entries)
    by_key: Dict[int, List[ManifestEntry]] = {}
    for entry in entries:
        if entry.extension == ".obj":
            by_key.setdefault(entry.key, []).append(entry)

    result: Dict[int, ManifestEntry] = {}
    keyed_count = 0
    for ref in wor.object_refs:
        candidates = by_key.get(ref.object_key)
        if not candidates:
            continue
        result[ref.index] = candidates[0]
        keyed_count += 1

    # Older exports wrote sequential placeholder keys. In that case almost
    # nothing resolves by key, so keep the old order fallback.
    if keyed_count:
        ordered = ordered_object_entries_for_world(wor, entries)
        for index, entry in ordered.items():
            result.setdefault(index, entry)
        return result, "manifest-key" if keyed_count == len(result) else "manifest-key+order"

    return ordered_object_entries_for_world(wor, entries), "manifest-order"


def _clean_geo_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.lower().endswith(".pc"):
        stem = stem[:-3]
    stem = re.sub(r"__variant_[0-9]+$", "", stem, flags=re.IGNORECASE)
    return stem


def _geo_match_key(name: str) -> str:
    text = _clean_geo_stem(name)
    text = text.lower()
    if text.startswith("pfb_"):
        text = text[4:]
    text = re.sub(r"\(\$[0-9a-f]+\)", "", text)
    text = re.sub(r"lod[0-9]+$", "", text)
    text = re.sub(r"_[0-9]+$", "", text)
    text = re.sub(r"aa+$", "", text)
    text = re.sub(r"[0-9]+$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _name_terms(name: str) -> List[str]:
    stem = _clean_geo_stem(name)
    stem = re.sub(r"\(\$[0-9A-Fa-f]+\)", "", stem)
    stem = re.sub(r"lod[0-9]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_[0-9]+$", "", stem)
    parts = [part for part in re.split(r"[_\s]+", stem) if part]
    terms: List[str] = []
    for start in range(len(parts)):
        joined = "".join(parts[start:])
        joined = re.sub(r"[^A-Za-z0-9]+", "", joined).lower()
        joined = re.sub(r"[0-9]+$", "", joined)
        if len(joined) >= 5 and joined not in terms:
            terms.append(joined)
    compact = _geo_match_key(stem)
    if len(compact) >= 5 and compact not in terms:
        terms.append(compact)
    return terms

# for now until i decode more of the world format, just use some guesses
NAME_TOKEN_STOPWORDS = {
    "ach",
    "brl",
    "co",
    "di",
    "ele",
    "fa",
    "fur",
    "gen",
    "gl",
    "grd",
    "hub",
    "la",
    "lod",
    "lou",
    "lt",
    "mat",
    "me",
    "pap",
    "pc",
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
NAME_TOKEN_ALIASES = {
    "camera": "cam",
    "cardboard": "cardbox",
    "junks": "junk",
    "telephone": "phone",
    "trash": "junk",
    "trashes": "junk",
    "vents": "vent",
    "ventilation": "vent",
}

# guess stuff
def _name_tokens(name: str) -> List[str]:
    stem = _clean_geo_stem(name)
    stem = re.sub(r"\(\$[0-9A-Fa-f]+\)", "_", stem)
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem)
    stem = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", stem)
    raw_tokens = [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", stem)
        if token
    ]
    tokens: List[str] = []
    for token in raw_tokens:
        for piece in re.findall(r"[a-z]+|[0-9]+", token):
            if not piece or piece.isdigit() or piece in NAME_TOKEN_STOPWORDS:
                continue
            alias = NAME_TOKEN_ALIASES.get(piece, piece)
            if alias not in tokens:
                tokens.append(alias)
    return tokens


def _token_match_score(object_stem: str, geo_name: str) -> int:
    object_tokens = _name_tokens(object_stem)
    geo_tokens = _name_tokens(geo_name)
    if not object_tokens or not geo_tokens:
        return 0

    matches = 0
    for object_token in object_tokens:
        for geo_token in geo_tokens:
            if object_token == geo_token:
                matches += 1
                break
            if len(object_token) >= 4 and len(geo_token) >= 4 and (
                object_token in geo_token or geo_token in object_token
            ):
                matches += 1
                break

    if matches <= 0:
        return 0
    return min(85, 45 + matches * 15)


def _object_stem_variants(stem: str) -> List[str]:
    variants: List[str] = []

    def add(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    add(stem)
    if stem.startswith("PFB_"):
        add(stem[4:])
    for value in list(variants):
        add(re.sub(r"__variant_[0-9]+$", "", value, flags=re.IGNORECASE))
        add(re.sub(r"aa+$", "", value))
        add(re.sub(r"_[0-9]+$", "", value))
        add(re.sub(r"_[0-9]+\(\$[0-9A-Fa-f]+\)$", "", value))
        add(re.sub(r"\(\$[0-9A-Fa-f]+\)_[0-9]+$", "", value))
        add(re.sub(r"\(\$[0-9A-Fa-f]+\)$", "", value))
    return variants


def build_geo_name_index(directory: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return result
    for name in os.listdir(directory):
        lower = name.lower()
        if not lower.endswith(".geo"):
            continue
        result.setdefault(lower, os.path.join(directory, name))
    return result


def build_geo_key_index(
    geo_name_index: Dict[str, str],
    keys: Iterable[int],
) -> Dict[int, List[str]]:
    key_values = {key & 0xFFFFFFFF for key in keys}
    result: Dict[int, List[str]] = {}
    if not key_values:
        return result

    key_bytes = {key: struct.pack("<I", key) for key in key_values}
    for path in geo_name_index.values():
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        for key, packed in key_bytes.items():
            if packed in data:
                result.setdefault(key, []).append(path)
    return result


def _geo_variant_bias(geo_name: str, object_stem: str) -> int:
    geo_is_variant = "__variant_" in geo_name.lower()
    object_wants_variant = "__variant_" in object_stem.lower()
    if geo_is_variant and not object_wants_variant:
        return -1
    return 0


def _trailing_number(name: str) -> Optional[int]:
    stem = _clean_geo_stem(name)
    stem = re.sub(r"\(\$[0-9A-Fa-f]+\)", "", stem)
    match = re.search(r"(?:_|)([0-9]+)$", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _geo_number_bias(geo_name: str, object_stem: str) -> int:
    object_number = _trailing_number(object_stem)
    geo_number = _trailing_number(geo_name)
    if object_number is None or geo_number is None:
        return 0
    return -abs(object_number - geo_number)


def _geo_candidate_score(object_stem: str, geo_path: str) -> Tuple[int, int, int, int, int]:
    geo_name = os.path.basename(geo_path)
    stem = os.path.splitext(os.path.basename(object_stem))[0]
    variant_bias = _geo_variant_bias(geo_name, stem)
    number_bias = _geo_number_bias(geo_name, stem)

    for variant in _object_stem_variants(stem):
        for candidate in (f"{variant}.PC.geo", f"{variant}.geo"):
            if geo_name.lower() == candidate.lower():
                return (100 if variant == stem else 90, len(variant), variant_bias, number_bias, -len(_geo_match_key(geo_name)))

    object_terms = _name_terms(stem)
    geo_key = _geo_match_key(geo_path)
    best = (0, 0, variant_bias, number_bias, -len(geo_key))
    for term in object_terms:
        if len(term) < 6:
            continue
        if geo_key.endswith(term):
            best = max(best, (70, len(term), variant_bias, number_bias, -len(geo_key)))
        elif len(geo_key) >= 6 and term.endswith(geo_key):
            best = max(best, (60, len(geo_key), variant_bias, number_bias, -len(term)))
        elif len(term) >= 8 and term in geo_key:
            best = max(best, (50, len(term), variant_bias, number_bias, -len(geo_key)))

    object_key = _geo_match_key(stem)
    token_score = _token_match_score(stem, geo_name)
    if token_score > 0:
        best = max(best, (token_score, token_score, variant_bias, number_bias, -len(geo_key)))
    if object_key == geo_key:
        best = max(best, (80, len(object_key), variant_bias, number_bias, -len(geo_key)))
    elif object_key.startswith(geo_key) or geo_key.startswith(object_key):
        best = max(best, (40, min(len(object_key), len(geo_key)), variant_bias, number_bias, -len(geo_key)))
    return best


def resolve_geo_for_object_path_info(
    object_path: str,
    geo_name_index: Dict[str, str],
    geo_key_index: Optional[Dict[int, List[str]]] = None,
    sidecar_keys: Optional[Iterable[int]] = None,
) -> Tuple[Optional[str], str]:
    stem = os.path.splitext(os.path.basename(object_path))[0]
    sidecar_key_list = [key & 0xFFFFFFFF for key in sidecar_keys or ()]

    for variant in _object_stem_variants(stem):
        for candidate in (f"{variant}.PC.geo", f"{variant}.geo"):
            resolved = geo_name_index.get(candidate.lower())
            if resolved:
                return resolved, "exact" if variant == stem else "variant"

    if geo_key_index and sidecar_key_list:
        candidates: List[Tuple[Tuple[int, int, int, int, int], str, int]] = []
        for key in sidecar_key_list:
            for path in geo_key_index.get(key, []):
                candidates.append((_geo_candidate_score(stem, path), path, key))
        if candidates:
            candidates.sort(reverse=True)
            best_score, best_path, _key = candidates[0]
            if best_score[0] > 0:
                return best_path, "key" if len(candidates) == 1 else "key+name"

    object_key = _geo_match_key(stem)
    if not object_key:
        return None, ""

    object_terms = _name_terms(stem)
    term_matches: List[tuple[int, int, int, int, int, str]] = []
    for path in geo_name_index.values():
        geo_name = os.path.basename(path)
        geo_key = _geo_match_key(path)
        if not geo_key:
            continue
        variant_bias = _geo_variant_bias(geo_name, stem)
        number_bias = _geo_number_bias(geo_name, stem)
        for term in object_terms:
            if len(term) < 5:
                continue
            if geo_key.endswith(term):
                term_matches.append((3, len(term), variant_bias, number_bias, -len(geo_key), path))
                break
            if len(geo_key) >= 6 and term.endswith(geo_key):
                term_matches.append((2, len(geo_key), variant_bias, number_bias, -len(term), path))
                break
            if len(term) >= 8 and term in geo_key:
                term_matches.append((1, len(term), variant_bias, number_bias, -len(geo_key), path))
                break
    if term_matches:
        term_matches.sort(reverse=True)
        return term_matches[0][5], "name"

    return None, ""


def resolve_geo_for_object_path(object_path: str, geo_name_index: Dict[str, str]) -> Optional[str]:
    return resolve_geo_for_object_path_info(object_path, geo_name_index)[0]


def refs_to_lines(
    refs: Iterable[WorObjectRef],
    key_map: Optional[Dict[int, List[str]]] = None,
    limit: int = 300,
) -> List[str]:
    lines: List[str] = []
    key_map = key_map or {}
    refs = list(refs)
    for ref in refs[:limit]:
        paths = key_map.get(ref.object_key, [])
        target = "unresolved"
        if paths:
            target = os.path.basename(paths[0])
            if len(paths) > 1:
                target += f" (+{len(paths) - 1})"
        flags = ""
        if ref.has_metadata:
            flags = f" flags=0x{ref.type_flags:08X} extra=0x{ref.extra:08X}"
        lines.append(f"- [{ref.index:03d}] {ref.key_hex} -> {target}{flags}")
    if len(refs) > limit:
        lines.append(f"... {len(refs) - limit:,} more")
    return lines
