from __future__ import annotations

import hashlib
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import bfz_archive, game_index


@dataclass(frozen=True)
class ResourceFile:
    archive_path: str
    archive_name: str
    archive_kind: str
    archive_priority: int
    entry_index: int
    key: int
    name: str
    normalized_path: str
    extension: str
    size: int

    @property
    def key_hex(self) -> str:
        return bfz_archive.key_hex(self.key)


class GameResourceIndex:
    def __init__(self):
        self.archives: Dict[str, bfz_archive.BfzArchive] = {}
        self.archive_export_paths: Dict[str, Dict[int, str]] = {}
        self.resources: List[ResourceFile] = []
        self.by_key: Dict[int, List[ResourceFile]] = {}
        self.by_archive_path: Dict[str, List[ResourceFile]] = {}
        self.by_archive_name: Dict[Tuple[str, str], List[ResourceFile]] = {}

    def mount_archive(self, path: str, kind: str, priority: int) -> None:
        archive_path = os.path.abspath(path)
        if archive_path in self.archives:
            return

        archive = bfz_archive.BfzArchive(archive_path)
        archive.parse(decompress=False)
        export_paths = archive.export_path_map()
        self.archives[archive_path] = archive
        self.archive_export_paths[archive_path] = export_paths

        for entry in archive.file_entries:
            normalized = bfz_archive.normalized_archive_path(entry.name)
            resource = ResourceFile(
                archive_path=archive_path,
                archive_name=os.path.basename(archive_path),
                archive_kind=kind,
                archive_priority=priority,
                entry_index=entry.index,
                key=entry.key & 0xFFFFFFFF,
                name=os.path.basename(normalized),
                normalized_path=normalized,
                extension=entry.extension,
                size=entry.size,
            )
            self.resources.append(resource)
            self.by_key.setdefault(resource.key, []).append(resource)
            self.by_archive_path.setdefault(archive_path, []).append(resource)
            self.by_archive_name.setdefault((archive_path, normalized.lower()), []).append(resource)

    def entries_for_archive(self, archive_path: str, extension: str | None = None) -> List[ResourceFile]:
        archive_path = os.path.abspath(archive_path)
        resources = self.by_archive_path.get(archive_path, [])
        if extension is None:
            return list(resources)
        extension = extension.lower()
        return [resource for resource in resources if resource.extension == extension]

    def archive_paths_for_kinds(self, kinds: Iterable[str]) -> List[str]:
        wanted = set(kinds)
        result: List[str] = []
        for path, archive in self.archives.items():
            resources = self.by_archive_path.get(path, ())
            kind = resources[0].archive_kind if resources else ""
            if kind in wanted:
                result.append(path)
        return result

    def resolve_key(
        self,
        key: int,
        extensions: Optional[Iterable[str]] = None,
        prefer_archive_path: str | None = None,
    ) -> List[ResourceFile]:
        candidates = list(self.by_key.get(key & 0xFFFFFFFF, ()))
        if extensions is not None:
            allowed = {extension.lower() for extension in extensions}
            candidates = [resource for resource in candidates if resource.extension in allowed]

        prefer_archive_path = os.path.abspath(prefer_archive_path) if prefer_archive_path else ""

        def sort_key(resource: ResourceFile) -> Tuple[int, int, str, int]:
            preferred = 0 if prefer_archive_path and resource.archive_path == prefer_archive_path else 1
            return (preferred, resource.archive_priority, resource.archive_name.lower(), resource.entry_index)

        candidates.sort(key=sort_key)
        return candidates

    def resources_with_extensions(self, extensions: Iterable[str]) -> List[ResourceFile]:
        allowed = {extension.lower() for extension in extensions}
        return [resource for resource in self.resources if resource.extension in allowed]

    def build_content_key_index(
        self,
        keys: Iterable[int],
        extensions: Iterable[str],
        prefer_archive_path: str | None = None,
        archive_paths: Iterable[str] | None = None,
    ) -> Dict[int, List[ResourceFile]]:
        key_values = {key & 0xFFFFFFFF for key in keys}
        if not key_values:
            return {}

        allowed_archives = None
        if archive_paths is not None:
            allowed_archives = {os.path.abspath(path) for path in archive_paths}

        packed_to_key = {struct.pack("<I", key): key for key in key_values}
        pattern = re.compile(b"|".join(re.escape(value) for value in packed_to_key))
        result: Dict[int, List[ResourceFile]] = {}
        for resource in self.resources_with_extensions(extensions):
            if allowed_archives is not None and resource.archive_path not in allowed_archives:
                continue
            try:
                data = self.read(resource)
            except Exception:
                continue
            seen_in_resource: set[int] = set()
            for match in pattern.finditer(data):
                key = packed_to_key.get(match.group(0))
                if key is None or key in seen_in_resource:
                    continue
                seen_in_resource.add(key)
                result.setdefault(key, []).append(resource)

        prefer_archive_path = os.path.abspath(prefer_archive_path) if prefer_archive_path else ""

        def sort_key(resource: ResourceFile) -> Tuple[int, int, str, int]:
            preferred = 0 if prefer_archive_path and resource.archive_path == prefer_archive_path else 1
            return (preferred, resource.archive_priority, resource.archive_name.lower(), resource.entry_index)

        for resources in result.values():
            resources.sort(key=sort_key)
        return result

    def companion_files(self, resource: ResourceFile, extensions: Sequence[str]) -> List[ResourceFile]:
        base, _old_extension = os.path.splitext(resource.normalized_path)
        directory, stem = os.path.split(base)
        result: List[ResourceFile] = []
        for extension in extensions:
            lookup = (resource.archive_path, (base + extension).lower())
            exact = list(self.by_archive_name.get(lookup, ()))
            result.extend(exact)
            if exact:
                continue

            # Some world sidecars keep the same object stem plus a saved key
            # tag, for example Foo_3.obj -> Foo_3($ADD).mtn.
            extension = extension.lower()
            pattern = re.compile(
                rf"^{re.escape(stem)}\(\$[0-9A-Fa-f]+\){re.escape(extension)}$",
                flags=re.IGNORECASE,
            )
            for candidate in self.by_archive_path.get(resource.archive_path, ()):
                if candidate.extension != extension:
                    continue
                candidate_dir, candidate_name = os.path.split(candidate.normalized_path)
                if candidate_dir.lower() != directory.lower():
                    continue
                if pattern.match(candidate_name):
                    result.append(candidate)
        return result

    def read(self, resource: ResourceFile) -> bytes:
        archive = self.archives[resource.archive_path]
        entry = archive.file_entries[resource.entry_index]
        return archive.read_entry_on_demand(entry)

    def extract(self, resource: ResourceFile, cache_root: str | None = None) -> str:
        cache_root = cache_root or default_resource_cache_dir()
        archive = self.archives[resource.archive_path]
        export_paths = self.archive_export_paths[resource.archive_path]
        archive_dir = archive_cache_name(resource.archive_path)
        relative = export_paths.get(resource.entry_index, resource.normalized_path)
        output_path = os.path.join(cache_root, archive_dir, relative)
        if os.path.exists(output_path) and os.path.getsize(output_path) == resource.size:
            return output_path

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(self.read(resource))
        return output_path


def archive_cache_name(path: str) -> str:
    stat = os.stat(path)
    identity = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", "replace")
    digest = hashlib.sha1(identity).hexdigest()[:12]
    stem = os.path.basename(path)
    for suffix in (".lin.bfz", ".bfz"):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return f"{stem}_{digest}"


def default_resource_cache_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "zombiu_blender_resource_cache")


def iter_startup_archives(data_dir: str, include_common: bool = True, include_sound: bool = False, include_video: bool = False) -> Iterable[Tuple[str, str, int]]:
    priority = 20
    for path in game_index.iter_bfz_paths(data_dir):
        name = os.path.basename(path).lower()
        if include_common and name.startswith("gen_common"):
            yield path, "Common", priority
            priority += 1
        elif include_sound and (name.startswith("snd") or name.startswith("sound")):
            yield path, "Sound", priority
            priority += 1
        elif include_video and name.startswith("video"):
            yield path, "Video", priority
            priority += 1


def iter_world_sibling_archives(world_archive_path: str) -> Iterable[str]:
    directory = os.path.dirname(os.path.abspath(world_archive_path))
    name = os.path.basename(world_archive_path)
    match = re.match(r"^(.*_)[0-9]+((?:\.lin)?\.bfz)$", name, flags=re.IGNORECASE)
    if not match:
        return

    prefix, suffix = match.groups()
    selected = os.path.abspath(world_archive_path)
    for candidate_name in sorted(os.listdir(directory), key=str.lower):
        if not candidate_name.lower().endswith(suffix.lower()):
            continue
        if not candidate_name.lower().startswith(prefix.lower()):
            continue
        if not re.match(rf"^{re.escape(prefix)}[0-9]+{re.escape(suffix)}$", candidate_name, flags=re.IGNORECASE):
            continue
        candidate = os.path.abspath(os.path.join(directory, candidate_name))
        if candidate != selected:
            yield candidate


def build_world_resource_index(
    game_dir: str,
    world_archive_path: str,
    include_common: bool = True,
    include_sound: bool = False,
    include_video: bool = False,
) -> GameResourceIndex:
    data_dir = game_index.data_dir_for_game_dir(game_dir)
    if not data_dir:
        raise ValueError("pick the ZOMBI folder or its Data folder")

    index = GameResourceIndex()
    index.mount_archive(world_archive_path, "World", 0)
    priority = 1
    for path in iter_world_sibling_archives(world_archive_path):
        index.mount_archive(path, "WorldChunk", priority)
        priority += 1
    for path, kind, priority in iter_startup_archives(data_dir, include_common, include_sound, include_video):
        if os.path.abspath(path) in index.archives:
            continue
        index.mount_archive(path, kind, priority)
    return index
