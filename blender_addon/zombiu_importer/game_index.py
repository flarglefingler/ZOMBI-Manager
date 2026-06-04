from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

from . import bfz_archive


@dataclass(frozen=True)
class ArchiveSummary:
    path: str
    name: str
    kind: str
    file_count: int
    world_count: int
    geo_count: int
    skn_count: int
    trl_count: int
    tex_count: int
    tdt_count: int
    mat_count: int
    mta_count: int
    first_world_name: str = ""
    error: str = ""

    @property
    def is_world(self) -> bool:
        return self.kind == "World"


def data_dir_for_game_dir(game_dir: str) -> str:
    root = os.path.abspath(os.path.expanduser(game_dir or ""))
    if not root:
        return ""
    if os.path.basename(root).lower() == "data" and os.path.isdir(root):
        return root
    data_dir = os.path.join(root, "Data")
    if os.path.isdir(data_dir):
        return data_dir
    return ""


def iter_bfz_paths(data_dir: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".bfz"):
                yield os.path.join(root, name)


def archive_kind(path: str, world_count: int) -> str:
    name = os.path.basename(path).lower()
    if name.startswith("gen_common"):
        return "Common"
    if name.startswith("snd") or name.startswith("sound"):
        return "Sound"
    if name.startswith("video"):
        return "Video"
    if name.startswith("wor_") and world_count:
        return "World"
    return "Archive"


def summarize_archive(path: str) -> ArchiveSummary:
    name = os.path.basename(path)
    try:
        archive = bfz_archive.BfzArchive(path)
        archive.parse(decompress=False)
    except Exception as exc:
        return ArchiveSummary(path, name, "Error", 0, 0, 0, 0, 0, 0, 0, 0, 0, error=str(exc))

    counts = {
        ".wor": 0,
        ".geo": 0,
        ".skn": 0,
        ".trl": 0,
        ".tex": 0,
        ".tdt": 0,
        ".mat": 0,
        ".mta": 0,
    }
    first_world_name = ""
    for entry in archive.file_entries:
        extension = entry.extension
        if extension in counts:
            counts[extension] += 1
            if extension == ".wor" and not first_world_name:
                first_world_name = bfz_archive.normalized_archive_path(entry.name)

    kind = archive_kind(path, counts[".wor"])
    return ArchiveSummary(
        path=path,
        name=name,
        kind=kind,
        file_count=len(archive.file_entries),
        world_count=counts[".wor"],
        geo_count=counts[".geo"],
        skn_count=counts[".skn"],
        trl_count=counts[".trl"],
        tex_count=counts[".tex"],
        tdt_count=counts[".tdt"],
        mat_count=counts[".mat"],
        mta_count=counts[".mta"],
        first_world_name=first_world_name,
    )


def scan_game_dir(game_dir: str) -> List[ArchiveSummary]:
    data_dir = data_dir_for_game_dir(game_dir)
    if not data_dir:
        raise ValueError("pick the ZOMBI folder or its Data folder")

    summaries = [summarize_archive(path) for path in iter_bfz_paths(data_dir)]
    return sorted(summaries, key=lambda item: (item.kind != "World", item.name.lower()))
