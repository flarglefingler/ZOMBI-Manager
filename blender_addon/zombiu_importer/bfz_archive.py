from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


BFZ_MAGIC = b"ABE"
MANIFEST_NAME = "_zombi_bfz_manifest.json"


@dataclass(frozen=True)
class BfzEntry:
    name: str
    offset: int
    size: int
    key: int
    index: int

    @property
    def extension(self) -> str:
        return os.path.splitext(self.name)[1].lower()


@dataclass(frozen=True)
class BfzChunk:
    new_offset: int
    offset: int
    size: int
    compressed_size: int


def _u32(handle) -> int:
    return struct.unpack("<I", handle.read(4))[0]


def _u64(handle) -> int:
    return struct.unpack("<Q", handle.read(8))[0]


def _fixed_string(handle, length: int) -> str:
    raw = handle.read(length)
    raw = raw.split(b"\x00", 1)[0]
    return raw.decode("utf-8", "replace")


def _load_lzo_backend():
    bundled_path = os.path.join(os.path.dirname(__file__), "dissect", "util", "compression", "lzo.py")
    if os.path.exists(bundled_path):
        try:
            module_name = f"{__package__ or 'zombiu_importer'}.bundled_dissect_lzo"
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(module_name, bundled_path)
                if spec is None or spec.loader is None:
                    raise ImportError("could not load bundled dissect lzo module")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            def decompress(data: bytes, expected_size: int) -> bytes:
                return module.decompress(data, header=False, buflen=expected_size)

            return decompress, "bundled dissect.util lzo", ""
        except Exception as exc:
            bundled_error = str(exc)
    else:
        bundled_error = "bundled dissect/util/compression/lzo.py not found"

    try:
        from dissect.util.compression import lzo as dissect_lzo

        def decompress(data: bytes, expected_size: int) -> bytes:
            return dissect_lzo.decompress(data, header=False, buflen=expected_size)

        return decompress, "dissect.util", ""
    except Exception as exc:
        dissect_error = str(exc)

    try:
        import lzo as python_lzo

        def decompress(data: bytes, expected_size: int) -> bytes:
            return python_lzo.decompress(data, False, expected_size)

        return decompress, "python-lzo", ""
    except Exception as exc:
        return None, "", f"bundled dissect.util: {bundled_error}; dissect.util: {dissect_error}; python-lzo: {exc}"


def require_lzo_backend():
    decompress, backend_name, error = _load_lzo_backend()
    if decompress is None:
        raise RuntimeError(
            "No LZO backend is available. The add-on first tries its bundled "
            "dissect.util LZO decoder, then Blender-installed dissect.util, "
            "then python-lzo. Last errors: "
            f"{error}"
        )
    return decompress, backend_name


def key_hex(key: int) -> str:
    return f"{key & 0xFFFFFFFF:08X}"


def normalized_archive_path(name: str) -> str:
    return name.replace("\\", "/").strip("/")


def variant_path(path: str, variant_index: int) -> str:
    if variant_index <= 1:
        return path

    directory, filename = os.path.split(path)
    lower = filename.lower()
    compound_suffix = next(
        (
            filename[-len(suffix):]
            for suffix in (".PC.geo", ".PC.tdt", ".PC.trl", ".PC.son")
            if lower.endswith(suffix.lower())
        ),
        "",
    )
    if compound_suffix:
        stem = filename[:-len(compound_suffix)]
        extension = compound_suffix
    else:
        stem, extension = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}__variant_{variant_index}{extension}").replace("\\", "/")


def archive_cache_dir(path: str, base_dir: Optional[str] = None) -> str:
    stat = os.stat(path)
    identity = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", "replace")
    digest = hashlib.sha1(identity).hexdigest()[:12]
    stem = os.path.basename(path)
    stem = re.sub(r"\.lin\.bfz$|\.bfz$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "world"
    root = base_dir or tempfile.gettempdir()
    return os.path.join(root, "zombiu_bfz_world_cache", f"{stem}_{digest}")


class BfzArchive:
    def __init__(self, path: str):
        self.path = path
        self.file_entries: List[BfzEntry] = []
        self.chunks: List[BfzChunk] = []
        self.real_chunk_count = 0
        self.memory: Optional[bytearray] = None
        self.lzo_backend = ""
        self._chunk_cache: Dict[int, bytes] = {}

    def parse(self, decompress: bool = True) -> None:
        with open(self.path, "rb") as handle:
            magic = handle.read(3)
            if magic != BFZ_MAGIC:
                raise ValueError("invalid BFZ magic, expected ABE")

            handle.seek(0x28)
            files_offset = _u64(handle)
            _folders_offset = _u64(handle)
            chunks_offset = _u64(handle)

            _file_count_header = _u32(handle)
            _folder_count = _u32(handle)
            _chunk_count_header = _u32(handle)
            _file_count_again = _u32(handle)
            _dummy = _u32(handle)
            self.real_chunk_count = _u32(handle)

            handle.seek(files_offset)
            file_count = _u32(handle)
            _ = _u32(handle)
            _ = _u64(handle)

            offsets: List[int] = []
            sizes: List[int] = []
            for _index in range(file_count):
                offsets.append(_u64(handle))
                sizes.append(_u64(handle))
                _ = _u32(handle)
                _ = _u32(handle)

            names: List[str] = []
            keys: List[int] = []
            for _index in range(file_count):
                names.append(_fixed_string(handle, 0x40))
                _ = _u64(handle)
                _ = _u64(handle)
                _ = _u64(handle)
                keys.append(_u32(handle))
                _ = _u32(handle)

            self.file_entries = [
                BfzEntry(names[index], offsets[index], sizes[index], keys[index], index)
                for index in range(file_count)
            ]

            handle.seek(chunks_offset)
            _chunks_count = _u32(handle)
            _ = _u32(handle)
            _ = _u64(handle)
            chunk_table_start = handle.tell()

            self.chunks = []
            max_offset = 0
            for _index in range(self.real_chunk_count):
                new_offset = _u64(handle)
                offset = _u64(handle)
                _ = _u64(handle)
                size = _u32(handle)
                compressed_size = _u32(handle)
                self.chunks.append(BfzChunk(new_offset, offset, size, compressed_size))
                max_offset = max(max_offset, new_offset + size)

            if not decompress:
                self.memory = None
                return

            lzo_decompress, backend_name = require_lzo_backend()
            self.lzo_backend = backend_name
            self.memory = bytearray(max_offset)

            handle.seek(chunk_table_start)
            for chunk in self.chunks:
                handle.seek(chunk.offset)
                compressed = handle.read(chunk.compressed_size)
                payload = lzo_decompress(compressed, chunk.size)
                if len(payload) != chunk.size:
                    raise RuntimeError(
                        f"BFZ chunk at 0x{chunk.offset:x} decompressed to {len(payload)} bytes, "
                        f"expected {chunk.size}"
                    )
                self.memory[chunk.new_offset:chunk.new_offset + chunk.size] = payload

    def read_entry(self, entry: BfzEntry) -> bytes:
        if self.memory is None:
            raise RuntimeError("BFZ archive has not been decompressed")
        end = entry.offset + entry.size
        if end > len(self.memory):
            raise RuntimeError(f"BFZ entry {entry.name} points outside the decompressed buffer")
        return bytes(self.memory[entry.offset:end])

    def read_entry_on_demand(self, entry: BfzEntry) -> bytes:
        if not self.file_entries or not self.chunks:
            self.parse(decompress=False)
        if self.memory is not None:
            return self.read_entry(entry)

        entry_start = entry.offset
        entry_end = entry.offset + entry.size
        if entry.size <= 0:
            return b""

        lzo_decompress, backend_name = require_lzo_backend()
        self.lzo_backend = backend_name
        output = bytearray(entry.size)
        copied = 0

        with open(self.path, "rb") as handle:
            for index, chunk in enumerate(self.chunks):
                chunk_start = chunk.new_offset
                chunk_end = chunk.new_offset + chunk.size
                if chunk_end <= entry_start or chunk_start >= entry_end:
                    continue

                payload = self._chunk_cache.get(index)
                if payload is None:
                    handle.seek(chunk.offset)
                    compressed = handle.read(chunk.compressed_size)
                    payload = lzo_decompress(compressed, chunk.size)
                    if len(payload) != chunk.size:
                        raise RuntimeError(
                            f"BFZ chunk at 0x{chunk.offset:x} decompressed to {len(payload)} bytes, "
                            f"expected {chunk.size}"
                        )
                    self._chunk_cache[index] = payload

                src_start = max(entry_start, chunk_start) - chunk_start
                src_end = min(entry_end, chunk_end) - chunk_start
                dst_start = max(entry_start, chunk_start) - entry_start
                payload_slice = payload[src_start:src_end]
                output[dst_start:dst_start + len(payload_slice)] = payload_slice
                copied += len(payload_slice)

        if copied != entry.size:
            raise RuntimeError(f"BFZ entry {entry.name} could only read {copied}/{entry.size} bytes")
        return bytes(output)

    def export_path_map(self, entries: Optional[Iterable[BfzEntry]] = None) -> Dict[int, str]:
        entries = list(entries or self.file_entries)
        duplicate_counts: Dict[str, int] = {}
        for entry in entries:
            path = normalized_archive_path(entry.name)
            duplicate_counts[path.lower()] = duplicate_counts.get(path.lower(), 0) + 1

        seen: Dict[str, int] = {}
        result: Dict[int, str] = {}
        for entry in entries:
            path = normalized_archive_path(entry.name)
            path_key = path.lower()
            seen[path_key] = seen.get(path_key, 0) + 1
            result[entry.index] = variant_path(path, seen[path_key]) if duplicate_counts[path_key] > 1 else path
        return result

    def build_manifest(self, exported_paths: Optional[Dict[int, str]] = None) -> dict:
        exported_paths = exported_paths or self.export_path_map()
        stat = os.stat(self.path)
        return {
            "format": "zombi-manager-bfz-manifest",
            "version": 2,
            "archive": os.path.basename(self.path),
            "archive_path": os.path.abspath(self.path),
            "archive_size": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "lzo_backend": self.lzo_backend,
            "files": [
                {
                    "key": key_hex(entry.key),
                    "path": exported_paths.get(entry.index, normalized_archive_path(entry.name)),
                    "archive_path": normalized_archive_path(entry.name),
                    "index": entry.index,
                    "offset": entry.offset,
                    "size": entry.size,
                }
                for entry in self.file_entries
            ],
        }

    def cache_is_complete(self, output_dir: str, exported_paths: Optional[Dict[int, str]] = None) -> bool:
        manifest_path = os.path.join(output_dir, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            return False

        stat = os.stat(self.path)
        if manifest.get("archive_size") != stat.st_size or manifest.get("archive_mtime_ns") != stat.st_mtime_ns:
            return False
        if len(manifest.get("files", [])) != len(self.file_entries):
            return False

        exported_paths = exported_paths or self.export_path_map()
        for entry in self.file_entries:
            candidate = os.path.join(output_dir, exported_paths[entry.index])
            if not os.path.exists(candidate) or os.path.getsize(candidate) != entry.size:
                return False
        return True

    def export_all(self, output_dir: str, refresh: bool = False) -> Tuple[str, Dict[int, str]]:
        if not self.file_entries:
            self.parse(decompress=False)

        exported_paths = self.export_path_map()
        os.makedirs(output_dir, exist_ok=True)
        if not refresh and self.cache_is_complete(output_dir, exported_paths):
            return os.path.join(output_dir, MANIFEST_NAME), exported_paths

        if self.memory is None:
            self.parse(decompress=True)

        for entry in self.file_entries:
            relative_path = exported_paths[entry.index]
            output_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as handle:
                handle.write(self.read_entry(entry))

        manifest_path = os.path.join(output_dir, MANIFEST_NAME)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self.build_manifest(exported_paths), handle, indent=2)
            handle.write("\n")
        return manifest_path, exported_paths

    def wor_entries(self) -> List[BfzEntry]:
        return [entry for entry in self.file_entries if entry.name.lower().endswith(".wor")]
