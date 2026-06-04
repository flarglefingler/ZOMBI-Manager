import json
import os
from dataclasses import dataclass
from utilities import binaryHelpers, lzo_backend
from typing import Dict, Iterable, List, Optional
try:
    from PySide6.QtWidgets import QApplication, QProgressDialog
except ModuleNotFoundError:
    QApplication = None
    QProgressDialog = object

# ----------------- Data classes -----------------
@dataclass
class BFZFileEntry:
    name: str
    offset: int
    size: int
    key: int = 0
    index: int = -1


@dataclass
class BFZArchiveInfo:
    path: str
    relative_path: str
    file_count: int
    archive_size: int
    unpacked_size: int
    key_count: int
    duplicate_name_count: int
    extension_counts: Dict[str, int]
    world_names: List[str]

    @property
    def display_name(self) -> str:
        return self.world_names[0] if self.world_names else os.path.basename(self.path)

    @property
    def has_world(self) -> bool:
        return bool(self.world_names)


def _read_file_table(f) -> List[BFZFileEntry]:
    f.seek(0x28)
    files_off = binaryHelpers.read_u64_le(f)
    _folders_off = binaryHelpers.read_u64_le(f)
    _chunks_off = binaryHelpers.read_u64_le(f)

    f.seek(files_off)
    files_count = binaryHelpers.read_u32_le(f)
    _ = binaryHelpers.read_u32_le(f)
    _ = binaryHelpers.read_u64_le(f)

    offsets, sizes, names, keys = [], [], [], []
    for _i in range(files_count):
        offset = binaryHelpers.read_u64_le(f)
        size = binaryHelpers.read_u64_le(f)
        _d1 = binaryHelpers.read_u32_le(f)
        _d2 = binaryHelpers.read_u32_le(f)
        offsets.append(offset)
        sizes.append(size)

    for _i in range(files_count):
        name = binaryHelpers.read_fixed_string(f, 0x40)
        _size = binaryHelpers.read_u64_le(f)
        _zero = binaryHelpers.read_u64_le(f)
        _d3 = binaryHelpers.read_u64_le(f)
        key = binaryHelpers.read_u32_le(f)
        _d4 = binaryHelpers.read_u32_le(f)
        names.append(name)
        keys.append(key)

    return [
        BFZFileEntry(names[i], offsets[i], sizes[i], keys[i], i)
        for i in range(files_count)
    ]


def scan_archive_info(path: str, base_dir: str = "") -> BFZArchiveInfo:
    with open(path, "rb") as f:
        magic = f.read(3)
        if magic != b"ABE":
            raise ValueError("Invalid BFZ magic (expected 'ABE')")
        entries = _read_file_table(f)

    archive_size = os.path.getsize(path)
    unpacked_size = sum(entry.size for entry in entries)
    names_seen: Dict[str, int] = {}
    extension_counts: Dict[str, int] = {}
    world_names: List[str] = []
    keys = set()

    for entry in entries:
        normalized = BFZArchive.normalized_archive_path(entry.name)
        lower_name = normalized.lower()
        names_seen[lower_name] = names_seen.get(lower_name, 0) + 1
        if entry.key:
            keys.add(entry.key)

        extension = os.path.splitext(lower_name)[1] or "(none)"
        if lower_name.endswith(".pc.geo"):
            extension = ".geo"
        elif lower_name.endswith(".pc.tdt"):
            extension = ".tdt"
        elif lower_name.endswith(".pc.trl"):
            extension = ".trl"
        elif lower_name.endswith(".pc.son"):
            extension = ".son"
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

        if lower_name.endswith(".wor"):
            world_names.append(os.path.basename(normalized))

    duplicate_name_count = sum(count - 1 for count in names_seen.values() if count > 1)
    relative_path = os.path.relpath(path, base_dir) if base_dir else os.path.basename(path)
    if relative_path.startswith(".."):
        relative_path = os.path.basename(path)

    return BFZArchiveInfo(
        path=os.path.abspath(path),
        relative_path=relative_path.replace("\\", "/"),
        file_count=len(entries),
        archive_size=archive_size,
        unpacked_size=unpacked_size,
        key_count=len(keys),
        duplicate_name_count=duplicate_name_count,
        extension_counts=extension_counts,
        world_names=world_names,
    )


def scan_archives_in_directory(data_dir: str) -> List[BFZArchiveInfo]:
    data_dir = os.path.abspath(data_dir)
    archive_paths: List[str] = []
    for root, _dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(".bfz"):
                archive_paths.append(os.path.join(root, filename))

    result: List[BFZArchiveInfo] = []
    for path in sorted(archive_paths, key=lambda item: item.lower()):
        try:
            result.append(scan_archive_info(path, data_dir))
        except Exception:
            continue
    return result
# ----------------- BFZ Archive -----------------

class BFZArchive:
    def __init__(self, path: str):
        self.path = path
        self.file_entries: List[BFZFileEntry] = []
        self.memory: Optional[bytearray] = None

    def _ensure_lzo(self):
        lzo_backend.require_backend()

    def parse(self, progress: Optional[QProgressDialog] = None):
        self._ensure_lzo()
        with open(self.path, "rb") as f:
            magic = f.read(3)
            if magic != b"ABE":
                raise ValueError("Invalid BFZ magic (expected 'ABE')")

            f.seek(0x28)
            FILES_OFF   = binaryHelpers.read_u64_le(f)
            FOLDERS_OFF = binaryHelpers.read_u64_le(f)
            CHUNKS_OFF  = binaryHelpers.read_u64_le(f)

            FILES  = binaryHelpers.read_u32_le(f)
            FOLDERS = binaryHelpers.read_u32_le(f)
            CHUNKS = binaryHelpers.read_u32_le(f)
            _FILES_again = binaryHelpers.read_u32_le(f)
            _DUMMY_1 = binaryHelpers.read_u32_le(f)
            REAL_CHUNKS = binaryHelpers.read_u32_le(f)

            self.file_entries = _read_file_table(f)

            # CHUNKS
            f.seek(CHUNKS_OFF)
            chunks_count = binaryHelpers.read_u32_le(f)
            _ = binaryHelpers.read_u32_le(f)
            _ = binaryHelpers.read_u64_le(f)
            CHUNKS_TABLE_START = f.tell()

            MAX_OFF = 0
            f.seek(CHUNKS_TABLE_START)
            for _i in range(REAL_CHUNKS):
                NEW_OFFSET = binaryHelpers.read_u64_le(f)
                OFFSET     = binaryHelpers.read_u64_le(f)
                _d64       = binaryHelpers.read_u64_le(f)
                SIZE       = binaryHelpers.read_u32_le(f)
                ZSIZE      = binaryHelpers.read_u32_le(f)
                new_end = NEW_OFFSET + SIZE
                MAX_OFF = max(MAX_OFF, new_end)

            self.memory = bytearray(MAX_OFF)

            if progress:
                progress.setLabelText("Decompressing chunks…")
                progress.setRange(0, REAL_CHUNKS)
                progress.setValue(0)
                if QApplication:
                    QApplication.processEvents()

            f.seek(CHUNKS_TABLE_START)
            for i in range(REAL_CHUNKS):
                NEW_OFFSET = binaryHelpers.read_u64_le(f)
                OFFSET     = binaryHelpers.read_u64_le(f)
                _d64       = binaryHelpers.read_u64_le(f)
                SIZE       = binaryHelpers.read_u32_le(f)
                ZSIZE      = binaryHelpers.read_u32_le(f)

                cur = f.tell()
                f.seek(OFFSET)
                comp = f.read(ZSIZE)
                f.seek(cur)

                decomp = lzo_backend.decompress(comp, SIZE)
                if len(decomp) != SIZE:
                    raise RuntimeError(f"Chunk {i} decompressed size mismatch")

                self.memory[NEW_OFFSET:NEW_OFFSET+SIZE] = decomp

                if progress:
                    progress.setValue(i+1)
                    if QApplication:
                        QApplication.processEvents()
                    if progress.wasCanceled():
                        raise RuntimeError("Operation canceled")

    def read_file_bytes(self, entry: BFZFileEntry) -> bytes:
        if self.memory is None:
            raise RuntimeError("Archive not parsed.")
        start, end = entry.offset, entry.offset + entry.size
        if end > len(self.memory):
            raise RuntimeError("File references bytes beyond memory buffer.")
        return bytes(self.memory[start:end])

    @staticmethod
    def key_hex(key: int) -> str:
        return f"{key & 0xffffffff:08X}"

    def entries_by_key(self) -> dict[int, List[BFZFileEntry]]:
        by_key: dict[int, List[BFZFileEntry]] = {}
        for entry in self.file_entries:
            by_key.setdefault(entry.key, []).append(entry)
        return by_key

    @staticmethod
    def normalized_archive_path(name: str) -> str:
        return name.replace("\\", "/").strip("/")

    @staticmethod
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
        variant_name = f"{stem}__variant_{variant_index}{extension}"
        return os.path.join(directory, variant_name).replace("\\", "/")

    def export_path_map(self, entries: Optional[Iterable[BFZFileEntry]] = None) -> Dict[int, str]:
        entries = list(entries or self.file_entries)
        duplicate_counts: Dict[str, int] = {}
        for entry in entries:
            path = self.normalized_archive_path(entry.name)
            duplicate_counts[path.lower()] = duplicate_counts.get(path.lower(), 0) + 1

        seen: Dict[str, int] = {}
        result: Dict[int, str] = {}
        for entry in entries:
            path = self.normalized_archive_path(entry.name)
            path_key = path.lower()
            seen[path_key] = seen.get(path_key, 0) + 1
            exported_path = path
            if duplicate_counts[path_key] > 1:
                exported_path = self.variant_path(path, seen[path_key])
            result[entry.index] = exported_path
        return result

    def build_manifest(
        self,
        entries: Optional[Iterable[BFZFileEntry]] = None,
        exported_paths: Optional[Dict[int, str]] = None,
    ) -> dict:
        entries = list(entries or self.file_entries)
        exported_paths = exported_paths or self.export_path_map(entries)
        files = []
        for entry in entries:
            archive_path = self.normalized_archive_path(entry.name)
            exported_path = exported_paths.get(entry.index, archive_path)
            files.append({
                "key": self.key_hex(entry.key),
                "path": exported_path,
                "archive_path": archive_path,
                "index": entry.index,
                "offset": entry.offset,
                "size": entry.size,
            })

        return {
            "format": "zombi-manager-bfz-manifest",
            "version": 2,
            "archive": os.path.basename(self.path),
            "archive_path": os.path.abspath(self.path),
            "files": files,
        }

    def write_manifest(
        self,
        output_dir: str,
        entries: Optional[Iterable[BFZFileEntry]] = None,
        exported_paths: Optional[Dict[int, str]] = None,
    ) -> str:
        path = os.path.join(output_dir, "_zombi_bfz_manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.build_manifest(entries, exported_paths), handle, indent=2)
            handle.write("\n")
        return path

# ----------------- BFZ Import ----------------------
# TODO: Allow importing of folders as BFZ's
# I cannot just do what we do with exporting the BFZ but in kind of a reverse way
# Because we lose too much data with it
# We can probably do what some game tools do, and have us pick a "base" bfz to build off of, and
# then transplant the data that way
