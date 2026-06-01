from dataclasses import dataclass
from utilities import binaryHelpers, lzo_backend
from typing import List, Optional
from PySide6.QtWidgets import (
    QApplication, QProgressDialog
)

# ----------------- Data classes -----------------
@dataclass
class BFZFileEntry:
    name: str
    offset: int
    size: int
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

            # FILES_OFF table
            f.seek(FILES_OFF)
            files_count = binaryHelpers.read_u32_le(f)
            _ = binaryHelpers.read_u32_le(f)
            _ = binaryHelpers.read_u64_le(f)

            offsets, sizes, names = [], [], []
            for _i in range(files_count):
                OFFSET = binaryHelpers.read_u64_le(f)
                SIZE   = binaryHelpers.read_u64_le(f)
                _d1 = binaryHelpers.read_u32_le(f)
                _d2 = binaryHelpers.read_u32_le(f)
                offsets.append(OFFSET)
                sizes.append(SIZE)

            for _i in range(files_count):
                NAME  = binaryHelpers.read_fixed_string(f, 0x40)
                _size = binaryHelpers.read_u64_le(f)
                _zero = binaryHelpers.read_u64_le(f)
                _d3   = binaryHelpers.read_u64_le(f)
                _id   = binaryHelpers.read_u32_le(f)
                _d4   = binaryHelpers.read_u32_le(f)
                names.append(NAME)

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
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        raise RuntimeError("Operation canceled")

            self.file_entries = [
                BFZFileEntry(names[i], offsets[i], sizes[i])
                for i in range(files_count)
            ]

    def read_file_bytes(self, entry: BFZFileEntry) -> bytes:
        if self.memory is None:
            raise RuntimeError("Archive not parsed.")
        start, end = entry.offset, entry.offset + entry.size
        if end > len(self.memory):
            raise RuntimeError("File references bytes beyond memory buffer.")
        return bytes(self.memory[start:end])

# ----------------- BFZ Import ----------------------
# TODO: Allow importing of folders as BFZ's
# I cannot just do what we do with exporting the BFZ but in kind of a reverse way
# Because we lose too much data with it
# We can probably do what some game tools do, and have us pick a "base" bfz to build off of, and
# then transplant the data that way
