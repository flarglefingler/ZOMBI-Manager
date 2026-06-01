"""referenced from: https://github.com/Valyyme/oli_file_extractor/blob/main/extract.py"""

from __future__ import annotations

import csv
import io
import os
import struct
from dataclasses import dataclass, field
from typing import List


@dataclass
class OliFile:
    path: str
    filename: str
    texts: List[str]
    lyn_output: str = ""
    agent: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def string_count(self) -> int:
        return len(self.texts)


def _clean_text(value: str) -> str:
    return value.replace("\x00", "").strip()


def read_utf16le_string(
    data: bytes,
    offset: int,
    override_next_offset: int | None = None,
) -> tuple[str, int]:
    if offset + 2 > len(data):
        raise ValueError(f"String length at 0x{offset:x} overruns file.")

    padding = 0xE if override_next_offset is None else override_next_offset
    length = struct.unpack_from("<H", data, offset)[0]
    start = offset + 2
    end = start + length
    if end > len(data):
        raise ValueError(f"UTF-16 string at 0x{offset:x} overruns file.")
    if length % 2:
        raise UnicodeDecodeError("utf-16le", data[start:end], length - 1, length, "odd byte length")

    text = data[start:end].decode("utf-16le")
    return _clean_text(text), end + padding


def parse_oli_data(data: bytes, path: str = "<memory>", force: bool = False) -> OliFile:
    if len(data) < 0x31:
        raise ValueError("File is too small to be a supported OLI.")

    texts: List[str] = []
    warnings: List[str] = []
    filename = ""
    lyn_output = ""
    agent = ""

    try:
        filename, next_offset = read_utf16le_string(data, offset=0x2D, override_next_offset=0x4)
        if filename:
            texts.append(filename)

        if 0 <= next_offset - 1 < len(data) and data[next_offset - 1:next_offset] == b"\x01":
            lyn_output, next_offset = read_utf16le_string(data, offset=next_offset, override_next_offset=0x2)
            agent, next_offset = read_utf16le_string(data, offset=next_offset, override_next_offset=0x2)

        while next_offset < len(data) - 1:
            text, next_offset = read_utf16le_string(data, offset=next_offset)
            if text:
                texts.append(text)
    except (UnicodeDecodeError, ValueError) as exc:
        if not force:
            raise
        warnings.append(str(exc))

    return OliFile(
        path=path,
        filename=filename or os.path.basename(path),
        texts=texts,
        lyn_output=lyn_output,
        agent=agent,
        warnings=warnings,
    )


def parse_oli_file(path: str, force: bool = False) -> OliFile:
    with open(path, "rb") as handle:
        return parse_oli_data(handle.read(), path, force)


def oli_to_csv_text(oli_file: OliFile) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["index", "text"])
    for index, text in enumerate(oli_file.texts):
        writer.writerow([index, text])
    return output.getvalue()


def write_oli_csv(data: bytes, path: str, source_name: str = "<memory>", force: bool = True) -> str:
    oli_file = parse_oli_data(data, source_name, force=force)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(oli_to_csv_text(oli_file))
    return path
