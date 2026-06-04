from __future__ import annotations

import binascii
import math
import os
import struct
import zlib
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TextureFormat:
    code: int
    name: str
    block_bytes: int | None
    bytes_per_pixel: int | None
    decoder: str


@dataclass
class TdtMipLevel:
    width: int
    height: int
    data_offset: int
    data_size: int


@dataclass
class TdtTexture:
    path: str
    width: int
    height: int
    format_code: int
    format_name: str
    levels: List[TdtMipLevel]


TEXTURE_FORMATS = {
    0x00: TextureFormat(0x00, "RGBA8", None, 4, "rgba8"),
    0x09: TextureFormat(0x09, "BC1/DXT1", 8, None, "bc1"),
    0x0B: TextureFormat(0x0B, "BC3/DXT5", 16, None, "bc3"),
    0x13: TextureFormat(0x13, "L16/AO", None, 2, "l16ao"),
    0x1F: TextureFormat(0x1F, "BC4/ATI1", 8, None, "bc4"),
    0x20: TextureFormat(0x20, "BC5/ATI2", 16, None, "bc5"),
}


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _texture_format(format_code: int) -> TextureFormat:
    if format_code not in TEXTURE_FORMATS:
        raise ValueError(f"Unsupported TDT texture format 0x{format_code:02x}.")
    return TEXTURE_FORMATS[format_code]


def texture_level_data_size(fmt: TextureFormat, width: int, height: int) -> int:
    if fmt.block_bytes is not None:
        return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * fmt.block_bytes
    if fmt.bytes_per_pixel is not None:
        return width * height * fmt.bytes_per_pixel
    raise ValueError(f"Unsupported TDT texture format {fmt.name}.")


def parse_tdt_texture(path: str) -> TdtTexture:
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 0x2E or data.find(b"TDT_") < 0:
        raise ValueError("Missing TDT_ texture marker.")

    header_mip_word = _u16(data, 0x28)
    format_code = header_mip_word & 0xFF
    fmt = _texture_format(format_code)
    width = _u16(data, 0x2A)
    height = _u16(data, 0x2C)
    levels: List[TdtMipLevel] = []

    offset = 0x2E
    level_width = width
    level_height = height
    while True:
        data_size = texture_level_data_size(fmt, level_width, level_height)
        if offset + data_size > len(data):
            raise ValueError(f"TDT mip payload overruns file at 0x{offset:x}.")
        levels.append(TdtMipLevel(level_width, level_height, offset, data_size))
        offset += data_size
        if offset == len(data):
            break
        if offset + 4 > len(data):
            raise ValueError("TDT has truncated trailing mip dimensions.")

        next_width = _u16(data, offset)
        next_height = _u16(data, offset + 2)
        expected_width = max(1, level_width // 2)
        expected_height = max(1, level_height // 2)
        if (next_width, next_height) != (expected_width, expected_height):
            raise ValueError(
                f"Unexpected TDT mip dimensions {next_width}x{next_height}; "
                f"expected {expected_width}x{expected_height}."
            )
        offset += 4
        level_width = next_width
        level_height = next_height

    return TdtTexture(path, width, height, format_code, fmt.name, levels)


def rgb565(value: int) -> Tuple[int, int, int]:
    r = (value >> 11) & 0x1F
    g = (value >> 5) & 0x3F
    b = value & 0x1F
    return (r * 255 + 15) // 31, (g * 255 + 31) // 63, (b * 255 + 15) // 31


def lerp_byte(a: int, b: int, wa: int, wb: int, div: int) -> int:
    return (a * wa + b * wb + div // 2) // div


def decode_bc1_color_block(block: bytes, force_four_color: bool) -> List[Tuple[int, int, int, int]]:
    c0, c1, bits = struct.unpack_from("<HHI", block, 0)
    r0, g0, b0 = rgb565(c0)
    r1, g1, b1 = rgb565(c1)
    palette = [(r0, g0, b0, 255), (r1, g1, b1, 255)]
    if force_four_color or c0 > c1:
        palette.append((
            lerp_byte(r0, r1, 2, 1, 3),
            lerp_byte(g0, g1, 2, 1, 3),
            lerp_byte(b0, b1, 2, 1, 3),
            255,
        ))
        palette.append((
            lerp_byte(r0, r1, 1, 2, 3),
            lerp_byte(g0, g1, 1, 2, 3),
            lerp_byte(b0, b1, 1, 2, 3),
            255,
        ))
    else:
        palette.append((
            lerp_byte(r0, r1, 1, 1, 2),
            lerp_byte(g0, g1, 1, 1, 2),
            lerp_byte(b0, b1, 1, 1, 2),
            255,
        ))
        palette.append((0, 0, 0, 0))
    return [palette[(bits >> (2 * i)) & 0x03] for i in range(16)]


def decode_bc_alpha_block(block: bytes) -> List[int]:
    a0 = block[0]
    a1 = block[1]
    bits = int.from_bytes(block[2:8], "little")
    palette = [a0, a1]
    if a0 > a1:
        for i in range(1, 7):
            palette.append(((7 - i) * a0 + i * a1 + 3) // 7)
    else:
        for i in range(1, 5):
            palette.append(((5 - i) * a0 + i * a1 + 2) // 5)
        palette.extend([0, 255])
    return [palette[(bits >> (3 * i)) & 0x07] for i in range(16)]


def decode_block_texture(width: int, height: int, payload: bytes, decoder: str) -> bytes:
    rgba = bytearray(width * height * 4)
    block_width = max(1, (width + 3) // 4)
    block_height = max(1, (height + 3) // 4)
    stride = 8 if decoder in {"bc1", "bc4"} else 16
    offset = 0

    for by in range(block_height):
        for bx in range(block_width):
            block = payload[offset:offset + stride]
            offset += stride
            if decoder == "bc1":
                pixels = decode_bc1_color_block(block, force_four_color=False)
            elif decoder == "bc3":
                alpha = decode_bc_alpha_block(block[:8])
                colors = decode_bc1_color_block(block[8:16], force_four_color=True)
                pixels = [(r, g, b, alpha[index]) for index, (r, g, b, _a) in enumerate(colors)]
            elif decoder == "bc4":
                red = decode_bc_alpha_block(block)
                pixels = [(value, value, value, 255) for value in red]
            elif decoder == "bc5":
                red = decode_bc_alpha_block(block[:8])
                green = decode_bc_alpha_block(block[8:16])
                pixels = []
                for index in range(16):
                    nx = red[index] / 127.5 - 1.0
                    ny = green[index] / 127.5 - 1.0
                    nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
                    pixels.append((red[index], green[index], int(nz * 255.0 + 0.5), 255))
            else:
                raise ValueError(f"Unsupported block decoder {decoder}.")

            for py in range(4):
                y = by * 4 + py
                if y >= height:
                    continue
                for px in range(4):
                    x = bx * 4 + px
                    if x >= width:
                        continue
                    src = py * 4 + px
                    dst = (y * width + x) * 4
                    rgba[dst:dst + 4] = bytes(pixels[src])
    return bytes(rgba)


def decode_l16ao_texture(width: int, height: int, payload: bytes) -> bytes:
    rgba = bytearray(width * height * 4)
    for index in range(width * height):
        value = payload[index * 2 + 1]
        dst = index * 4
        rgba[dst:dst + 4] = bytes((value, value, value, 255))
    return bytes(rgba)


def decode_tdt_top_mip_rgba(path: str, info: TdtTexture | None = None) -> Tuple[int, int, bytes]:
    info = info or parse_tdt_texture(path)
    fmt = _texture_format(info.format_code)
    level = info.levels[0]
    with open(path, "rb") as handle:
        handle.seek(level.data_offset)
        payload = handle.read(level.data_size)

    if fmt.decoder == "rgba8":
        return info.width, info.height, payload
    if fmt.decoder == "l16ao":
        return info.width, info.height, decode_l16ao_texture(info.width, info.height, payload)
    if fmt.decoder in {"bc1", "bc3", "bc4", "bc5"}:
        return info.width, info.height, decode_block_texture(info.width, info.height, payload, fmt.decoder)
    raise ValueError(f"Unsupported TDT PNG decoder {fmt.name}.")


def tdt_top_mip_has_non_opaque_alpha(path: str, info: TdtTexture | None = None) -> bool:
    info = info or parse_tdt_texture(path)
    fmt = _texture_format(info.format_code)
    if fmt.decoder in {"bc4", "bc5", "l16ao"}:
        return False

    level = info.levels[0]
    with open(path, "rb") as handle:
        handle.seek(level.data_offset)
        payload = handle.read(level.data_size)

    if fmt.decoder == "rgba8":
        return any(payload[index + 3] != 255 for index in range(0, len(payload), 4))

    block_width = max(1, (info.width + 3) // 4)
    block_height = max(1, (info.height + 3) // 4)
    if fmt.decoder == "bc1":
        offset = 0
        for by in range(block_height):
            for bx in range(block_width):
                block = payload[offset:offset + 8]
                offset += 8
                c0, c1, bits = struct.unpack_from("<HHI", block, 0)
                if c0 > c1:
                    continue
                for py in range(4):
                    y = by * 4 + py
                    if y >= info.height:
                        continue
                    for px in range(4):
                        x = bx * 4 + px
                        if x >= info.width:
                            continue
                        if ((bits >> (2 * (py * 4 + px))) & 0x03) == 3:
                            return True
        return False

    if fmt.decoder == "bc3":
        offset = 0
        for by in range(block_height):
            for bx in range(block_width):
                block = payload[offset:offset + 16]
                offset += 16
                alpha = decode_bc_alpha_block(block[:8])
                for py in range(4):
                    y = by * 4 + py
                    if y >= info.height:
                        continue
                    for px in range(4):
                        x = bx * 4 + px
                        if x < info.width and alpha[py * 4 + px] != 255:
                            return True
        return False

    return False


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def write_png_rgba(path: str, width: int, height: int, rgba: bytes) -> None:
    rows = bytearray()
    row_size = width * 4
    for y in range(height):
        rows.append(0)
        start = y * row_size
        rows.extend(rgba[start:start + row_size])

    png = bytearray()
    png += b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
    png += png_chunk(b"IEND", b"")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(bytes(png))


def convert_tdt_to_png(tdt_path: str, png_path: str) -> str:
    width, height, rgba = decode_tdt_top_mip_rgba(tdt_path)
    write_png_rgba(png_path, width, height, rgba)
    return png_path
