from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Tuple

MDF_MAGIC = b"MDF_"
MDF_HEADER_SIZE = 0x20
MDF_LINK_SIZE = 12

# these were mostly referenced from Jaded
MDF_TYPE_NAMES = {
    0: "Snap",
    1: "OnduleTonCorps",
    2: "Explode",
    3: "LegLink",
    4: "Morphing",
    5: "SemiLookAt",
    6: "Shadow",
    7: "SpecialLookAt",
    8: "Sound",
    9: "XMEN",
    10: "XMEC",
    11: "SPG",
    12: "Symetrie",
    13: "ROTR",
    14: "SNAKE",
    15: "SoundFx",
    16: "PROTEX",
    17: "SaveAddMatrix",
    18: "PAG",
    19: "SoundLoading",
    20: "InfoPhoto",
    21: "StoreTransformedPoints",
    22: "Crush",
    23: "RLICarte",
    24: "Lazy",
    25: "GPG",
    26: "FUR",
    27: "VertexPerturb",
    28: "SpriteMapper2",
    29: "ODE",
    30: "MatrixBore",
    31: "GRID",
    32: "SoundVolume",
    33: "WATER3D",
    34: "Disturber",
    35: "Sfx",
    36: "RotationPaste",
    37: "TranslationPaste",
    38: "AnimatedGAO",
    39: "Weather",
    40: "SoftBody",
    41: "Wind",
    42: "DYNFUR",
    43: "SPG2Holder",
    49: "Vine",
    50: "FogDyn",
    51: "FogDynEmitter",
    52: "HalfAngle",
    53: "BoneMeca",
    54: "FClone",
    55: "UVTexWave",
}


@dataclass(frozen=True)
class MdfModifierLink:
    index: int
    resource_key: int
    flags: int
    type_word: int

    @property
    def key_hex(self) -> str:
        return f"{self.resource_key & 0xffffffff:08X}"

    @property
    def modifier_type(self) -> int:
        return self.type_word & 0xFF

    @property
    def order(self) -> int:
        return (self.type_word >> 16) & 0xFFFF

    @property
    def type_name(self) -> str:
        return MDF_TYPE_NAMES.get(self.modifier_type, f"type {self.modifier_type}")


@dataclass(frozen=True)
class MdfFile:
    name: str
    stored_size: int
    version: int
    links_size: int
    unknown_header: int
    modifier_links: Tuple[MdfModifierLink, ...]

    @property
    def link_count(self) -> int:
        return len(self.modifier_links)

    @property
    def resource_keys(self) -> Tuple[int, ...]:
        return tuple(link.resource_key for link in self.modifier_links)


def _u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        return 0
    return struct.unpack_from("<I", data, offset)[0]


def parse_mdf(data: bytes, name: str = "") -> MdfFile:
    if len(data) < MDF_HEADER_SIZE or data[0x14:0x18] != MDF_MAGIC:
        raise ValueError("missing MDF_ magic")

    links_size = _u32(data, 0x08)
    if links_size == 0:
        link_end = MDF_HEADER_SIZE
    elif links_size % MDF_LINK_SIZE == 0 and MDF_HEADER_SIZE + links_size <= len(data):
        link_end = MDF_HEADER_SIZE + links_size
    else:
        # some future files may not keep the size field trustworthy; keep the
        # parser useful by reading complete link-sized records from the body.
        link_end = len(data) - ((len(data) - MDF_HEADER_SIZE) % MDF_LINK_SIZE)

    links = []
    for index, offset in enumerate(range(MDF_HEADER_SIZE, link_end, MDF_LINK_SIZE)):
        resource_key, flags, type_word = struct.unpack_from("<III", data, offset)
        if resource_key == 0 and flags == 0 and type_word == 0:
            continue
        links.append(MdfModifierLink(index, resource_key, flags, type_word))

    return MdfFile(
        name=name,
        stored_size=len(data),
        version=_u32(data, 0x18),
        links_size=links_size,
        unknown_header=_u32(data, 0x1C),
        modifier_links=tuple(links),
    )


def parse_mdf_file(path: str) -> MdfFile:
    with open(path, "rb") as handle:
        return parse_mdf(handle.read(), os.path.basename(path))
