from __future__ import annotations
from utilities import tdt

def decode_tdt_rgba(data: bytes):
    return tdt.decode_tdt_top_mip_rgba_data(data)

def write_tdt_png(data: bytes, path: str) -> str:
    return tdt.write_tdt_data_as_png(data, path)

# todo: remove me im fucking old