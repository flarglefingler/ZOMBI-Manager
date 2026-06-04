from __future__ import annotations
from typing import Callable

_decompress: Callable[[bytes, int], bytes] | None = None
backend_name = ""
backend_error = ""

try:
    from dissect.util.compression import lzo as _dissect_lzo

    def _decompress_with_dissect(data: bytes, expected_size: int) -> bytes:
        return _dissect_lzo.decompress(data, header=False, buflen=expected_size)

    _decompress = _decompress_with_dissect
    backend_name = "dissect.util"
except Exception as exc:
    backend_error = str(exc)

if _decompress is None:
    try:
        import lzo as _python_lzo #fallback incase we aint have no dissect util

        def _decompress_with_python_lzo(data: bytes, expected_size: int) -> bytes:
            return _python_lzo.decompress(data, False, expected_size)

        _decompress = _decompress_with_python_lzo
        backend_name = "python-lzo"
    except Exception as exc:
        backend_error = str(exc)


def is_available() -> bool:
    return _decompress is not None


def require_backend() -> None:
    if _decompress is None:
        raise RuntimeError(
            "No LZO backend is installed. Install dependencies with:\n"
            "python -m pip install PySide6 dissect.util Pillow numpy\n\n"
            "The old python-lzo package is still supported as a fallback, "
            "but dissect.util is recommended because it works 4 windows."
        )


def decompress(data: bytes, expected_size: int) -> bytes:
    require_backend()
    assert _decompress is not None
    return _decompress(data, expected_size)
