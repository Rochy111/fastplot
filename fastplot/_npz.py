"""numpy's .npz files read without zipfile, whose import (shutil, bz2, lzma:
~7 ms) a fresh process paid at its first colour map or coastline. The
archive's directory and members are read here, each array by numpy's own
.npy reader (numpy.lib.format, which importing numpy loads), so the arrays
are np.load's. Anything else -- zip64, another compression, a member whose
CRC is wrong -- goes to np.load, which says what is wrong with it.
"""
import io
import struct
import zlib

import numpy as np
from numpy.lib import format as _format

_END = b'PK\x05\x06'
_CENTRAL = struct.Struct('<4s6H3I5HII')         # a central directory entry, 46 bytes
_LOCAL = struct.Struct('<4s5H3I2H')             # a local file header, 30 bytes


class Npz:
    """np.load's NpzFile as fastplot reads one: files, [name], with."""

    def __init__(self, members):
        self._members = members
        self.files = list(members)

    def __getitem__(self, key):
        return _format.read_array(io.BytesIO(self._members[key]), allow_pickle=False)

    def __contains__(self, key):
        return key in self._members

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def load(path):
    """The arrays of an .npz file (as np.load(path, allow_pickle=False))."""
    try:
        return Npz(_members(path))
    except (ValueError, struct.error, zlib.error):
        return np.load(path, allow_pickle=False)


def _members(path):
    with open(path, 'rb') as f:
        b = f.read()
    end = b.rfind(_END, max(0, len(b) - (1 << 16) - 22))
    if end < 0:
        raise ValueError('not a zip file')
    count, _, at = struct.unpack_from('<HII', b, end + 10)
    if at == 0xFFFFFFFF:
        raise ValueError('zip64')
    out = {}
    for _ in range(count):
        (sig, _, _, flags, method, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _,
         local) = _CENTRAL.unpack_from(b, at)
        if sig != b'PK\x01\x02' or 0xFFFFFFFF in (csize, usize, local) or flags & 1:
            raise ValueError('not a member read here')
        name = b[at + 46:at + 46 + nlen]                # (ASCII is cp437's too: its codec's import,
        name = name.decode('ascii' if name.isascii() else 'utf-8' if flags & 0x800 else 'cp437')  # ~1 ms, saved)
        at += 46 + nlen + xlen + clen
        head = _LOCAL.unpack_from(b, local)
        if head[0] != b'PK\x03\x04':
            raise ValueError('a damaged member')
        start = local + 30 + head[9] + head[10]
        raw = b[start:start + csize]
        if method == 0:
            data = raw
        elif method == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise ValueError('compression %d' % method)
        if len(data) != usize or zlib.crc32(data) != crc:
            raise ValueError('a damaged member')
        out[name[:-4] if name.endswith('.npy') else name] = data
    return out
