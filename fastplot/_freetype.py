"""FreeType: the few calls fastplot makes, from the very library freetype-py
ships, without importing freetype-py. Its import is most of a fresh
process's import of fastplot (~30-50 ms of ~75): 28 modules of constants,
ctypes.util with subprocess and shutil, and platform.system(), which starts
a process on Windows -- all for a library path it then hands to ctypes.

Faces are made from the font's bytes (FT_New_Memory_Face), which _text has
read already, so any path works as FreeType's own file opening might not.
Calls are those freetype-py makes (FT_Set_Char_Size, FT_Load_Glyph, ...)
on the same library: what comes out is the same to the bit. Where freetype-py
has no library of its own beside it (a system FreeType), its loader is used.
"""
import ctypes
import importlib.util
import os
import sys
from ctypes import POINTER, Structure, byref, c_char_p, c_int, c_long, c_short, c_ubyte, c_uint, c_ushort, c_void_p

FT_LOAD_DEFAULT = 0
FT_LOAD_NO_HINTING = 2
FT_RENDER_MODE_NORMAL = 0
FT_GLYPH_BBOX_UNSCALED = 0                  # (of a scaled glyph: 26.6 pixels)


class FT_Exception(Exception):
    pass


class Vector(Structure):
    _fields_ = [('x', c_long), ('y', c_long)]


class Matrix(Structure):
    _fields_ = [('xx', c_long), ('xy', c_long), ('yx', c_long), ('yy', c_long)]


class BBox(Structure):
    _fields_ = [('xMin', c_long), ('yMin', c_long), ('xMax', c_long), ('yMax', c_long)]


class _Generic(Structure):
    _fields_ = [('data', c_void_p), ('finalizer', c_void_p)]


class Bitmap(Structure):
    _fields_ = [('rows', c_uint), ('width', c_uint), ('pitch', c_int), ('buffer', POINTER(c_ubyte)),
                ('num_grays', c_ushort), ('pixel_mode', c_ubyte), ('palette_mode', c_ubyte),
                ('palette', c_void_p)]


class GlyphRec(Structure):
    _fields_ = [('library', c_void_p), ('clazz', c_void_p), ('format', c_int), ('advance', Vector)]


class BitmapGlyphRec(Structure):
    _fields_ = [('root', GlyphRec), ('left', c_int), ('top', c_int), ('bitmap', Bitmap)]


class _Metrics(Structure):
    _fields_ = [(n, c_long) for n in ('width', 'height', 'horiBearingX', 'horiBearingY', 'horiAdvance',
                                      'vertBearingX', 'vertBearingY', 'vertAdvance')]


class _SlotHead(Structure):                 # FT_GlyphSlotRec, as far as fastplot reads it
    _fields_ = [('library', c_void_p), ('face', c_void_p), ('next', c_void_p), ('glyph_index', c_uint),
                ('generic', _Generic), ('metrics', _Metrics)]


class _FaceHead(Structure):                 # FT_FaceRec, as far as the glyph slot
    _fields_ = [('num_faces', c_long), ('face_index', c_long), ('face_flags', c_long), ('style_flags', c_long),
                ('num_glyphs', c_long), ('family_name', c_char_p), ('style_name', c_char_p),
                ('num_fixed_sizes', c_int), ('available_sizes', c_void_p), ('num_charmaps', c_int),
                ('charmaps', c_void_p), ('generic', _Generic), ('bbox', BBox), ('units_per_EM', c_ushort),
                ('ascender', c_short), ('descender', c_short), ('height', c_short),
                ('max_advance_width', c_short), ('max_advance_height', c_short),
                ('underline_position', c_short), ('underline_thickness', c_short), ('glyph', c_void_p)]


Glyph = POINTER(GlyphRec)                   # FreeType's FT_Glyph


def _library_file():
    """freetype-py's own FreeType beside its package, found without importing it."""
    spec = importlib.util.find_spec('freetype')
    if spec is None or not spec.submodule_search_locations:
        raise ImportError('no freetype-py')
    name = {'win32': 'libfreetype.dll', 'darwin': 'libfreetype.dylib'}.get(sys.platform, 'libfreetype.so')
    p = os.path.join(spec.submodule_search_locations[0], name)
    if os.path.exists(p):
        return ctypes.CDLL(p)
    from freetype import raw                # (a system FreeType: as freetype-py finds it)
    return raw._lib


_lib = _library_file()


def _fn(name, res, *args):
    f = getattr(_lib, name)
    f.restype, f.argtypes = res, list(args)
    return f


FT_Init_FreeType = _fn('FT_Init_FreeType', c_int, POINTER(c_void_p))
FT_New_Memory_Face = _fn('FT_New_Memory_Face', c_int, c_void_p, c_char_p, c_long, c_long, POINTER(c_void_p))
FT_Set_Char_Size = _fn('FT_Set_Char_Size', c_int, c_void_p, c_long, c_long, c_uint, c_uint)
FT_Set_Transform = _fn('FT_Set_Transform', None, c_void_p, POINTER(Matrix), POINTER(Vector))
FT_Get_Char_Index = _fn('FT_Get_Char_Index', c_uint, c_void_p, ctypes.c_ulong)
FT_Load_Glyph = _fn('FT_Load_Glyph', c_int, c_void_p, c_uint, ctypes.c_int32)
FT_Get_Glyph = _fn('FT_Get_Glyph', c_int, c_void_p, POINTER(Glyph))
FT_Glyph_To_Bitmap = _fn('FT_Glyph_To_Bitmap', c_int, POINTER(Glyph), c_int, POINTER(Vector), ctypes.c_ubyte)
FT_Glyph_Get_CBox = _fn('FT_Glyph_Get_CBox', None, Glyph, c_uint, POINTER(BBox))
FT_Done_Glyph = _fn('FT_Done_Glyph', None, Glyph)

_library = c_void_p()
if FT_Init_FreeType(byref(_library)):
    raise ImportError('FreeType did not start')


def _check(err):
    if err:
        raise FT_Exception('FreeType error %d' % err)


class Face:
    """One font's FreeType face, from its bytes (kept: FreeType reads them)."""

    def __init__(self, data):
        self.data = data
        self.handle = c_void_p()
        _check(FT_New_Memory_Face(_library, data, len(data), 0, byref(self.handle)))
        self.slot = ctypes.cast(self.handle, POINTER(_FaceHead)).contents.glyph
        self._slot = ctypes.cast(self.slot, POINTER(_SlotHead))

    def set_char_size(self, width, height, hres, vres):
        _check(FT_Set_Char_Size(self.handle, width, height, hres, vres))

    def set_transform(self, matrix, delta):
        FT_Set_Transform(self.handle, byref(matrix), byref(delta))

    def get_char_index(self, code):
        return FT_Get_Char_Index(self.handle, code)

    def load_glyph(self, gid, flags):
        _check(FT_Load_Glyph(self.handle, gid, flags))

    def advance64(self):
        """The loaded glyph's (hinted) advance, 26.6 px."""
        return self._slot.contents.metrics.horiAdvance
