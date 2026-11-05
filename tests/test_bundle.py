"""The one-file import cache: modules come out of it as they would from
their .pyc files, or the usual way whenever it does not match."""
import marshal
import os
import subprocess
import sys

from fastplot import _bundle


def run(code, **env):
    e = dict(os.environ, **env)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=e,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_modules_come_from_it_as_from_their_files():
    run('import numba.core.types, freetype; from fastplot import _bundle, _text; _bundle.write()')
    out = run('''
import sys, importlib.resources
import fastplot
from fastplot import _bundle, _text
_bundle.install_numba()                     # (as _jit does, numba about to be imported)
import numba, freetype
import numba.core.types as t
assert _bundle._Finder in sys.meta_path
assert type(numba.__spec__.loader) is _bundle._Loader
assert type(_text.__spec__.loader) is _bundle._Loader
assert numba.__file__.endswith('__init__.py') and numba.__cached__.endswith('.pyc')
assert t.__file__.endswith('types' + __import__('os').sep + '__init__.py') and t.__path__
assert 'def ' in _text.__spec__.loader.get_source('fastplot._text')
assert importlib.resources.files('numba').joinpath('__init__.py').is_file()
print('ok')
''')
    assert out.strip().endswith('ok')


def test_holds_only_what_is_imported_and_lets_it_go():
    """A bundle written holds the modules its process imported; a module's
    code is let go once it is imported."""
    run('import numba.core.types, freetype; from fastplot import _bundle, _text; _bundle.write()')
    out = run('''
import json, marshal, sys
import fastplot
from fastplot import _bundle
_, table = marshal.load(open(_bundle._file('imports'), 'rb'))
print(json.dumps(dict(text='fastplot._text' in table, some_pil=sum(k.startswith('PIL.') for k in table),
                      all_pil=sum(1 for _ in _bundle._walk('PIL', __import__('os').path.dirname(
                          __import__('PIL').__file__))),
                      kept_text='fastplot._text' in _bundle._TABLE,
                      left=sorted(set(_bundle._TABLE) & set(sys.modules)        # (imported
                                  - {'fastplot', 'fastplot._bundle'}))))        # before it was)
''')
    got = __import__('json').loads(out.strip().splitlines()[-1])
    assert got['text'] and got['some_pil'] < got['all_pil']
    assert not got['kept_text'] and not got['left']


def test_turned_off():
    out = run('''
import sys, fastplot
from fastplot import _bundle
assert _bundle._Finder not in sys.meta_path and not _bundle._TABLE
print('ok')
''', FASTPLOT_BUNDLE='0')
    assert out.strip().endswith('ok')


def test_changed_source_imports_the_usual_way(tmp_path, monkeypatch):
    p = tmp_path / 'plain.py'
    p.write_text('x = 1\n')
    entry = _bundle._entry('plain', str(p), False)
    monkeypatch.setitem(_bundle._TABLE, 'plain', entry)
    spec = _bundle._Finder.find_spec('plain')
    assert spec.origin == str(p) and not spec.submodule_search_locations
    p.write_text('x = 22\n')
    assert _bundle._Finder.find_spec('plain') is None
    assert _bundle._Finder.find_spec('not_in_it') is None


def test_gone_source_imports_the_usual_way(monkeypatch):
    monkeypatch.setitem(_bundle._TABLE, 'plain', ('no/such/file.py', 0, 0, False, b''))
    assert _bundle._Finder.find_spec('plain') is None


def test_another_copy_first_on_the_path():
    table = {'numba': ('elsewhere/numba/__init__.py', 0, 0, True, b''),
             'numba.core': ('elsewhere/numba/core/__init__.py', 0, 0, True, b''),
             'fastplot': ('here/fastplot/__init__.py', 0, 0, True, b'')}
    assert set(_bundle._usable(table)) == {'fastplot'}


def test_unreadable_bundle_is_no_bundle(tmp_path, monkeypatch):
    bad = tmp_path / 'imports.bin'
    bad.write_bytes(b'not marshal')
    monkeypatch.setattr(_bundle, '_file', lambda group: str(bad))
    monkeypatch.setattr(_bundle, '_STATE', {'wanted': set(), 'read': set()})
    before = list(sys.meta_path), dict(_bundle._TABLE)
    _bundle.install()
    assert (list(sys.meta_path), dict(_bundle._TABLE)) == before
    assert _bundle._STATE['wanted'] == {'imports'}
    bad.write_bytes(marshal.dumps((b'old magic', {})))
    _bundle._STATE['read'].clear()
    _bundle.install_numba()
    assert (list(sys.meta_path), dict(_bundle._TABLE)) == before
    assert _bundle._STATE['wanted'] == {'imports', 'imports-numba'}
