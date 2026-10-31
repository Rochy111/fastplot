"""The machine fastplot runs on, looked at once a process (lazily, in
well under a millisecond on most systems) -- and the settings that follow
from it, in place of numbers tuned to one desktop.

    _hw.cpus()            logical CPUs this process may run on
    _hw.topology()        cores: physical, the fastest class's, SMT
    _hw.draw_threads()    numba's threads a draw uses
    _hw.fast_cores()      a hybrid CPU's performance cores kept, on Windows
    _hw.gpu_work()        'auto' device: the work from which the GPU draws
    _hw.info()            all of it (fastplot.hardware())

CPUs: what the process may use, not what the box has -- its affinity
(os.process_cpu_count, sched_getaffinity, GetProcessAffinityMask) and,
on Linux, a container's CPU quota (cgroup v2 cpu.max, v1 cfs_quota_us).

Topology: physical cores, and on hybrid CPUs the performance ones -- the
cores of the highest efficiency class on Windows
(GetLogicalProcessorInformationEx), cpu_core's CPUs or the largest
cpu_capacity on Linux, perflevel0 on macOS.

Draw threads. A figure is many short parallel steps, each costing more
the more threads it wakes, and SMT siblings or efficiency cores add
little to a step that lasts a millisecond, while costing their share of
power and of cores other processes could use. So a draw runs on the
performance cores, one thread each: 8 of a 24-thread i7-13700K, which
measured fastest or within noise of it for small plots and 3600x1950
maps alike (4-24 tried). At least 4 where the CPUs allow (a 2-core,
4-thread laptop gains from its siblings), at most MAX_THREADS (a frame
does not have the work to keep more busy; tune() finds out otherwise).
FASTPLOT_THREADS, or what tune() measured on this machine, come first.

Fast cores. Windows moves a busy process without a window in front (a
batch job, a service, a script behind another program) to the efficiency
cores of a hybrid CPU after ~2 s: on an i7-13700K a six-panel figure went
from 17 to 36 ms and a GFS map from 11 to 20 ms, halfway through a run,
and stayed there. fast_cores(), at the first draw, asks Windows not to
slow this process down for the sake of power (for as long as it lives; it
changes nothing else, and no setting outlives it). FASTPLOT_ECOQOS=1 leaves
the process as it is.

GPU. 'auto' draws a figure on the card from AUTO_GPU_WORK (grid points +
pixels; 4M measured by tune() with 8 CPU threads against an RTX 5070
Ti's 70 SMs -- a GFS 0.25 deg map at 1800x975 px, ~2.8M, draws as fast on
the CPU), scaled by the CPU's threads against the card's
multiprocessors: a smaller card, or a stronger CPU, waits for bigger
figures. Fields already on the card (cupy arrays) always draw there. An integrated
GPU (sharing the CPU's memory) needs AUTO_GPU_IGP times the work.
FASTPLOT_GPU_WORK, or tune()'s measurement, come first.

tune() (fastplot.tune) times real draws on this machine -- a plot and a
map at a few thread counts, and the map on CPU and GPU at a few sizes --
and keeps what it found in fastplot's cache directory, keyed by the CPU,
the CPUs allowed, the GPU and fastplot's version: a different machine
(or container size) never reads another's.
"""
import json
import math
import os
import platform
import struct
import sys
import threading

MAX_THREADS = 16                # a draw's threads by default, at most
MIN_THREADS = 4                 # and at least, CPUs allowing
REF_THREADS = 8                 # the machine AUTO_GPU_WORK was measured on: its draw threads
REF_SMS = 70                    # and its card's multiprocessors
AUTO_GPU_WORK = 4_000_000       # grid points + pixels from which that card won
AUTO_GPU_IGP = 4.0              # an integrated GPU: that times as much work
TUNE_FORMAT = 1                 # tune.json's layout: another is ignored

_lock = threading.RLock()
_found = {}                     # name -> value, each looked at once


def _once(fn):
    def get():
        key = fn.__name__
        if key not in _found:
            with _lock:
                if key not in _found:
                    try:
                        _found[key] = fn()
                    except Exception:                   # noqa: BLE001 -- unknown, then
                        _found[key] = None
        return _found[key]
    get.__name__, get.__doc__ = fn.__name__, fn.__doc__
    return get


def _env_int(name):
    v = os.environ.get(name, '').strip()
    try:
        return int(float(v)) if v else None
    except ValueError:
        return None


# ------------------------------------------------------------------ CPUs

def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cpu_list(s):
    """'0-3,8,10-11' -> {0, 1, 2, 3, 8, 10, 11}."""
    out = set()
    for part in (s or '').split(','):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return out


def _quota():
    """A Linux container's CPU quota, in CPUs (None: none set)."""
    v = _read('/sys/fs/cgroup/cpu.max')                 # cgroup v2: "quota period" | "max period"
    if v:
        q, _, p = v.partition(' ')
        if q != 'max' and p:
            return int(q) / int(p)
        return None
    q = _read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')    # cgroup v1
    p = _read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if q and p and int(q) > 0 and int(p) > 0:
        return int(q) / int(p)
    return None


@_once
def _allowed():
    """The logical CPUs (their numbers) this process may run on, or None."""
    if hasattr(os, 'sched_getaffinity'):
        return set(os.sched_getaffinity(0))
    if sys.platform == 'win32':
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
        if k32.GetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()),
                                      ctypes.byref(proc), ctypes.byref(sysm)):
            mask = proc.value
            if mask and (os.cpu_count() or 1) <= 64:    # one processor group: the mask says it all
                return {i for i in range(64) if mask >> i & 1}
    return None


@_once
def cpus():
    """Logical CPUs this process may use: its affinity, a container's quota."""
    n = None
    if hasattr(os, 'process_cpu_count'):                # Python 3.13+
        n = os.process_cpu_count()
    if n is None:
        a = _allowed()
        n = len(a) if a else os.cpu_count() or 1
    if sys.platform.startswith('linux'):
        q = _quota()
        if q:
            n = min(n, max(1, math.ceil(q)))
    return max(1, int(n))


# -------------------------------------------------------------- topology

def _win_topology():
    import ctypes
    k32 = ctypes.windll.kernel32
    size = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # RelationProcessorCore
    if not size.value:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        return None
    raw, off, cores = buf.raw[:size.value], 0, []
    word = ctypes.sizeof(ctypes.c_void_p)               # KAFFINITY
    allowed = _allowed()
    while off + 8 <= len(raw):
        rel, rsize = struct.unpack_from('<II', raw, off)
        if rsize <= 0:
            break
        if rel == 0:
            # PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20], GroupCount,
            # then GroupCount GROUP_AFFINITY (Mask, Group, Reserved[3])
            eff = raw[off + 9]
            ngroups = struct.unpack_from('<H', raw, off + 8 + 22)[0]
            first = off + 8 + 24
            step = word + 8
            ids = []
            for g in range(ngroups):
                mask = int.from_bytes(raw[first + g * step:first + g * step + word], 'little')
                group = struct.unpack_from('<H', raw, first + g * step + word)[0]
                ids += [64 * group + i for i in range(8 * word) if mask >> i & 1]
            if allowed is not None:
                ids = [i for i in ids if i in allowed]
            if ids:
                cores.append((eff, len(ids)))
        off += rsize
    if not cores:
        return None
    top = max(e for e, _ in cores)
    perf = [n for e, n in cores if e == top]
    return dict(cores=len(cores), logical=sum(n for _, n in cores), perf_cores=len(perf),
                perf_logical=sum(perf), hybrid=len(perf) != len(cores))


def _linux_topology():
    allowed = _allowed() or set(range(os.cpu_count() or 1))
    base = '/sys/devices/system/cpu/cpu%d/'
    core_of, cap = {}, {}
    for c in sorted(allowed):
        cid = _read(base % c + 'topology/core_id')
        if cid is None:
            continue
        pkg = _read(base % c + 'topology/physical_package_id') or '0'
        core_of[c] = (pkg, cid)
        v = _read(base % c + 'cpu_capacity')            # big.LITTLE (ARM)
        if v:
            cap[c] = int(v)
    if not core_of:
        return None
    perf = set(core_of)
    pc = _read('/sys/devices/cpu_core/cpus')            # Intel hybrid: the performance cores
    if pc and _read('/sys/devices/cpu_atom/cpus'):
        perf &= _cpu_list(pc)
    elif cap and len(set(cap.values())) > 1:
        top = max(cap.values())
        perf = {c for c in perf if cap.get(c) == top}
    if not perf:
        perf = set(core_of)
    cores = set(core_of.values())
    pcores = {core_of[c] for c in perf}
    return dict(cores=len(cores), logical=len(core_of), perf_cores=len(pcores),
                perf_logical=len(perf), hybrid=len(pcores) != len(cores))


def _mac_topology():
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library('c'))

    def get(name):
        v, n = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if libc.sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(n), None, 0):
            return None
        return v.value

    cores, logical = get('hw.physicalcpu'), get('hw.logicalcpu')
    if not cores:
        return None
    pc = get('hw.perflevel0.physicalcpu') or cores      # Apple silicon: the performance cores
    pl = get('hw.perflevel0.logicalcpu') or logical or cores
    return dict(cores=cores, logical=logical or cores, perf_cores=pc, perf_logical=pl,
                hybrid=pc != cores)


@_once
def topology():
    """{'cores', 'logical', 'perf_cores', 'perf_logical', 'hybrid'} of the
    CPUs this process may use (None for what cannot be found)."""
    t = None
    if sys.platform == 'win32':
        t = _win_topology()
    elif sys.platform.startswith('linux'):
        t = _linux_topology()
    elif sys.platform == 'darwin':
        t = _mac_topology()
    if t is None:                                       # unknown: every CPU a core
        n = cpus()
        t = dict(cores=n, logical=n, perf_cores=n, perf_logical=n, hybrid=False)
    n = cpus()                                          # a quota below the cores: the quota
    if t['perf_cores'] > n:
        t = dict(t, perf_cores=n, perf_logical=min(t['perf_logical'], n))
    return t


# ------------------------------------------------------------- the tuning

def cpu_name():
    """The CPU's model, as the OS names it ('' when it will not say)."""
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                return str(winreg.QueryValueEx(k, 'ProcessorNameString')[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith('linux'):
        for line in (_read('/proc/cpuinfo') or '').splitlines():
            if line.lower().startswith(('model name', 'hardware', 'cpu model')):
                return line.partition(':')[2].strip()
    elif sys.platform == 'darwin':
        import subprocess
        try:
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:                               # noqa: BLE001
            pass
    return platform.processor() or ''


def cache_dir():
    """fastplot's cache directory (FASTPLOT_CACHE_DIR, else the OS's)."""
    d = os.environ.get('FASTPLOT_CACHE_DIR')
    if d:
        return d
    if sys.platform == 'win32':
        root = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~\\AppData\\Local')
    elif sys.platform == 'darwin':
        root = os.path.expanduser('~/Library/Caches')
    else:
        root = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(root, 'fastplot')


def _tune_path():
    return os.path.join(cache_dir(), 'tune.json')


def machine_key(gpu_name=''):
    """What a tuning is good for: this CPU, the CPUs allowed, the GPU."""
    from . import __version__
    return '|'.join([cpu_name(), str(cpus()), gpu_name or '', '.'.join(__version__.split('.')[:2])])


@_once
def _tune_file():
    try:
        with open(_tune_path()) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d if d.get('format') == TUNE_FORMAT else {}


def tuned(gpu_name=None):
    """tune()'s results for this machine: the CPU's ({} when it has not
    been run here), or with gpu_name the GPU's for that card."""
    d = _tune_file()
    if gpu_name is None:
        return d.get('cpu', {}) if d.get('cpu_key') == machine_key() else {}
    g = d.get('gpu') or {}
    return g if g.get('key') == machine_key(gpu_name) else {}


def save_tuned(cpu=None, gpu=None):
    """Keep tune()'s results (atomically: a reader sees the old or the new)."""
    path = _tune_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {'format': TUNE_FORMAT}
    try:
        with open(path) as f:
            old = json.load(f)
        if old.get('format') == TUNE_FORMAT:
            d.update(old)
    except (OSError, ValueError):
        pass
    if cpu is not None:
        d['cpu_key'], d['cpu'] = machine_key(), cpu
    if gpu is not None:
        d['gpu'] = gpu                                  # its own 'key': machine_key(its name)
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, path)
    with _lock:
        _found.pop('_tune_file', None)
        _found.pop('draw_threads', None)
    return path


# --------------------------------------------------------------- choices

@_once
def draw_threads():
    """numba's threads a draw uses: FASTPLOT_THREADS, tune()'s, else the
    performance cores (MIN_THREADS..MAX_THREADS, within the CPUs allowed)."""
    n = _env_int('FASTPLOT_THREADS')
    if n is None:
        n = tuned().get('threads')
    if n is None:
        t = topology()
        n = t['perf_cores']
        if n < MIN_THREADS:
            n = min(MIN_THREADS, t['perf_logical'])
        n = min(n, MAX_THREADS)
    return max(1, min(int(n), cpus()))


@_once
def wide_threads():
    """Threads for even work that pays for itself (_jit.wider): the
    performance cores' logical CPUs, unless the count was set by hand."""
    if _env_int('FASTPLOT_THREADS') is not None:
        return draw_threads()
    return max(draw_threads(), min(topology()['perf_logical'], cpus()))


@_once
def fast_cores():
    """The process off Windows' power throttling, on a hybrid CPU (see the
    module). True if it was, False if Windows refused, None if not asked."""
    if sys.platform != 'win32' or os.environ.get('FASTPLOT_ECOQOS', '0') == '1' or \
            not topology().get('hybrid'):
        return None
    import ctypes

    class Throttling(ctypes.Structure):                 # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [('Version', ctypes.c_ulong), ('ControlMask', ctypes.c_ulong),
                    ('StateMask', ctypes.c_ulong)]
    # version 1; EXECUTION_SPEED ours to say, and off: ProcessPowerThrottling is 4
    state = Throttling(1, 1, 0)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    return bool(k32.SetProcessInformation(ctypes.c_void_p(k32.GetCurrentProcess()), 4,
                                          ctypes.byref(state), ctypes.sizeof(state)))


def gpu_props():
    """{'name', 'sms', 'memory', 'integrated', 'cc'} of the current CUDA
    device -- cupy imported (the caller has found it usable) -- or None."""
    if 'gpu_props' in _found:
        return _found['gpu_props']
    try:
        import cupy
        p = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
        name = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
        props = dict(name=name, sms=int(p['multiProcessorCount']),
                     memory=int(p['totalGlobalMem']), integrated=bool(p.get('integrated', 0)),
                     cc='%d.%d' % (p['major'], p['minor']))
    except Exception:                                   # noqa: BLE001
        props = None
    _found['gpu_props'] = props
    return props


def gpu_work():
    """'auto': the work (grid points + pixels) from which a figure draws on
    the GPU -- FASTPLOT_GPU_WORK, tune()'s, else AUTO_GPU_WORK scaled by
    the CPU's draw threads against the card's multiprocessors."""
    n = _env_int('FASTPLOT_GPU_WORK')
    if n is not None:
        return n
    p = gpu_props()
    if p is None:
        return AUTO_GPU_WORK
    g = tuned(p['name'])
    if g.get('work'):
        return int(g['work'])
    w = AUTO_GPU_WORK * (draw_threads() / REF_THREADS) * (REF_SMS / max(1, p['sms']))
    if p['integrated']:
        w *= AUTO_GPU_IGP
    return int(w)


def info():
    """What was found, and what fastplot does with it."""
    out = dict(cpu=cpu_name(), cpus_allowed=cpus(), cpus_total=os.cpu_count(),
               topology=topology(), draw_threads=draw_threads(),
               threads_from=('FASTPLOT_THREADS' if _env_int('FASTPLOT_THREADS') is not None
                             else 'tune()' if tuned().get('threads') is not None else 'cores'),
               tune_file=_tune_path() if _tune_file() else None)
    from . import _jit
    if _jit._numba is not None:
        out['numba_threads'] = _jit._numba.config.NUMBA_NUM_THREADS
        out['threading_layer'] = _jit.layer()
    return out
