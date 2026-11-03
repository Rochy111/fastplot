"""A thread pool of fastplot's own for the parallel kernels, in place of
numba's OpenMP layer (on Windows and Linux, x86-64).

OpenMP's workers wait for more work by spinning. MSVC's OpenMP runtime (vcomp, which numba's omp layer runs on there) keeps
every worker of a parallel region busy-waiting ~100 ms after it, for a
region that might follow, and reads no setting for how long but
OMP_WAIT_POLICY, once. A frame is tens of small regions with Python
between them, so its workers never stop: 7 cores spun for a frame of
mostly Python -- on an i7-13700K a six-panel plot took 105 ms of CPU a
15 ms frame, a GFS map 208 a 27 ms one. Passive waits gave most of that
back for 3 to 15% more time a frame; TBB's workers came to a region late
(the caller ran several of its parts alone: 5 to 9% slower on two
figures). Here a worker waits as long as a wake has lately cost it (ski
rental: then never more than twice the best), measured on this machine,
then sleeps: the same figures took 17 and 119 ms of CPU, in as little
time or less. GNU's (libgomp, numba's omp layer on Linux) spins for a
shorter while, but a team of all the CPUs: under WSL on the same machine
a frame took 12 to 14 times its time in CPU.

numba's kernels call numba_parallel_for(fn, args, dims, steps, data,
inner_ndim, array_count, num_threads) with dims[0] parts its scheduler
made (do_scheduling, numba's own still), and get_thread_id,
get_num_threads and set_num_threads; those four are taken from here:

- a calling thread takes a team of num_threads - 1 workers (made the first
  time one is wanted, kept for the next), so several threads' regions run
  at once (frames());
- the parts go in blocks as OpenMP's static schedule deals them to its
  threads (the first size % num_threads one part more), and the blocks
  are claimed from a counter, the caller's thread among them, so a worker
  late to wake leaves nothing waiting for it;
- a block's parts run in order, get_thread_id() being the OpenMP thread
  whose block it is: a reduction's partial result for that thread adds up
  as OpenMP's does, to the bit, whichever thread ran it;
- a region inside a region, or on one thread, runs on the calling thread
  as OpenMP's team of one (thread id 0).

The code is LLVM IR (below): linked into the kernel image as one of its
units (object_code, at build time), or compiled by llvmlite where numba
runs the kernels (jit). Either way one copy runs a process (start);
FASTPLOT_POOL=0 keeps numba's layer.
"""
import ctypes
import os
import platform
import sys

EXPORTS = ('fastplot_pool_parallel_for', 'fastplot_pool_get_thread_id', 'fastplot_pool_get_num_threads',
           'fastplot_pool_set_num_threads', 'fastplot_pool_init', 'fastplot_pool_forget')
REPLACES = {'parallel_for': 'fastplot_pool_parallel_for', 'get_thread_id': 'fastplot_pool_get_thread_id',
            'get_num_threads': 'fastplot_pool_get_num_threads',
            'set_num_threads': 'fastplot_pool_set_num_threads'}     # numba's layer's names -> these
EXTERNALS = (('kernel32', ('TlsGetValue', 'TlsSetValue', 'QueryPerformanceCounter', 'CreateThread',
                           'CloseHandle')),
             ('api-ms-win-core-synch-l1-2-0', ('WaitOnAddress', 'WakeByAddressAll', 'WakeByAddressSingle')))
LIBRARY = {n: lib for lib, names in EXTERNALS for n in names}     # (by name: some are forwarded)
SPIN_US = (50.0, 2.0, 2000.0)       # a new team's first spin, and the least and most it learns, us

_started = {}                       # the running copy: {exported name: address}, once started


def possible(layer):
    """Can this pool stand in for numba's `layer` ('omp', 'tbb',
    'workqueue')? OpenMP's runtimes are what it is for -- vcomp on Windows,
    libgomp on Linux -- on x86-64 (the IR's pause)."""
    return (sys.platform in ('win32', 'linux') and layer == 'omp'
            and platform.machine().lower() in ('amd64', 'x86_64'))


def wanted(layer):
    """possible(), and asked for: FASTPLOT_POOL=1, or unset on Windows.
    Under WSL on an i7-13700K libgomp's frames took 12 to 14 times their
    time in CPU, and here 2 to 5 -- but 2 to 7% longer on four of six
    figures (4 to 7% shorter on two), however long the spin; so on Linux
    it is the user's to choose, and numba's layer is the default."""
    v = os.environ.get('FASTPLOT_POOL', '').strip()
    if v == '0' or (not v and sys.platform != 'win32'):
        return False
    return possible(layer)


def ir(triple):
    return 'target triple = "%s"\n%s%s' % (triple, WINDOWS if sys.platform == 'win32' else LINUX, IR)


def _register():
    """The Windows functions the IR calls, for llvmlite to find (by name,
    as numba's JIT finds the C library's; on Linux the process's search
    finds them)."""
    if sys.platform != 'win32':
        return
    import llvmlite.binding as ll
    for lib, names in EXTERNALS:
        d = ctypes.WinDLL(lib)
        for n in names:
            ll.add_symbol(n, ctypes.cast(getattr(d, n), ctypes.c_void_p).value)


def object_code(codegen):
    """The pool as an ELF object for the kernel image (numba's target, as
    its kernels are)."""
    import llvmlite.binding as ll
    _register()
    m = ll.parse_assembly(ir(codegen._tm.triple))
    m.verify()
    return codegen._tm.emit_object(m)


_engine = []                        # (llvmlite's engine, kept alive with the code it made)


def jit():
    """The pool compiled here by llvmlite (numba runs the kernels): its
    exported functions' addresses."""
    import llvmlite.binding as ll
    ll.initialize_native_target()
    ll.initialize_native_asmprinter()
    _register()
    if sys.platform == 'win32':
        ll.add_symbol('malloc', ctypes.cast(ctypes.CDLL('ucrtbase').malloc, ctypes.c_void_p).value)
    tm = ll.Target.from_default_triple().create_target_machine(opt=3)
    m = ll.parse_assembly(ir(ll.get_default_triple()))
    m.verify()
    engine = ll.create_mcjit_compiler(m, tm)
    engine.finalize_object()
    _engine.append(engine)
    return {n: engine.get_function_address(n) for n in EXPORTS}


def start(addresses, size):
    """The pool at `addresses` set going (once a process: the first copy
    started is the one; later calls get it): its thread-local slot, the
    count a new thread starts with (`size`, numba's), and how long a new
    team first spins. Returns the running copy's addresses."""
    if _started:
        return _started
    if sys.platform == 'win32':
        k32 = ctypes.WinDLL('kernel32')
        k32.TlsAlloc.restype = ctypes.c_uint32
        freq = ctypes.c_int64()
        k32.QueryPerformanceFrequency(ctypes.byref(freq))
        tick = freq.value / 1e6                                         # ticks a microsecond
        tls = k32.TlsAlloc()
        if tls == 0xFFFFFFFF:
            raise OSError('no thread-local slot to be had')
    else:
        key = ctypes.c_uint32()
        if ctypes.CDLL(None).pthread_key_create(ctypes.byref(key), None):
            raise OSError('no thread-local key to be had')
        tls, tick = key.value, 1000.0                                   # (clock_gettime's ns)
        forget = ctypes.CFUNCTYPE(None)(addresses['fastplot_pool_forget'])
        os.register_at_fork(after_in_child=forget)      # (the parent's workers are not the child's)
        _forget.append(forget)
    ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)(
        addresses['fastplot_pool_init'])(tls, size, *(max(1, int(us * tick)) for us in SPIN_US))
    _started.update(addresses)
    return _started


_forget = []                        # (the fork handler, kept alive)


def started():
    """The running copy's addresses, or None."""
    return _started or None


def install_in_numba(size):
    """numba's parallel kernels (those it loads from now) and its thread
    counts on this pool: llvmlite's symbols for them pointed here, after
    numba's own layer is up (its scheduler stays numba's). The running copy
    if there is one (the image's), else one compiled here."""
    import llvmlite.binding as ll
    from numba.np.ufunc import parallel
    parallel._launch_threads()
    running = started()                                 # (the image's: its counts are the ones)
    count = None if running else parallel._get_num_threads()   # (else this thread's numba had)
    a = start(running or jit(), size)
    ll.add_symbol('numba_parallel_for', a['fastplot_pool_parallel_for'])
    for name in ('get_num_threads', 'set_num_threads', 'get_thread_id'):
        ll.add_symbol(name, a[REPLACES[name]])
    parallel._set_num_threads = ctypes.CFUNCTYPE(None, ctypes.c_int)(a['fastplot_pool_set_num_threads'])
    parallel._get_num_threads = ctypes.CFUNCTYPE(ctypes.c_int)(a['fastplot_pool_get_num_threads'])
    parallel._get_thread_id = ctypes.CFUNCTYPE(ctypes.c_int)(a['fastplot_pool_get_thread_id'])
    if count is not None:
        parallel._set_num_threads(count)


# what differs by system: a thread-local slot, a clock, a thread started, sleeping on a word until
# woken (or it changes), waking
WINDOWS = r'''
declare i8* @TlsGetValue(i32)
declare i32 @TlsSetValue(i32, i8*)
declare i32 @QueryPerformanceCounter(i64*)
declare i8* @CreateThread(i8*, i64, i32 (i8*)*, i8*, i32, i32*)
declare i32 @CloseHandle(i8*)
declare i32 @WaitOnAddress(i8*, i8*, i64, i32)
declare void @WakeByAddressAll(i8*)
declare void @WakeByAddressSingle(i8*)

define internal i8* @tls_get(i32 %k) alwaysinline {
  %p = call i8* @TlsGetValue(i32 %k)
  ret i8* %p
}

define internal void @tls_set(i32 %k, i8* %v) {
  call i32 @TlsSetValue(i32 %k, i8* %v)
  ret void
}

define internal i64 @now() alwaysinline {
  %c = alloca i64
  call i32 @QueryPerformanceCounter(i64* %c)
  %v = load i64, i64* %c
  ret i64 %v
}

define internal i32 @thread_main(i8* %a) {
  call void @worker(i8* %a)
  ret i32 0
}

define internal void @spawn(i8* %a) {
  %th = call i8* @CreateThread(i8* null, i64 0, i32 (i8*)* @thread_main, i8* %a, i32 0, i32* null)
  call i32 @CloseHandle(i8* %th)
  ret void
}

define internal void @wait_word(i64* %p, i64 %v, i64* %vv) {
  store i64 %v, i64* %vv
  %p8 = bitcast i64* %p to i8*
  %vv8 = bitcast i64* %vv to i8*
  call i32 @WaitOnAddress(i8* %p8, i8* %vv8, i64 8, i32 -1)
  ret void
}

define internal void @wake_all(i64* %p) {
  %p8 = bitcast i64* %p to i8*
  call void @WakeByAddressAll(i8* %p8)
  ret void
}

define internal void @wake_one(i64* %p) {
  %p8 = bitcast i64* %p to i8*
  call void @WakeByAddressSingle(i8* %p8)
  ret void
}
'''

# (futex on the word's low 32 bits, which every change of it changes: gen goes by 2, done by 1)
LINUX = r'''
declare i8* @pthread_getspecific(i32)
declare i32 @pthread_setspecific(i32, i8*)
declare i32 @clock_gettime(i32, i64*)
declare i32 @pthread_create(i64*, i8*, i8* (i8*)*, i8*)
declare i32 @pthread_detach(i64)
declare i64 @syscall(i64, ...)

define internal i8* @tls_get(i32 %k) alwaysinline {
  %p = call i8* @pthread_getspecific(i32 %k)
  ret i8* %p
}

define internal void @tls_set(i32 %k, i8* %v) {
  call i32 @pthread_setspecific(i32 %k, i8* %v)
  ret void
}

define internal i64 @now() alwaysinline {
  %ts = alloca [2 x i64]
  %p = getelementptr [2 x i64], [2 x i64]* %ts, i32 0, i32 0
  call i32 @clock_gettime(i32 1, i64* %p)
  %s = load i64, i64* %p
  %np = getelementptr [2 x i64], [2 x i64]* %ts, i32 0, i32 1
  %n = load i64, i64* %np
  %sn = mul i64 %s, 1000000000
  %v = add i64 %sn, %n
  ret i64 %v
}

define internal i8* @thread_main(i8* %a) {
  call void @worker(i8* %a)
  ret i8* null
}

define internal void @spawn(i8* %a) {
  %th = alloca i64
  call i32 @pthread_create(i64* %th, i8* null, i8* (i8*)* @thread_main, i8* %a)
  %h = load i64, i64* %th
  call i32 @pthread_detach(i64 %h)
  ret void
}

define internal void @wait_word(i64* %p, i64 %v, i64* %vv) {
  %v32 = trunc i64 %v to i32
  call i64 (i64, ...) @syscall(i64 202, i64* %p, i32 128, i32 %v32, i8* null)
  ret void
}

define internal void @wake_all(i64* %p) {
  call i64 (i64, ...) @syscall(i64 202, i64* %p, i32 129, i32 2147483647)
  ret void
}

define internal void @wake_one(i64* %p) {
  call i64 (i64, ...) @syscall(i64 202, i64* %p, i32 129, i32 1)
  ret void
}
'''

IR = r'''
%state = type { i32, i32, i32, i32 }           ; this thread's: thread id, thread count, in a region

@tls = internal global i32 -1
@lock = internal global i64 0                  ; a spinlock over the free teams
@free_head = internal global i64* null
@default_nt = internal global i32 1
@spin0 = internal global i64 500               ; clock ticks (now())
@spin_min = internal global i64 20
@spin_max = internal global i64 20000

declare i8* @malloc(i64)
declare void @llvm.x86.sse2.pause()
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

define internal void @lock_take() {
entry:
  br label %try
try:
  %old = atomicrmw xchg i64* @lock, i64 1 acquire
  %free = icmp eq i64 %old, 0
  br i1 %free, label %got, label %spin
spin:
  call void @llvm.x86.sse2.pause()
  br label %try
got:
  ret void
}

define internal void @lock_give() {
  store atomic i64 0, i64* @lock release, align 8
  ret void
}

; a child of fork(): the parent's workers are not in it, so neither are their teams
define void @fastplot_pool_forget() {
  store i64* null, i64** @free_head
  store i64 0, i64* @lock
  ret void
}

; a team, 64 words, 64-byte aligned: gen 0, next 8, done 16, busy 24, sleepers 32 (each its own cache
; line); the job: fn 40, args 41, dims 42, steps 43, data 44, inner_ndim 45, array_count 46,
; nthreads 47, size 48; nworkers 49, caller asleep 50, spin 51, free-list link 52, posted at 53,
; wake cost lately 54.

define internal i64* @slot(i64* %t, i64 %i) alwaysinline {
  %p = getelementptr i64, i64* %t, i64 %i
  ret i64* %p
}

define internal i64 @qpc() alwaysinline {
  %v = call i64 @now()
  ret i64 %v
}

define internal %state* @state() {
entry:
  %ix = load i32, i32* @tls
  %p = call i8* @tls_get(i32 %ix)
  %isnull = icmp eq i8* %p, null
  br i1 %isnull, label %make, label %have
make:
  %m = call i8* @malloc(i64 16)
  %s = bitcast i8* %m to %state*
  %nt = load i32, i32* @default_nt
  %f0 = getelementptr %state, %state* %s, i32 0, i32 0
  store i32 0, i32* %f0
  %f1 = getelementptr %state, %state* %s, i32 0, i32 1
  store i32 %nt, i32* %f1
  %f2 = getelementptr %state, %state* %s, i32 0, i32 2
  store i32 0, i32* %f2
  call void @tls_set(i32 %ix, i8* %m)
  ret %state* %s
have:
  %h = bitcast i8* %p to %state*
  ret %state* %h
}

define i32 @fastplot_pool_get_thread_id() {
  %s = call %state* @state()
  %f = getelementptr %state, %state* %s, i32 0, i32 0
  %v = load i32, i32* %f
  ret i32 %v
}

define i32 @fastplot_pool_get_num_threads() {
  %s = call %state* @state()
  %f = getelementptr %state, %state* %s, i32 0, i32 1
  %v = load i32, i32* %f
  ret i32 %v
}

define void @fastplot_pool_set_num_threads(i32 %n) {
  %s = call %state* @state()
  %f = getelementptr %state, %state* %s, i32 0, i32 1
  store i32 %n, i32* %f
  ret void
}

; part r of the team's job, on this thread, as OpenMP's thread tid
define internal void @part(i64* %t, i64 %r, i64 %tid, i64* %count, i8** %arrs, %state* %s) {
entry:
  %fnp = call i64* @slot(i64* %t, i64 40)
  %fni = load i64, i64* %fnp
  %fn = inttoptr i64 %fni to void (i8**, i64*, i64*, i8*)*
  %argsp = call i64* @slot(i64* %t, i64 41)
  %argsi = load i64, i64* %argsp
  %args = inttoptr i64 %argsi to i8**
  %dimsp = call i64* @slot(i64* %t, i64 42)
  %dimsi = load i64, i64* %dimsp
  %dims = inttoptr i64 %dimsi to i64*
  %stepsp = call i64* @slot(i64* %t, i64 43)
  %stepsi = load i64, i64* %stepsp
  %steps = inttoptr i64 %stepsi to i64*
  %datap = call i64* @slot(i64* %t, i64 44)
  %datai = load i64, i64* %datap
  %data = inttoptr i64 %datai to i8*
  %inp = call i64* @slot(i64* %t, i64 45)
  %inner = load i64, i64* %inp
  %acp = call i64* @slot(i64* %t, i64 46)
  %ac = load i64, i64* %acp
  %tid32 = trunc i64 %tid to i32
  %f0 = getelementptr %state, %state* %s, i32 0, i32 0
  store i32 %tid32, i32* %f0
  %arglen = add i64 %inner, 1
  %bytes = mul i64 %arglen, 8
  %cnt8 = bitcast i64* %count to i8*
  %dims8 = bitcast i64* %dims to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %cnt8, i8* %dims8, i64 %bytes, i1 false)
  store i64 1, i64* %count
  br label %loop
loop:
  %j = phi i64 [0, %entry], [%j1, %body]
  %more = icmp slt i64 %j, %ac
  br i1 %more, label %body, label %call
body:
  %ap = getelementptr i8*, i8** %args, i64 %j
  %a = load i8*, i8** %ap
  %sp = getelementptr i64, i64* %steps, i64 %j
  %st = load i64, i64* %sp
  %off = mul i64 %st, %r
  %ar = getelementptr i8, i8* %a, i64 %off
  %dp = getelementptr i8*, i8** %arrs, i64 %j
  store i8* %ar, i8** %dp
  %j1 = add i64 %j, 1
  br label %loop
call:
  call void %fn(i8** %arrs, i64* %count, i64* %steps, i8* %data)
  ret void
}

; OpenMP's static schedule of size parts on nt threads: thread b's parts are lo..hi-1, the first
; size % nt threads one more than the rest; min(size, nt) threads have any
define internal i64 @blocks(i64 %size, i64 %nt) alwaysinline {
  %few = icmp slt i64 %size, %nt
  %b = select i1 %few, i64 %size, i64 %nt
  ret i64 %b
}

define internal i64 @block_lo(i64 %b, i64 %size, i64 %nt) alwaysinline {
  %q = sdiv i64 %size, %nt
  %rem = srem i64 %size, %nt
  %bq = mul i64 %b, %q
  %lt = icmp slt i64 %b, %rem
  %m = select i1 %lt, i64 %b, i64 %rem
  %lo = add i64 %bq, %m
  ret i64 %lo
}

; threads' blocks claimed until none is left, each block's parts in order as that thread (so a
; reduction's partial result for it adds up as OpenMP's does); the last block done wakes a caller
define internal void @run(i64* %t, %state* %s) {
entry:
  %sv = call i8* @llvm.stacksave()
  %inp = call i64* @slot(i64* %t, i64 45)
  %inner = load i64, i64* %inp
  %acp = call i64* @slot(i64* %t, i64 46)
  %ac = load i64, i64* %acp
  %arglen = add i64 %inner, 1
  %count = alloca i64, i64 %arglen
  %ac1 = add i64 %ac, 1
  %arrs = alloca i8*, i64 %ac1
  %szp = call i64* @slot(i64* %t, i64 48)
  %size = load i64, i64* %szp
  %ntp = call i64* @slot(i64* %t, i64 47)
  %nt = load i64, i64* %ntp
  %f0 = getelementptr %state, %state* %s, i32 0, i32 0
  %f1 = getelementptr %state, %state* %s, i32 0, i32 1
  %f2 = getelementptr %state, %state* %s, i32 0, i32 2
  %otid = load i32, i32* %f0
  %ont = load i32, i32* %f1
  %oin = load i32, i32* %f2
  %nt32 = trunc i64 %nt to i32
  store i32 %nt32, i32* %f1
  store i32 1, i32* %f2
  %nextp = call i64* @slot(i64* %t, i64 8)
  %donep = call i64* @slot(i64* %t, i64 16)
  %nb = call i64 @blocks(i64 %size, i64 %nt)
  br label %claim
claim:
  %b = atomicrmw add i64* %nextp, i64 1 seq_cst
  %left = icmp slt i64 %b, %nb
  br i1 %left, label %work, label %out
work:
  %lo = call i64 @block_lo(i64 %b, i64 %size, i64 %nt)
  %b1 = add i64 %b, 1
  %hi = call i64 @block_lo(i64 %b1, i64 %size, i64 %nt)
  br label %parts
parts:
  %r = phi i64 [%lo, %work], [%r1, %onepart]
  %inblock = icmp slt i64 %r, %hi
  br i1 %inblock, label %onepart, label %blockdone
onepart:
  call void @part(i64* %t, i64 %r, i64 %b, i64* %count, i8** %arrs, %state* %s)
  %r1 = add i64 %r, 1
  br label %parts
blockdone:
  %d = atomicrmw add i64* %donep, i64 1 seq_cst
  %d1 = add i64 %d, 1
  %last = icmp eq i64 %d1, %nb
  br i1 %last, label %wake, label %claim
wake:
  %msp = call i64* @slot(i64* %t, i64 50)
  %ms = load atomic i64, i64* %msp seq_cst, align 8
  %asleep = icmp ne i64 %ms, 0
  br i1 %asleep, label %dowake, label %claim
dowake:
  call void @wake_one(i64* %donep)
  br label %claim
out:
  store i32 %otid, i32* %f0
  store i32 %ont, i32* %f1
  store i32 %oin, i32* %f2
  call void @llvm.stackrestore(i8* %sv)
  ret void
}

; until *p is not v (and, `even`, not odd): spin the team's spin, then sleep; `worker`: counted asleep
; while it sleeps, and what the wake cost it learnt (the next spin)
define internal i64 @wait_change(i64* %t, i64* %p, i64 %v, i1 %even, i1 %worker) {
entry:
  %vv = alloca i64
  %spinp = call i64* @slot(i64* %t, i64 51)
  %spin = load atomic i64, i64* %spinp monotonic, align 8
  %t0 = call i64 @qpc()
  %until = add i64 %t0, %spin
  br label %spinloop
spinloop:
  %k = phi i64 [0, %entry], [%k1, %pause], [%k1, %clock]
  %g = load atomic i64, i64* %p acquire, align 8
  %odd1 = and i64 %g, 1
  %isodd = icmp ne i64 %odd1, 0
  %odd = and i1 %isodd, %even
  %same = icmp eq i64 %g, %v
  %stay = or i1 %same, %odd
  br i1 %stay, label %pause, label %got
pause:
  call void @llvm.x86.sse2.pause()
  %k1 = add i64 %k, 1
  %m = and i64 %k1, 15
  %check = icmp eq i64 %m, 0
  br i1 %check, label %clock, label %spinloop
clock:
  %now = call i64 @qpc()
  %over = icmp sgt i64 %now, %until
  br i1 %over, label %sleep, label %spinloop
sleep:
  %slp = call i64* @slot(i64* %t, i64 32)
  br i1 %worker, label %inc, label %sleeploop
inc:
  atomicrmw add i64* %slp, i64 1 seq_cst
  br label %sleeploop
sleeploop:
  %g2 = load atomic i64, i64* %p seq_cst, align 8
  %odd2 = and i64 %g2, 1
  %isodd2 = icmp ne i64 %odd2, 0
  %oddb = and i1 %isodd2, %even
  %same2 = icmp eq i64 %g2, %v
  %stay2 = or i1 %same2, %oddb
  br i1 %stay2, label %block, label %woke
block:
  br i1 %oddb, label %brief, label %dowait
brief:
  call void @llvm.x86.sse2.pause()
  br label %sleeploop
dowait:
  call void @wait_word(i64* %p, i64 %g2, i64* %vv)
  br label %sleeploop
woke:
  br i1 %worker, label %dec, label %plain
plain:
  ret i64 %g2
dec:
  atomicrmw sub i64* %slp, i64 1 seq_cst
  %wq = call i64* @slot(i64* %t, i64 53)
  %w0 = load atomic i64, i64* %wq monotonic, align 8
  %now2 = call i64 @qpc()
  %lat = sub i64 %now2, %w0
  %latp = call i64* @slot(i64* %t, i64 54)
  %l0 = load atomic i64, i64* %latp monotonic, align 8
  %dl = sub i64 %lat, %l0
  %dl8 = sdiv i64 %dl, 8
  %l1 = add i64 %l0, %dl8
  store atomic i64 %l1, i64* %latp monotonic, align 8
  %lo = load i64, i64* @spin_min
  %hi = load i64, i64* @spin_max
  %c1 = icmp slt i64 %l1, %lo
  %s1 = select i1 %c1, i64 %lo, i64 %l1
  %c2 = icmp sgt i64 %s1, %hi
  %s2 = select i1 %c2, i64 %hi, i64 %s1
  store atomic i64 %s2, i64* %spinp monotonic, align 8
  ret i64 %g2
got:
  ret i64 %g
}

; a worker: waits for the team's next job (an even gen not seen), counts itself in it (busy), takes
; parts; a gen moved on in between is not its job's
define internal void @worker(i8* %arg) {
entry:
  %t = bitcast i8* %arg to i64*
  %s = call %state* @state()
  %gp = call i64* @slot(i64* %t, i64 0)
  %busyp = call i64* @slot(i64* %t, i64 24)
  br label %wait
wait:
  %mine = phi i64 [0, %entry], [%mine, %retry], [%g, %done]
  %g = call i64 @wait_change(i64* %t, i64* %gp, i64 %mine, i1 true, i1 true)
  atomicrmw add i64* %busyp, i64 1 seq_cst
  %g2 = load atomic i64, i64* %gp seq_cst, align 8
  %still = icmp eq i64 %g2, %g
  br i1 %still, label %go, label %retry
retry:
  atomicrmw sub i64* %busyp, i64 1 seq_cst
  br label %wait
go:
  call void @run(i64* %t, %state* %s)
  br label %done
done:
  atomicrmw sub i64* %busyp, i64 1 seq_cst
  br label %wait
}

; a free team of k workers, else a new one
define internal i64* @acquire(i64 %k) {
entry:
  call void @lock_take()
  %h = load i64*, i64** @free_head
  br label %scan
scan:
  %prevlink = phi i64** [@free_head, %entry], [%lp, %next]
  %cur = phi i64* [%h, %entry], [%nx, %next]
  %none = icmp eq i64* %cur, null
  br i1 %none, label %make, label %test
test:
  %nwp = call i64* @slot(i64* %cur, i64 49)
  %nw = load i64, i64* %nwp
  %lnk = call i64* @slot(i64* %cur, i64 52)
  %lp = bitcast i64* %lnk to i64**
  %nx = load i64*, i64** %lp
  %fit = icmp eq i64 %nw, %k
  br i1 %fit, label %take, label %next
next:
  br label %scan
take:
  store i64* %nx, i64** %prevlink
  call void @lock_give()
  ret i64* %cur
make:
  call void @lock_give()
  %m = call i8* @malloc(i64 576)
  %mi = ptrtoint i8* %m to i64
  %al = add i64 %mi, 63
  %al2 = and i64 %al, -64
  %t = inttoptr i64 %al2 to i64*
  br label %zero
zero:
  %z = phi i64 [0, %make], [%z1, %zero]
  %zp = getelementptr i64, i64* %t, i64 %z
  store i64 0, i64* %zp
  %z1 = add i64 %z, 1
  %zmore = icmp slt i64 %z1, 64
  br i1 %zmore, label %zero, label %fill
fill:
  %nwq = call i64* @slot(i64* %t, i64 49)
  store i64 %k, i64* %nwq
  %sp = call i64* @slot(i64* %t, i64 51)
  %s0 = load i64, i64* @spin0
  store i64 %s0, i64* %sp
  %lq = call i64* @slot(i64* %t, i64 54)
  store i64 %s0, i64* %lq
  %t8 = bitcast i64* %t to i8*
  br label %spawn
spawn:
  %i = phi i64 [0, %fill], [%i1, %spawn]
  call void @spawn(i8* %t8)
  %i1 = add i64 %i, 1
  %imore = icmp slt i64 %i1, %k
  br i1 %imore, label %spawn, label %made
made:
  ret i64* %t
}

define internal void @release(i64* %t) {
  call void @lock_take()
  %h = load i64*, i64** @free_head
  %lnk = call i64* @slot(i64* %t, i64 52)
  %lp = bitcast i64* %lnk to i64**
  store i64* %h, i64** %lp
  store i64* %t, i64** @free_head
  call void @lock_give()
  ret void
}

; every part on this thread, thread id 0 throughout (OpenMP's team of one), nt its thread count
define internal void @alone(i8* %fn, i8** %args, i64* %dims, i64* %steps, i8* %data, i64 %inner,
                            i64 %ac, i64 %nt, %state* %s) {
entry:
  %buf = alloca i64, i64 64
  %t = getelementptr i64, i64* %buf, i64 0
  %fni = ptrtoint i8* %fn to i64
  %argsi = ptrtoint i8** %args to i64
  %dimsi = ptrtoint i64* %dims to i64
  %stepsi = ptrtoint i64* %steps to i64
  %datai = ptrtoint i8* %data to i64
  %p40 = call i64* @slot(i64* %t, i64 40)
  store i64 %fni, i64* %p40
  %p41 = call i64* @slot(i64* %t, i64 41)
  store i64 %argsi, i64* %p41
  %p42 = call i64* @slot(i64* %t, i64 42)
  store i64 %dimsi, i64* %p42
  %p43 = call i64* @slot(i64* %t, i64 43)
  store i64 %stepsi, i64* %p43
  %p44 = call i64* @slot(i64* %t, i64 44)
  store i64 %datai, i64* %p44
  %p45 = call i64* @slot(i64* %t, i64 45)
  store i64 %inner, i64* %p45
  %p46 = call i64* @slot(i64* %t, i64 46)
  store i64 %ac, i64* %p46
  %p47 = call i64* @slot(i64* %t, i64 47)
  store i64 1, i64* %p47
  %size = load i64, i64* %dims
  %p48 = call i64* @slot(i64* %t, i64 48)
  store i64 %size, i64* %p48                   ; (one thread, size parts: part() gives each thread 0)
  %arglen = add i64 %inner, 1
  %count = alloca i64, i64 %arglen
  %ac1 = add i64 %ac, 1
  %arrs = alloca i8*, i64 %ac1
  %f0 = getelementptr %state, %state* %s, i32 0, i32 0
  %f1 = getelementptr %state, %state* %s, i32 0, i32 1
  %otid = load i32, i32* %f0
  %ont = load i32, i32* %f1
  %nt32 = trunc i64 %nt to i32
  store i32 %nt32, i32* %f1
  br label %loop
loop:
  %r = phi i64 [0, %entry], [%r1, %body]
  %more = icmp slt i64 %r, %size
  br i1 %more, label %body, label %out
body:
  call void @part(i64* %t, i64 %r, i64 0, i64* %count, i8** %arrs, %state* %s)
  %r1 = add i64 %r, 1
  br label %loop
out:
  store i32 %otid, i32* %f0
  store i32 %ont, i32* %f1
  ret void
}

define internal i64 @wait_until(i64* %t, i64* %p, i64 %want) {
entry:
  br label %loop
loop:
  %v = load atomic i64, i64* %p seq_cst, align 8
  %ok = icmp eq i64 %v, %want
  br i1 %ok, label %out, label %w
w:
  call i64 @wait_change(i64* %t, i64* %p, i64 %v, i1 false, i1 false)
  br label %loop
out:
  ret i64 %v
}

; numba_parallel_for: the job posted to a team (gen made odd while no worker is still in the last
; one, the fields written, gen made even, sleepers woken), parts taken here too, then waited for
define void @fastplot_pool_parallel_for(i8* %fn, i8** %args, i64* %dims, i64* %steps, i8* %data,
                                        i64 %inner, i64 %ac, i32 %nthreads) {
entry:
  %s = call %state* @state()
  %f2 = getelementptr %state, %state* %s, i32 0, i32 2
  %inr = load i32, i32* %f2
  %size = load i64, i64* %dims
  %nt = sext i32 %nthreads to i64
  %nested = icmp ne i32 %inr, 0
  %one = icmp sle i64 %nt, 1
  %tiny = icmp sle i64 %size, 1
  %a = or i1 %nested, %one
  %solo = or i1 %a, %tiny
  br i1 %solo, label %serial, label %team
serial:
  %ntx = select i1 %nested, i64 1, i64 %nt
  call void @alone(i8* %fn, i8** %args, i64* %dims, i64* %steps, i8* %data, i64 %inner, i64 %ac,
                   i64 %ntx, %state* %s)
  ret void
team:
  %k = sub i64 %nt, 1
  %t = call i64* @acquire(i64 %k)
  %gp = call i64* @slot(i64* %t, i64 0)
  %g0 = load atomic i64, i64* %gp seq_cst, align 8
  %godd = add i64 %g0, 1
  store atomic i64 %godd, i64* %gp seq_cst, align 8
  %busyp = call i64* @slot(i64* %t, i64 24)
  br label %drain
drain:
  %b = load atomic i64, i64* %busyp seq_cst, align 8
  %idle = icmp eq i64 %b, 0
  br i1 %idle, label %post, label %drainwait
drainwait:
  call void @llvm.x86.sse2.pause()
  br label %drain
post:
  %fni = ptrtoint i8* %fn to i64
  %argsi = ptrtoint i8** %args to i64
  %dimsi = ptrtoint i64* %dims to i64
  %stepsi = ptrtoint i64* %steps to i64
  %datai = ptrtoint i8* %data to i64
  %p40 = call i64* @slot(i64* %t, i64 40)
  store i64 %fni, i64* %p40
  %p41 = call i64* @slot(i64* %t, i64 41)
  store i64 %argsi, i64* %p41
  %p42 = call i64* @slot(i64* %t, i64 42)
  store i64 %dimsi, i64* %p42
  %p43 = call i64* @slot(i64* %t, i64 43)
  store i64 %stepsi, i64* %p43
  %p44 = call i64* @slot(i64* %t, i64 44)
  store i64 %datai, i64* %p44
  %p45 = call i64* @slot(i64* %t, i64 45)
  store i64 %inner, i64* %p45
  %p46 = call i64* @slot(i64* %t, i64 46)
  store i64 %ac, i64* %p46
  %p47 = call i64* @slot(i64* %t, i64 47)
  store i64 %nt, i64* %p47
  %p48 = call i64* @slot(i64* %t, i64 48)
  store i64 %size, i64* %p48
  %nextp = call i64* @slot(i64* %t, i64 8)
  store atomic i64 0, i64* %nextp seq_cst, align 8
  %donep = call i64* @slot(i64* %t, i64 16)
  store atomic i64 0, i64* %donep seq_cst, align 8
  %msp = call i64* @slot(i64* %t, i64 50)
  store atomic i64 0, i64* %msp seq_cst, align 8
  %wq = call i64* @slot(i64* %t, i64 53)
  %now = call i64 @qpc()
  store atomic i64 %now, i64* %wq monotonic, align 8
  %gnew = add i64 %g0, 2
  store atomic i64 %gnew, i64* %gp seq_cst, align 8
  %slp = call i64* @slot(i64* %t, i64 32)
  %sl = load atomic i64, i64* %slp seq_cst, align 8
  %anysleep = icmp sgt i64 %sl, 0
  br i1 %anysleep, label %wake, label %mine
wake:
  call void @wake_all(i64* %gp)
  br label %mine
mine:
  call void @run(i64* %t, %state* %s)
  %nb = call i64 @blocks(i64 %size, i64 %nt)
  %d = load atomic i64, i64* %donep seq_cst, align 8
  %all = icmp eq i64 %d, %nb
  br i1 %all, label %end, label %await
await:
  store atomic i64 1, i64* %msp seq_cst, align 8
  call i64 @wait_until(i64* %t, i64* %donep, i64 %nb)
  store atomic i64 0, i64* %msp seq_cst, align 8
  br label %end
end:
  call void @release(i64* %t)
  ret void
}

define void @fastplot_pool_init(i32 %tls, i32 %default_nt, i64 %spin0, i64 %lo, i64 %hi) {
  store i32 %tls, i32* @tls
  store i32 %default_nt, i32* @default_nt
  store i64 %spin0, i64* @spin0
  store i64 %lo, i64* @spin_min
  store i64 %hi, i64* @spin_max
  ret void
}
'''
