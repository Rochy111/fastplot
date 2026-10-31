"""python -m fastplot warm | tune | hardware

    warm       compile every kernel now (installing already does; several at a time)
    tune       time draws on this machine and keep the best settings for it
    hardware   what fastplot found on this machine, and what it uses
"""
import json
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ''
    import fastplot
    if cmd == 'warm':
        fastplot.warm(verbose=True)
    elif cmd == 'tune':
        r = fastplot.tune()
        print('draw threads: %d' % r['threads'])
        if 'gpu_work' in r:
            print("GPU from: %d (pixels + grid points)" % r['gpu_work'])
        print('kept in %s (%.1f s)' % (r.get('saved_to'), r['seconds']))
    elif cmd == 'hardware':
        print(json.dumps(fastplot.hardware(), indent=1, default=str))
    else:
        print(__doc__.strip())
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
