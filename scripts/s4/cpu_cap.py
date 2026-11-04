#!/usr/bin/env python3
"""CPU cap by pausing jobs .

    cpu_cap.py [slots=6] [period_s=20]

Every job here is single-threaded (OMP_NUM_THREADS=1, ~100 % CPU each), so a 600 % cap = at most 6 running jobs. Every
`period_s` seconds: list this user's python3.12 job processes (run_models.py trainings; evals / checks / tables), keep the
first `slots` running and SIGSTOP the rest, SIGCONT-ing paused ones as slots free. Order: short jobs (anything that is not
run_models.py) first, then trainings oldest first (FIFO, so a paused training never starves). A paused process resumes
exactly where it stopped. On SIGTERM / SIGINT / any crash, every process it paused is resumed before it exits.
Log: logs/cpu_cap.log (one line per change + a heartbeat every 10 min). Stop: kill <pid> (never kill -9: that would leave
paused jobs paused; if that happens, `kill -CONT` the 'T'-state python3.12 processes)."""
import os, sys, time, signal, subprocess, datetime

SLOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
LOG = 'logs/cpu_cap.log'
JOB_MARKERS = ('run_models.py', 'seeded_eval.py', 'test_film_matched.py', 'scripts/theory/', 'scripts/s4/', '/tmp/doseh')
me = os.getpid()
paused_by_me = set()


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n')


def etime_s(s):
    """ps etime [[dd-]hh:]mm:ss -> seconds"""
    days, _, rest = s.partition('-') if '-' in s else ('0', '', s)
    parts = [int(x) for x in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def jobs():
    out = subprocess.run(['ps', '-x', '-o', 'pid=,stat=,etime=,pcpu=,command='], capture_output=True, text=True).stdout
    res = []
    for line in out.splitlines():
        f = line.split(None, 4)
        if len(f) < 5:
            continue
        pid, stat, et, pcpu, cmd = int(f[0]), f[1], f[2], float(f[3].replace(',', '.')), f[4]
        if pid == me or 'python3.12' not in cmd or 'multiprocessing' in cmd or 'cpu_cap.py' in cmd:
            continue
        if not any(m in cmd for m in JOB_MARKERS):
            continue
        res.append(dict(pid=pid, stopped=stat.startswith('T'), age=etime_s(et), pcpu=pcpu,
                        short=('run_models.py' not in cmd), cmd=cmd))
    return res


def resume_all(*_):
    for pid in list(paused_by_me):
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    log(f'exit: resumed {len(paused_by_me)} paused processes')
    paused_by_me.clear()
    sys.exit(0)


signal.signal(signal.SIGTERM, resume_all)
signal.signal(signal.SIGINT, resume_all)
log(f'start: pid {me}, {SLOTS} slots (~{100 * SLOTS}% CPU), period {PERIOD:g} s')
last_beat = 0.0
try:
    while True:
        J = sorted(jobs(), key=lambda j: (not j['short'], -j['age']))   # short jobs first, then oldest first
        run, stop = J[:SLOTS], J[SLOTS:]
        for j in run:
            if j['stopped']:
                try:
                    os.kill(j['pid'], signal.SIGCONT)
                except ProcessLookupError:   # exited since the listing
                    pass
                paused_by_me.discard(j['pid'])
                log(f"resume {j['pid']} ({'short' if j['short'] else 'train'}, age {j['age']} s)")
        for j in stop:
            if not j['stopped']:
                try:
                    os.kill(j['pid'], signal.SIGSTOP)
                except ProcessLookupError:
                    continue
                paused_by_me.add(j['pid'])
                log(f"pause  {j['pid']} ({'short' if j['short'] else 'train'}, age {j['age']} s)")
        paused_by_me &= {j['pid'] for j in J}
        if time.time() - last_beat > 600:
            log(f'heartbeat: {len(run)} running, {len(stop)} paused, running %CPU {sum(j["pcpu"] for j in run):.0f}')
            last_beat = time.time()
        time.sleep(PERIOD)
except Exception as e:
    log(f'CRASH {e!r}')
    resume_all()
