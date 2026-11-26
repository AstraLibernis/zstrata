#!/usr/bin/env python3
"""clkrun.py <zstrata args...>: run zstrata, sample the dGPU clock/busy/power every 10 ms,
print each row with the clock (min/median/max MHz) seen while that row was being measured."""
import sys, subprocess, threading, time, statistics as st, os, glob
H = glob.glob('/sys/class/drm/card1/device/hwmon/hwmon*')[0]
samples = []; stop = False
def sampler():
    f = open(H + '/freq1_input'); b = open('/sys/class/drm/card1/device/gpu_busy_percent'); p = open(H + '/power1_average')
    while not stop:
        t = time.monotonic()
        try:
            f.seek(0); b.seek(0); p.seek(0)
            samples.append((t, int(f.read()) / 1e6, int(b.read()), int(p.read()) / 1e6))
        except Exception: pass
        time.sleep(0.01)
th = threading.Thread(target=sampler, daemon=True); th.start()
env = dict(os.environ)
exe = os.path.expanduser(os.environ.get('ZSTRATA', '~/workspace/zstrata/zig-out/bin/zstrata'))
pr = subprocess.Popen([exe] + sys.argv[1:], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
last = time.monotonic(); i0 = 0
for line in pr.stderr:
    now = time.monotonic()
    w = [s for s in samples[i0:] if last <= s[0] <= now]
    i0 = max(0, len(samples) - 2)
    if w:
        mhz = [s[1] for s in w]
        print(f"{line.rstrip():<86} | {now-last:5.2f}s clk {min(mhz):5.0f}/{st.median(mhz):5.0f}/{max(mhz):5.0f} MHz busy {st.median([s[2] for s in w]):3.0f}% {st.median([s[3] for s in w]):4.0f} W", flush=True)
    else:
        print(f"{line.rstrip():<86} | {now-last:5.2f}s", flush=True)
    last = now
stop = True; sys.exit(pr.wait())
