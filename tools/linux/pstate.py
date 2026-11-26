#!/usr/bin/env python3
"""pstate.py <render node> <seconds> [pstate]: hold an amdgpu "stable pstate" (default 4 = PEAK) on a
context for <seconds>, printing the GPU clock before, during and after. No privilege needed."""
import sys, os, fcntl, struct, time, glob
node, secs = sys.argv[1], float(sys.argv[2]); want = int(sys.argv[3]) if len(sys.argv) > 3 else 4
IOCTL = 0xC0106442  # DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union drm_amdgpu_ctx)
fd = os.open(node, os.O_RDWR)
minor = os.minor(os.fstat(fd).st_rdev)
H = glob.glob(f'/sys/dev/char/226:{minor}/device/hwmon/hwmon*')[0]
clk = lambda: int(open(H + '/freq1_input').read()) / 1e6
def ctx(op, flags=0, cid=0):
    b = bytearray(struct.pack('IIIi', op, flags, cid, 0)); fcntl.ioctl(fd, IOCTL, b); return struct.unpack('II', b[:8])
print('before', [round(clk()) for _ in range(3)])
cid = ctx(1)[0]
print('ctx', cid, 'current pstate', ctx(5, 0, cid)[0])
ctx(6, want, cid)
print('set ->', ctx(5, 0, cid)[0])
t0 = time.time()
while time.time() - t0 < secs:
    print(f'  {time.time()-t0:5.2f}s {clk():.0f} MHz', flush=True); time.sleep(max(0.02, secs / 10))
ctx(2, 0, cid); time.sleep(0.3)
print('after free', [round(clk()) for _ in range(3)])
