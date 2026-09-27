"""Self-check: exit this process if its phys_footprint exceeds MEMLIMIT_MB (checked every 20 s via `footprint`)."""
import os, subprocess, sys, threading, time


def _fp_mb(pid):
    try:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return 0
    for line in out.splitlines():
        if "phys_footprint:" in line:
            v = line.split(":")[1].split()
            x = float(v[0])
            return x * 1024 if v[1].startswith("GB") else x / 1024 if v[1].startswith("KB") else x
    return 0


def start(default_mb):
    lim = float(os.environ.get("MEMLIMIT_MB", default_mb))
    pid = os.getpid()

    def loop():
        while True:
            time.sleep(20)
            mb = _fp_mb(pid)
            if mb > lim:
                sys.stderr.write(f"memguard: footprint {mb:.0f} MB > limit {lim:.0f} MB, exiting\n"); sys.stderr.flush()
                os._exit(3)
    threading.Thread(target=loop, daemon=True).start()
