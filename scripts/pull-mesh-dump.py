#!/usr/bin/env python3
"""Pull the frames that actually entered the mesh and show how much depth they had.

The phone writes them under Android/data/.../files/mesh-dump: index.csv plus one
little-endian float32 depth image per frame. A public RGB-D set would not show
this camera's holes.
"""

import csv
import math
import struct
import subprocess
import sys
from pathlib import Path

ADB = "/home/alex/dev/android-sdk/platform-tools/adb"
REMOTE = "/sdcard/Android/data/com.pocketscanner.capture/files/mesh-dump"
OUT = Path("/tmp/pocket-mesh-dump")


def adb(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([ADB, *args], check=False, text=True, capture_output=True)


def main() -> int:
    devices = adb("devices")
    serials = [line.split()[0] for line in devices.stdout.splitlines() if line.endswith("\tdevice")]
    if not serials:
        print("phone is not connected")
        return 1
    serial = serials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    pulled = adb("-s", serial, "pull", REMOTE, str(OUT))
    if pulled.returncode != 0:
        print(pulled.stderr.strip() or pulled.stdout.strip() or "pull failed")
        return 1
    root = OUT / "mesh-dump"
    index = root / "index.csv"
    if not index.exists():
        print(f"no index at {index}")
        return 1
    rows = list(csv.DictReader(index.open()))
    if not rows:
        print("index is empty: walk a little after tracking starts")
        return 0
    ratios = []
    print(f"{'id':>12} {'kind':<6} {'valid':>7} {'step m':>8}")
    previous = None
    preview = root / "preview"
    preview.mkdir(exist_ok=True)
    for row in rows:
        valid = int(row["valid"])
        pixels = int(row["pixels"])
        ratio = valid / pixels if pixels else 0.0
        ratios.append(ratio)
        position = (float(row["tx"]), float(row["ty"]), float(row["tz"]))
        step = ""
        if previous is not None:
            step = f"{math.dist(previous, position):.2f}"
        previous = position
        print(f"{row['id']:>12} {row['kind']:<6} {ratio:7.0%} {step:>8}")
        write_pgm(preview / f"{row['id']}.pgm", int(row["width"]), int(row["height"]), root / f"{row['id']}.f32")
    ratios.sort()
    mid = ratios[len(ratios) // 2]
    print(f"frames {len(rows)}  valid depth min {ratios[0]:.0%}  median {mid:.0%}  max {ratios[-1]:.0%}")
    print(f"depth pictures: {preview}")
    if mid < 0.5:
        print("the frames going into the mesh are already mostly empty, so the pieces are the depth")
    else:
        print("the frames are mostly full, so the pieces appear while fusing them")
    return 0


def write_pgm(path: Path, width: int, height: int, depth_path: Path) -> None:
    if not depth_path.exists() or width <= 0 or height <= 0:
        return
    raw = depth_path.read_bytes()
    count = min(width * height, len(raw) // 4)
    values = struct.unpack_from("<" + "f" * count, raw)
    usable = [value for value in values if value > 0.05]
    far = max(usable) if usable else 1.0
    gray = bytearray(width * height)
    for index, value in enumerate(values):
        if value > 0.05:
            gray[index] = max(1, min(255, int(255 * value / far)))
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(gray))


if __name__ == "__main__":
    sys.exit(main())
