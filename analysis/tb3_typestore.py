"""tb3_typestore.py - Build a rosbags typestore from the real tb3_interfaces .msg files.

Single source of truth: the .msg files in <repo>/tb3_interfaces/msg. Used by
make_synthetic_run.py to fabricate ROS2-CDR bags for testing the pipeline
without hardware. The analysis reader (mcap-ros2-support) does NOT need this —
it decodes schemas embedded in the bags.
"""

from __future__ import annotations

import functools
from pathlib import Path

from rosbags.typesys import Stores, get_types_from_msg, get_typestore


def find_interfaces_dir(start: Path | None = None) -> Path:
    """Walk up from `start` (or this file) to locate tb3_interfaces/msg."""
    here = (start or Path(__file__)).resolve()
    candidates = (
        Path("robot") / "src" / "tb3_interfaces" / "msg",
        Path("src") / "tb3_interfaces" / "msg",
        Path("tb3_interfaces") / "msg",
    )
    for parent in [here, *here.parents]:
        for rel in candidates:
            cand = parent / rel
            if cand.is_dir():
                return cand
    raise FileNotFoundError(f"tb3_interfaces/msg not found upward from {here}")


@functools.lru_cache(maxsize=1)
def get_tb3_typestore():
    """Return a ROS2_HUMBLE typestore with all tb3_interfaces messages registered.

    Registration is order-independent: a few passes resolve intra-package
    dependencies (e.g. MoveToGridResult -> GridPose).
    """
    ts = get_typestore(Stores.ROS2_HUMBLE)
    msg_dir = find_interfaces_dir()
    files = sorted(msg_dir.glob("*.msg"))
    pending = {f"tb3_interfaces/msg/{f.stem}": f for f in files}
    for _ in range(5):
        if not pending:
            break
        for name in list(pending):
            try:
                ts.register(get_types_from_msg(pending[name].read_text(), name))
                del pending[name]
            except Exception:
                continue  # unresolved dependency; retry next pass
    if pending:
        raise RuntimeError(f"could not register: {sorted(pending)}")
    return ts
