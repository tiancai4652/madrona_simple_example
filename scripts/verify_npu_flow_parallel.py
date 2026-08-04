"""Compatibility entry point for the Stage-B system integration checks.

FakeSystem was removed in Stage B, so the former synthetic flow-id and pool
tests are now covered by the real Chakra SEND/RECV integration scenario.
"""

import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("verify_system_integration.py")

print(
    "[INFO] FakeSystem verification was replaced by Stage-B integration "
    "verification.",
    flush=True,
)
runpy.run_path(str(SCRIPT), run_name="__main__")
