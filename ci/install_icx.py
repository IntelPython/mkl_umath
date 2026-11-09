"""
Installs Intel's icx for mkl_umath's Windows wheel builds.

cibuildwheel runs this before the builds, as pyproject.toml says. It fetches
micromamba, checked against its published sha256, and creates PREFIX with
conda-forge's dpcpp_win-64, the compiler the conda recipe builds with.

Usage:
  python ci/install_icx.py PREFIX
"""

import hashlib
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MICROMAMBA = (
    "https://github.com/mamba-org/micromamba-releases/releases/download/"
    "2.10.0-0/micromamba-win-64"
)
MICROMAMBA_SHA256 = (
    "f8b4284bd4eb1134122eaf93b3ea6c7be567e596de5829e1a6f478e34e41974a"
)
COMPILER = "dpcpp_win-64=2026.2.0"


def main() -> int:
    """Creates PREFIX with the compiler."""
    prefix = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as scratch:
        micromamba = Path(scratch) / "micromamba.exe"
        with urllib.request.urlopen(MICROMAMBA, timeout=300) as response:
            micromamba.write_bytes(response.read())
        digest = hashlib.sha256(micromamba.read_bytes()).hexdigest()
        if digest != MICROMAMBA_SHA256:
            print(
                f"install_icx: micromamba has sha256 {digest}, "
                f"not {MICROMAMBA_SHA256}",
                file=sys.stderr,
            )
            return 1
        command = [
            str(micromamba),
            "create",
            "-y",
            "-r",
            str(Path(scratch) / "root"),
            "-p",
            str(prefix),
            "--override-channels",
            "-c",
            "conda-forge",
            COMPILER,
        ]
        print("+ " + " ".join(command), flush=True)
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
