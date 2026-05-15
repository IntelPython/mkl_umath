# Trigger MKL patching once per ASV worker process.
# ASV uses --launch-method spawn in CI, so each worker is a fresh process
# and this runs exactly once before any benchmark is collected or timed.
from . import _patch_setup  # noqa: F401
