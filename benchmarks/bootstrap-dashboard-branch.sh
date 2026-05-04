#!/usr/bin/env bash
# bootstrap-dashboard-branch.sh
#
# One-time setup: creates the mkl-umath-results branch that ASV uses to
# store benchmark results.  Run this once against the first commit you want
# to anchor results to.
#
# Usage:
#   SEED_SHA=<first-benchmarked-git-sha>  bash bootstrap-dashboard-branch.sh
#
# The script must be run from inside benchmarks/ (where asv.conf.json lives).
# The conda env with asv installed must already be active.

set -euo pipefail

RESULTS_BRANCH="mkl-umath-results"
SEED_SHA="${SEED_SHA:?ERROR: set SEED_SHA=<commit-sha> before running this script}"

echo "[bootstrap] Seeding results branch: ${RESULTS_BRANCH}"
echo "[bootstrap] Anchored to commit:     ${SEED_SHA}"

# Run a single quick pass to generate the first results JSON
asv run \
    --python=same \
    --quick \
    --show-stderr \
    --set-commit-hash "${SEED_SHA}" \
    HEAD

# Publish results to HTML (creates .asv/html/)
asv publish

# Push results to the dedicated branch
asv gh-pages \
    --rewrite \
    --no-push \
    --html-dir .asv/html

echo "[bootstrap] Done.  Push .asv/results to ${RESULTS_BRANCH} manually or"
echo "            configure asv gh-pages --push to automate."
