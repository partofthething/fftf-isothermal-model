#!/usr/bin/env bash
# Reproduce the DIF3D output-reading timing table: the ARMI CCCC matrix-read change
# crossed with the armicontrib-dif3d region-indexing change, read time for a full FFTF core.
#
# Each combination is checked out into a temporary git worktree and run with `python -S`
# with only the worktrees and the environment's site-packages on PYTHONPATH. -S skips .pth
# files, so editable installs of armi or armicontrib-dif3d in the environment can't take over.
#
# Usage (from the root of this repo):
#     profiling/dif3d_read_table.sh
#
# Override with environment variables if needed:
#     PYTHON         Python with ARMI's dependencies and jinja2 installed
#     ARMI_REPO      ARMI clone containing the commits below
#     PLUGIN_REPO    armicontrib-dif3d clone containing the commits below
#     ARMI_BEFORE / ARMI_AFTER        value-by-value vs whole-matrix CCCC reads
#     PLUGIN_BEFORE / PLUGIN_AFTER    per-block mesh search vs region indexing
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARMI_REPO=$(cd "${ARMI_REPO:-$HERE/../../armi}" && pwd)
PLUGIN_REPO=$(cd "${PLUGIN_REPO:-$HERE/../../armicontrib-dif3d}" && pwd)
PYTHON=${PYTHON:-$ARMI_REPO/.venv/bin/python}

ARMI_BEFORE=${ARMI_BEFORE:-f1df51d4~1}  # main before "Read binary CCCC matrices in one call"
ARMI_AFTER=${ARMI_AFTER:-f1df51d4}      # "Read binary CCCC matrices in one call"
PLUGIN_BEFORE=${PLUGIN_BEFORE:-b525362} # "Upgrade to work with ARMI 0.7"
PLUGIN_AFTER=${PLUGIN_AFTER:-7954de8}   # "Speed up applying DIF3D output to the reactor"

WORK=$(mktemp -d -t dif3d-read-table-XXXXXX)
cleanup() {
    for name in armi-before armi-after; do
        git -C "$ARMI_REPO" worktree remove --force "$WORK/$name" 2>/dev/null || true
    done
    for name in plugin-before plugin-after; do
        git -C "$PLUGIN_REPO" worktree remove --force "$WORK/$name" 2>/dev/null || true
    done
    rm -rf "$WORK"
}
trap cleanup EXIT

git -C "$ARMI_REPO" worktree add -q --detach "$WORK/armi-before" "$ARMI_BEFORE"
git -C "$ARMI_REPO" worktree add -q --detach "$WORK/armi-after" "$ARMI_AFTER"
git -C "$PLUGIN_REPO" worktree add -q --detach "$WORK/plugin-before" "$PLUGIN_BEFORE"
git -C "$PLUGIN_REPO" worktree add -q --detach "$WORK/plugin-after" "$PLUGIN_AFTER"

SITE=$("$PYTHON" -c 'import sysconfig; p = sysconfig.get_paths(); print(p["purelib"] + ":" + p["platlib"])')

# prints the "read output" best time in seconds for one ARMI/plugin combination
readTime() {
    local armi=$1 plugin=$2 path="$WORK/$1:$WORK/$2:$SITE" out
    # make sure the worktrees are what actually gets imported
    PYTHONPATH=$path "$PYTHON" -S -c "
import armi, armicontrib.dif3d.outputReaders as o
assert armi.__file__.startswith('$WORK/$armi/'), armi.__file__
assert o.__file__.startswith('$WORK/$plugin/'), o.__file__
" >&2
    out=$(cd "$HERE/.." && PYTHONPATH=$path "$PYTHON" -S profiling/bench_dif3d.py --workdir "$WORK/run-$armi-$plugin" 2>&1) || {
        echo "$out" >&2
        return 1
    }
    echo "$armi + $plugin: $(grep -E '^(write input|read output)' <<<"$out" | tr '\n' ' ')" >&2
    grep '^read output' <<<"$out" | awk '{print $4}'
}

ORIG_BEFORE=$(readTime armi-before plugin-before)
ORIG_AFTER=$(readTime armi-after plugin-before)
INDEXED_BEFORE=$(readTime armi-before plugin-after)
INDEXED_AFTER=$(readTime armi-after plugin-after)

cat <<EOF

| armicontrib-dif3d plugin reader | Before (s) | After (s) |
|---|---:|---:|
| Original | $ORIG_BEFORE | $ORIG_AFTER |
| With region indexing (separate plugin change) | $INDEXED_BEFORE | $INDEXED_AFTER |
EOF
