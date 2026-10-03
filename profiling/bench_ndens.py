"""
Benchmark reading and writing number densities on every block of a full core.

* **get**: ``b.getNumberDensities()`` on every block (all nuclides).
* **set**: ``b.setNumberDensities(...)`` on every block, writing back values
  scaled by a small factor so the data actually changes.
* **temp**: ``c.setTemperature(...)`` on every component, alternating -/+1 C each
  repeat, which exercises material thermal expansion.

Models:

* **fftf**: this repo's FFTF model. All ``Custom`` materials at fixed temperatures,
  so there is no thermal expansion.
* **sodium**: ARMI's sodium-cooled hex test reactor, grown to full core. Real
  materials (UZr, HT9, sodium, B4C) heated from input to hot temperatures.

Usage::

    python profiling/bench_ndens.py [--model fftf|sodium] [--ops get,set,temp]
        [--repeats N] [--dump out.npz] [--check ref.npz]

``--dump`` writes the final component-level number densities so a later run
(e.g. after optimizing ARMI) can be compared against it with ``--check``.
"""

import argparse
import os
import pathlib
import time

import numpy as np

import armi

if not armi.isConfigured():
    armi.configure(permissive=True)

from armi import runLog, settings, testing  # noqa: E402
from armi.reactor import reactors  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SETTINGS = HERE.parent / "FFTF.yaml"
MODELS = {
    "fftf": SETTINGS,
    "sodium": pathlib.Path(testing.__file__).parent / "reactors" / "sodiumHexReactor" / "armiRun.yaml",
}
FACTOR = 1.0001


def loadReactor(settingsPath=SETTINGS):
    runLog.setVerbosity("warning")
    settingsPath = pathlib.Path(settingsPath)
    cwd = os.getcwd()
    os.chdir(settingsPath.parent)
    try:
        cs = settings.Settings(str(settingsPath))
        r = reactors.loadFromCs(cs)
        if not r.core.isFullCore:
            r.core.growToFullCore(cs)
    finally:
        os.chdir(cwd)
    return cs, r


def loadCore():
    return loadReactor()[1].core


def readAll(blocks):
    return [b.getNumberDensities() for b in blocks]


def setAll(blocks, allNdens):
    for b, ndens in zip(blocks, allNdens):
        b.setNumberDensities({nuc: val * FACTOR for nuc, val in ndens.items()})


def changeTemperatures(blocks, delta):
    for b in blocks:
        for c in b:
            c.setTemperature(c.temperatureInC + delta)


def componentState(core):
    """Flatten every component's nuclide vector into a deterministic dict."""
    state = {}
    for bi, b in enumerate(core.iterBlocks()):
        for ci, c in enumerate(b):
            nucs = c.p.nuclides
            if nucs is None or len(nucs) == 0:
                continue
            order = np.argsort(nucs)
            key = f"{bi}:{ci}:{c.name}"
            state[key + ":n"] = np.asarray(nucs)[order]
            state[key + ":d"] = np.asarray(c.p.numberDensities)[order]
    return state


def compare(state, refPath):
    ref = np.load(refPath)
    assert set(ref.files) == set(state), "component layout differs from reference"
    worst = 0.0
    for k in ref.files:
        if k.endswith(":n"):
            assert (ref[k] == state[k]).all(), f"nuclide mismatch at {k}"
        else:
            a, b = ref[k], state[k]
            rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
            worst = max(worst, float(rel.max()) if rel.size else 0.0)
    print(f"max relative difference vs reference: {worst:.3e}")
    assert worst < 1e-12, "number densities diverged from reference"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", choices=sorted(MODELS), default="fftf")
    parser.add_argument("--ops", default="get,set", help="comma-separated subset of get,set,temp")
    parser.add_argument("--dump")
    parser.add_argument("--check")
    args = parser.parse_args()

    ops = args.ops.split(",")
    t0 = time.perf_counter()
    core = loadReactor(MODELS[args.model])[1].core
    blocks = core.getBlocks()
    print(f"load: {time.perf_counter() - t0:.2f} s")
    nucs = {n for b in blocks for n in b.getNuclides()}
    print(f"{len(core)} assemblies, {len(blocks)} blocks, {sum(len(b) for b in blocks)} components, {len(nucs)} nuclides")

    times = {op: [] for op in ops}
    allNdens = None
    for i in range(args.repeats):
        if "get" in ops or allNdens is None:
            t = time.perf_counter()
            allNdens = readAll(blocks)
            if "get" in ops:
                times["get"].append(time.perf_counter() - t)
        if "set" in ops:
            t = time.perf_counter()
            setAll(blocks, allNdens)
            times["set"].append(time.perf_counter() - t)
        if "temp" in ops:
            t = time.perf_counter()
            changeTemperatures(blocks, -1.0 if i % 2 == 0 else 1.0)
            times["temp"].append(time.perf_counter() - t)

    for op, opTimes in times.items():
        print(f"{op}: best {min(opTimes):.3f} s  mean {np.mean(opTimes):.3f} s  ({len(opTimes)} runs)")

    if args.dump or args.check:
        state = componentState(core)
        if args.dump:
            np.savez(args.dump, **state)
            print(f"wrote {args.dump}")
        if args.check:
            compare(state, args.check)


if __name__ == "__main__":
    main()
