"""
Benchmark writing a DIF3D input for the FFTF core and reading DIF3D results back in.

There's no DIF3D run here. Instead, output files (GEODST, LABELS, RTFLUX, PWDINT,
PKEDIT, DIF3D, and the stdout REGION TOTALS table) are fabricated with dummy data on
a mesh consistent with the core: hex i/j indices radially and the core-wide axial
mesh axially, so a block spanning several axial mesh intervals maps to several
meshes, like a real nodal hex-z run. The binary files are produced by reading the
plugin's test fixtures, resizing their data, and writing them back out, so the format
metadata stays valid.

Requires the ``armicontrib-dif3d`` plugin to be installed.

Usage::

    python profiling/bench_dif3d.py [--repeats N] [--workdir DIR]
"""

import argparse
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

import armi
from armicontrib.dif3d.tests.dif3dTestingApp import Dif3dTestingApp

armi.configure(Dif3dTestingApp())

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_ndens  # noqa: E402

from armi.nuclearDataIO.cccc import geodst, labels, pwdint, rtflux  # noqa: E402
from armi.physics.neutronics.settings import CONF_NEUTRONICS_KERNEL  # noqa: E402
from armicontrib.dif3d import executionOptions, inputWriters, outputReaders  # noqa: E402
from armicontrib.dif3d.binaryIO import dif3dFile, pkedit  # noqa: E402
from armicontrib.dif3d.tests import FIXTURE_DIR  # noqa: E402

NUM_GROUPS = 33
READ_PARAMS = ("pdens", "ppdens", "power", "mgFlux", "flux", "fluxPeak")
REGIONS_PER_STDOUT_LINE = 10


def makeOptions(cs, r):
    cs = cs.modified(newSettings={CONF_NEUTRONICS_KERNEL: "DIF3D-Nodal"})
    opts = executionOptions.Dif3dOptions("fftf-dif3d")
    opts.fromUserSettings(cs)
    opts.resolveDerivedOptions()
    opts.fromReactor(r)
    return opts


def writeInput(r, opts):
    writer = inputWriters.Dif3dWriter(r, opts)
    with open(opts.inputFile, "w") as stream:
        writer.write(stream)


def readOutput(r, opts):
    reader = outputReaders.Dif3dReader(opts)
    reader.apply(r)


def _setMetadata(metadata, values):
    for key, val in values.items():
        metadata[key] = val


def _meshMap(core):
    """Map each block to the (i, j, k) coarse meshes it covers."""
    assems = list(core)
    ijs = np.array([a.spatialLocator.getCompleteIndices()[:2] for a in assems])
    ijs -= ijs.min(axis=0)
    ni, nj = ijs.max(axis=0) + 1

    zmesh = np.asarray(core.findAllAxialMeshPoints())
    midpoints = 0.5 * (zmesh[1:] + zmesh[:-1])
    nk = len(midpoints)

    blockMeshes = []
    for a, (i, j) in zip(assems, ijs):
        for b in a:
            ks = np.where((midpoints >= b.p.zbottom) & (midpoints < b.p.ztop))[0]
            blockMeshes.append((b, int(i), int(j), ks))
    return (int(ni), int(nj), nk), zmesh, blockMeshes


def fabricateOutputs(r, opts, seed=0):
    """Write dummy DIF3D output files for this core into the current directory."""
    rng = np.random.default_rng(seed)
    (ni, nj, nk), zmesh, blockMeshes = _meshMap(r.core)
    nreg = len(blockMeshes)

    coarse = np.zeros((ni, nj, nk), dtype=np.int32)
    regionLabels = []
    for reg, (b, i, j, ks) in enumerate(blockMeshes, start=1):
        coarse[i, j, ks] = reg
        regionLabels.append(inputWriters.getDIF3DStyleLocatorLabel(b))

    geom = geodst.readBinary(os.path.join(FIXTURE_DIR, geodst.GEODST))
    _setMetadata(geom.metadata, {"NZONE": nreg, "NREG": nreg, "NCINTI": ni, "NCINTJ": nj, "NCINTK": nk, "NINTI": ni, "NINTJ": nj, "NINTK": nk})
    geom.xmesh = np.arange(ni + 1, dtype=float)
    geom.ymesh = np.arange(nj + 1, dtype=float)
    geom.zmesh = zmesh
    geom.iintervals = np.ones(ni, dtype=int)
    geom.jintervals = np.ones(nj, dtype=int)
    geom.kintervals = np.ones(nk, dtype=int)
    geom.regionVolumes = np.array([b.getVolume() for b, *_ in blockMeshes])
    geom.zoneClassifications = np.ones(nreg, dtype=int)
    geom.regionZoneNumber = np.arange(1, nreg + 1)
    geom.coarseMeshRegions = coarse
    geodst.writeBinary(geom, geodst.GEODST)

    lab = labels.readBinary(os.path.join(FIXTURE_DIR, labels.LABELS))
    _setMetadata(lab.metadata, {"numZones": nreg, "numRegions": nreg})
    lab.zoneLabels = regionLabels
    lab.regionLabels = regionLabels
    labels.writeBinary(lab, labels.LABELS)

    flux = rtflux.RtfluxStream.readBinary(os.path.join(FIXTURE_DIR, rtflux.RTFLUX))
    _setMetadata(flux.metadata, {"NGROUP": NUM_GROUPS, "NINTI": ni, "NINTJ": nj, "NINTK": nk, "NBLOK": 1})
    flux.groupFluxes = rng.uniform(1e13, 1e15, size=(ni, nj, nk, NUM_GROUPS))
    rtflux.RtfluxStream.writeBinary(flux, rtflux.RTFLUX)

    power = pwdint.PwdintStream.readBinary(os.path.join(FIXTURE_DIR, pwdint.PWDINT))
    _setMetadata(power.metadata, {"NINTI": ni, "NINTJ": nj, "NINTK": nk, "NBLOK": 1})
    power.powerDensity = rng.uniform(0.0, 300.0, size=(ni, nj, nk)).astype(np.float32)
    pwdint.PwdintStream.writeBinary(power, pwdint.PWDINT)

    peak = pkedit.PkeditStream.readBinary(os.path.join(FIXTURE_DIR, pkedit.PKEDIT))
    _setMetadata(peak.metadata, {"IM": ni, "JM": nj, "KM": nk, "NJBLOK": 1})
    peak.peakPowerDensity = power.powerDensity.astype(np.float64) * 1.3
    pkedit.PkeditStream.writeBinary(peak, pkedit.PKEDIT)

    # keff, convergence state, etc. don't depend on the geometry
    d3d = dif3dFile.Dif3dStream.readBinary(os.path.join(FIXTURE_DIR, dif3dFile.DIF3D))
    dif3dFile.Dif3dStream.writeBinary(d3d, dif3dFile.DIF3D)
    # the flux file keff must be consistent with the problem keff
    flux.metadata["EFFK"] = d3d.keff
    rtflux.RtfluxStream.writeBinary(flux, rtflux.RTFLUX)

    _writeStdout(opts.outputFile, regionLabels, rng)


def _writeStdout(path, regionLabels, rng):
    """Write just enough of a DIF3D stdout file for the REGION TOTALS peak flux reader."""
    with open(path, "w") as f:
        f.write("0" + 55 * " " + "REGION TOTALS\n")
        for start in range(0, len(regionLabels), REGIONS_PER_STDOUT_LINE):
            chunk = regionLabels[start : start + REGIONS_PER_STDOUT_LINE]
            # 15-char prefix, then 13-char windows of zone number and region label
            f.write("     REGION    ")
            f.write("".join(f"{start + n + 1:6d} {label:6s}" for n, label in enumerate(chunk)) + "\n")
            f.write(" PEAK FLUX  " + "".join(f" {v:12.5E}" for v in rng.uniform(1e14, 1e16, len(chunk))) + "\n")
        f.write("0" + 40 * " " + "REACTION INTEGRALS IN DIRECTION OF CALCULATION\n")


def readState(blocks):
    """Collect the parameters the reader sets on every block."""
    return {p: np.array([np.atleast_1d(b.p[p]) for b in blocks]) for p in READ_PARAMS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workdir")
    parser.add_argument("--dump", help="save block parameters set by the reader to this .npz")
    parser.add_argument("--check", help="compare block parameters set by the reader to this .npz")
    args = parser.parse_args()

    cs, r = bench_ndens.loadReactor()
    # the operator normally sets this at each time node
    r.core.p.power = cs["power"]
    blocks = r.core.getBlocks()
    print(f"{len(r.core)} assemblies, {len(blocks)} blocks")

    dump = args.dump and os.path.abspath(args.dump)
    check = args.check and os.path.abspath(args.check)
    workdir = pathlib.Path(args.workdir or tempfile.mkdtemp(prefix="bench-dif3d-"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    opts = makeOptions(cs, r)

    fabricateOutputs(r, opts)
    (ni, nj, nk), _, _ = _meshMap(r.core)
    print(f"fabricated outputs on a {ni}x{nj}x{nk} mesh x {NUM_GROUPS} groups in {workdir}")

    writeTimes, readTimes = [], []
    for _ in range(args.repeats):
        t = time.perf_counter()
        writeInput(r, opts)
        writeTimes.append(time.perf_counter() - t)
        t = time.perf_counter()
        readOutput(r, opts)
        readTimes.append(time.perf_counter() - t)

    size = os.path.getsize(opts.inputFile) / 1e6
    print(f"write input: best {min(writeTimes):.3f} s  mean {np.mean(writeTimes):.3f} s  ({size:.1f} MB)")
    print(f"read output: best {min(readTimes):.3f} s  mean {np.mean(readTimes):.3f} s")
    state = readState(blocks)
    if dump:
        np.savez(dump, **state)
        print(f"wrote {dump}")
    if check:
        ref = np.load(check)
        for p in READ_PARAMS:
            np.testing.assert_array_equal(state[p], ref[p], err_msg=p)
        print(f"block parameters identical to {check}")
    b = blocks[len(blocks) // 2]
    print(f"sample {b}: flux={b.p.flux:.4e} pdens={b.p.pdens:.4e} fluxPeak={b.p.fluxPeak:.4e} keff={r.core.p.keff:.6f}")


if __name__ == "__main__":
    main()
