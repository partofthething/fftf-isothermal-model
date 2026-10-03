"""
Profile the get/set number density benchmark with cProfile and line_profiler.

Usage::

    python profiling/profile_ndens.py [outDir]

Writes ``get.prof`` and ``set.prof`` (viewable with snakeviz or pstats) to ``outDir``
and prints the top functions plus line-level timings of the hot ARMI methods.
"""

import cProfile
import pathlib
import pstats
import sys

from line_profiler import LineProfiler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_ndens as bn  # noqa: E402

from armi.reactor import composites  # noqa: E402
from armi.reactor.components import component  # noqa: E402

HOT_FUNCTIONS = [
    component.Component.updateNumberDensities,
    component.Component.getNuclideNumberDensities,
    component.Component.getNuclides,
    composites.Composite.updateNumberDensities,
    composites.Composite.getNuclideNumberDensities,
    composites.Composite._getNdensHelper,
]


def main():
    outDir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    core = bn.loadCore()
    blocks = core.getBlocks()
    ndens = bn.readAll(blocks)
    phases = [("get", lambda: bn.readAll(blocks)), ("set", lambda: bn.setAll(blocks, ndens))]

    for name, fn in phases:
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(3):
            fn()
        pr.disable()
        pr.dump_stats(outDir / f"{name}.prof")
        print(f"================= {name}")
        pstats.Stats(pr).sort_stats("tottime").print_stats(15)

    lp = LineProfiler(*HOT_FUNCTIONS)
    for _name, fn in phases:
        lp.runcall(fn)
    lp.print_stats(output_unit=1e-3, stripzeros=True)


if __name__ == "__main__":
    main()
