# registry.py
"""
LOCAL DEVELOPMENT REGISTRY — NOT USED BY EVALSCOPE AT RUNTIME
==============================================================
This file is a lightweight local registry for development and notebook testing.
It is NOT what wires adapters into evalscope's CLI.

How evalscope's real registration works:
  The @register_benchmark(BenchmarkMeta(...)) decorator in each adapter file
  calls evalscope.api.registry.register_benchmark() at import time.
  evalscope discovers adapters by importing the benchmark package's __init__.py,
  which must import the adapter modules so the decorators fire.

This file is provided as a convenience reference showing which benchmarks
exist in this package and what their extra_params look like.
To actually use the pruned benchmark via CLI, see README usage instructions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BenchmarkMeta:
    name: str
    adapter_class: str
    description: str = ''
    extra_params: Dict[str, Any] = field(default_factory=dict)


_BENCHMARKS: Dict[str, BenchmarkMeta] = {}


def register_benchmark(
    name: str,
    adapter_class: str,
    description: str = '',
    extra_params: Optional[Dict[str, Any]] = None,
) -> None:
    _BENCHMARKS[name] = BenchmarkMeta(
        name=name,
        adapter_class=adapter_class,
        description=description,
        extra_params=extra_params or {},
    )


def get_benchmark(name: str) -> BenchmarkMeta:
    if name not in _BENCHMARKS:
        raise KeyError(
            f"Benchmark '{name}' not registered. Available: {sorted(_BENCHMARKS.keys())}"
        )
    return _BENCHMARKS[name]


def list_benchmarks() -> list:
    return sorted(_BENCHMARKS.keys())


# ── Registered benchmarks (for local reference only) ─────────────────────────

register_benchmark(
    name='live_code_bench',
    adapter_class='live_code_bench_adapter.LiveCodeBenchAdapter',
    description='Full LiveCodeBench — all problems, no pruning.',
    extra_params={
        'start_date': None,
        'end_date': None,
        'debug': False,
    },
)

register_benchmark(
    name='live_code_bench_pruned',
    adapter_class='livecodebench_pruned_adapter.LiveCodeBenchPrunedAdapter',
    description='Pruned LiveCodeBench — disagreement-first + coverage-fill selection.',
    extra_params={
        # Point ONE of these at your reference data:
        'joined_csv_path': None,        # e.g. '/path/to/LiveCodeBench_joined_all_models.csv'
        'prediction_dir': None,         # e.g. '/path/to/Evals/Part 1/predictions'
        'review_dir': None,             # e.g. '/path/to/Evals/Part 1/reviews'
        'min_per_cell': 1,
        'max_unanimous_fill_per_cell': 1,
        'start_date': None,
        'end_date': None,
        'debug': False,
    },
)
