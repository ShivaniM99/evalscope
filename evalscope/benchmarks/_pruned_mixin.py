# flake8: noqa: E501
"""
_pruned_mixin.py
================
PrunedAdapterMixin — shared plumbing that wires the evalscope ``sample_filter``
hook to a benchmark-specific index-selection algorithm.

How to use
----------
1. Inherit from both this mixin AND the base benchmark adapter.
   Put PrunedAdapterMixin FIRST so its sample_filter takes priority in the MRO.

2. Call ``self.__init_pruned_mixin__()`` inside your ``__init__`` after super().

3. Implement ``_select_indices() -> List`` — return the keys of samples to keep.
   Keys must match what ``_get_sample_key(sample)`` returns (default: int index).
   Override ``_get_sample_key`` for string IDs or other key types.

Example::

    class MyBenchmarkPrunedAdapter(PrunedAdapterMixin, MyBenchmarkAdapter):
        def __init__(self, benchmark_meta, task_config=None):
            super().__init__(benchmark_meta, task_config)
            self.__init_pruned_mixin__()

        def _select_indices(self):
            return my_selection_algorithm(...)

        # optional: if the base adapter also has a sample_filter (e.g. date range)
        def sample_filter(self, sample):
            return (MyBenchmarkAdapter.sample_filter(self, sample) and
                    PrunedAdapterMixin.sample_filter(self, sample))
"""
from typing import Any, List, Optional, Set

from evalscope.api.dataset import Sample
from evalscope.utils.logger import get_logger

logger = get_logger()


class PrunedAdapterMixin:
    """
    Mixin that adds index-based pruning to any evalscope adapter.

    - ``_select_indices()`` is called **once** (lazily on the first
      ``sample_filter`` call) and its result is cached as a set.
    - ``_get_sample_key(sample)`` maps a Sample to the key checked against
      that set.  Default: ``int(sample.metadata['index'])``.
      Override to use string IDs or composite keys.
    - ``_get_extra_param(key, default)`` is a convenience helper for reading
      from ``self.extra_params`` regardless of whether the entry is stored as
      a raw value or as a spec-dict ``{type, description, value}``.
    """

    # ── initialisation ────────────────────────────────────────────────────────

    def __init_pruned_mixin__(self) -> None:
        """Initialise mixin state.  Must be called from __init__ after super()."""
        self._pruned_keys: Optional[List] = None
        self._pruned_key_set: Optional[Set] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_extra_param(self, key: str, default: Any = None) -> Any:
        """
        Read a value from ``self.extra_params``, handling both:
          - raw value: ``extra_params['k'] = 6``
          - spec-dict: ``extra_params['k'] = {'type': 'int', 'value': 6, ...}``
        """
        params = getattr(self, 'extra_params', {}) or {}
        val = params.get(key, default)
        if isinstance(val, dict):
            return val.get('value', default)
        return val if val is not None else default

    def _get_sample_key(self, sample: Sample) -> Optional[Any]:
        """
        Extract the membership-check key from a sample.

        Default implementation returns ``int(sample.metadata['index'])``.
        Override in subclasses that use string IDs or other key types.
        """
        if not sample.metadata:
            return None
        idx = sample.metadata.get('index')
        if idx is None:
            return None
        try:
            return int(idx)
        except (TypeError, ValueError):
            return idx

    # ── selection (subclass API) ──────────────────────────────────────────────

    def _select_indices(self) -> List:
        """
        Return the list of sample keys to keep.
        Keys must match the type returned by ``_get_sample_key``.
        Implemented by every pruned-adapter subclass.
        """
        raise NotImplementedError(
            f'{type(self).__name__} must implement _select_indices()'
        )

    def _ensure_keys_loaded(self) -> Set:
        """Compute and cache the pruned key set (lazy — one-time cost)."""
        if self._pruned_key_set is not None:
            return self._pruned_key_set

        logger.info(f'[{type(self).__name__}] computing pruned index set …')
        self._pruned_keys = self._select_indices()
        self._pruned_key_set = set(self._pruned_keys)
        logger.info(
            f'[{type(self).__name__}] selected {len(self._pruned_keys)} samples.'
        )
        return self._pruned_key_set

    # ── evalscope hook ────────────────────────────────────────────────────────

    def sample_filter(self, sample: Sample) -> bool:
        """
        evalscope calls this after record_to_sample() for every sample.
        Returns True only if the sample's key is in the pruned set.

        If the base adapter also has a sample_filter (e.g. a date-range filter),
        override this in the subclass and call both explicitly::

            def sample_filter(self, sample):
                return (BaseAdapter.sample_filter(self, sample) and
                        PrunedAdapterMixin.sample_filter(self, sample))
        """
        key = self._get_sample_key(sample)
        if key is None:
            logger.warning(
                f'[{type(self).__name__}] sample missing key in metadata — excluded'
            )
            return False
        return key in self._ensure_keys_loaded()
