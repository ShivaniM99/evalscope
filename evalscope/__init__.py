# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope import agent  # noqa: F401  # registered agent strategies
from evalscope.benchmarks import *  # registered benchmarks
from evalscope.config import TaskConfig
from evalscope.evaluator import *  # registered evaluators
from evalscope.filters import extraction, selection  # registered filters
from evalscope.metrics import metric  # registered metrics
from evalscope.models import model_apis  # need for register model apis
from evalscope.run import run_task
from .version import __release_datetime__, __version__


try:
    from evalscope.benchmarks import *
except Exception as e:
    print("Skipping benchmark auto-import during local debugging:", e)