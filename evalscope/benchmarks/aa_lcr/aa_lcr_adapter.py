# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import re
import urllib.request
import zipfile
from pathlib import Path

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, Tags
from evalscope.utils.logger import get_logger


# Additional imports
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from evalscope.benchmarks.aa_lcr.pruner import StratifiedDiscriminabilityPruner

logger = get_logger()

# Default judge prompt template
JUDGE_PROMPT = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT. For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {correct_answer}
CANDIDATE ANSWER TO ASSESS: {response}

Reply only with CORRECT or INCORRECT."""

PROMPT_TEMPLATE = """
BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

# New constants for auto-download
DOWNLOAD_URL: str = (
    'https://modelscope.cn/datasets/evalscope/AA-LCR/resolve/master/extracted_text/AA-LCR_extracted-text.zip'
)
DEFAULT_CACHE_SUBDIR: str = 'aa_lcr'
DEFAULT_ZIP_NAME: str = 'AA-LCR_extracted-text.zip'
DEFAULT_EXTRACTED_DIR_NAME: str = 'lcr'


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr',
        pretty_name='AA-LCR',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

AA-LCR (Artificial Analysis Long Context Retrieval) is a benchmark for evaluating long-context retrieval and reasoning capabilities of language models. It requires models to find and synthesize information across multiple documents.

## Task Description

- **Task Type**: Long-Context Question Answering
- **Input**: Multiple documents + question requiring cross-document reasoning
- **Output**: Answer synthesized from document information
- **Context**: Very long context (multiple documents concatenated)

## Key Features

- Tests long-context retrieval abilities
- Multiple document understanding
- Cross-document reasoning required
- LLM-based judging for answer correctness
- Auto-download of document corpus

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy** (via LLM judge)
- Evaluates on **test** split
- Documents auto-downloaded if `text_dir` not specified
- Judge prompt compares candidate answer against reference
""",  # noqa: E501
        dataset_id='evalscope/AA-LCR',        
        metric_list=['acc'],                   
        few_shot_num=0,                        
        train_split=None,                      
        eval_split='test',                     
        prompt_template=PROMPT_TEMPLATE,       
        extra_params={           #  ONE block only, with all 4 keys
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files; if null will auto-download & extract.',
                'value': None
            },
            'pruning_strategy': {
                'type': 'str | null',
                'description': "Set to 'stratified_discriminability' to enable pruning.",
                'value': None
            },
            'scores_path': {
                'type': 'str | null',
                'description': 'Path to aa_lcr_flat CSV or JSONL with columns: index, model, acc, reasoning_len, input_tokens.',
                'value': None
            },
            'random_state': {
                'type': 'int',
                'description': 'Random seed for pruner reproducibility.',
                'value': 42
            },
        }
    )
)
class AALCRAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._use_llm_judge = True

        # Get extra parameters
        self.text_dir = self.extra_params.get('text_dir')

        # ── Pruning ────────────────────────────────────
        self._pruned_indices: Optional[List[int]] = None
        self._pruned_index_set: Optional[set] = None
        pruning_strategy = self.extra_params.get('pruning_strategy')
        if pruning_strategy == 'stratified_discriminability':
            scores_path = self.extra_params.get('scores_path')
            if not scores_path:
                raise ValueError(
                    "pruning_strategy='stratified_discriminability' requires "
                    "'scores_path' in extra_params pointing to the aa_lcr_flat "
                    "CSV or JSONL file."
                )
            # Accept both CSV and JSONL
            p = Path(scores_path)
            if p.suffix.lower() == '.csv':
                scores_df = pd.read_csv(scores_path)
            else:
                scores_df = pd.read_json(scores_path, lines=True)
            pruner = StratifiedDiscriminabilityPruner(
                random_state=int(self.extra_params.get('random_state', 42))
            )
            self._pruned_indices = pruner.fit_transform(scores_df)
            self._pruned_index_set = set(self._pruned_indices)
            logger.info(
                f'[AA-LCR Pruner] Selected {len(self._pruned_indices)}/100 '
                f"questions via '{pruning_strategy}'"
            )
        # ── End pruning ────────────────────────────────

    def load(self):
        # Auto download and extract when text_dir is not provided
        if not self.text_dir:
            self.text_dir = self._ensure_text_dir_downloaded()
        elif not Path(self.text_dir).exists():
            raise ValueError(
                'AA-LCR text_dir does not exist: '
                f'{self.text_dir}. Please provide a valid directory or omit text_dir to auto-download.'
            )

        self.text_dir = Path(self.text_dir)
        return super().load()

    def sample_filter(self, sample: Sample) -> bool:
        """
        Return True only if this sample's index is in the pruned set.
        When no pruning strategy is configured, all samples pass.

        evalscope calls this for every sample after record_to_sample().
        This is the correct evalscope hook for subsetting — avoids touching
        the DatasetDict returned by load().
        """
        if self._pruned_index_set is None:
            return True
        idx = sample.metadata.get('index')
        if idx is None:
            logger.warning('[AA-LCR Pruner] sample missing "index" in metadata — excluded')
            return False
        return int(idx) in self._pruned_index_set

    def _ensure_text_dir_downloaded(self) -> Path:
        """Ensure AA-LCR extracted texts are available locally; download and extract if missing."""
        cache_root = Path(DEFAULT_EVALSCOPE_CACHE_DIR) / DEFAULT_CACHE_SUBDIR
        extracted_dir = cache_root / DEFAULT_EXTRACTED_DIR_NAME

        if extracted_dir.exists():
            logger.info(f'AA-LCR documents found: {extracted_dir}')
            return extracted_dir

        cache_root.mkdir(parents=True, exist_ok=True)
        zip_path = cache_root / DEFAULT_ZIP_NAME

        try:
            logger.info(f'Downloading AA-LCR documents from {DOWNLOAD_URL} to {zip_path}...')
            urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)

            logger.info(f'Extracting {zip_path} to {cache_root}...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_root)

            if not extracted_dir.exists():
                raise ValueError(f'Extraction succeeded but target directory not found: {extracted_dir}')

            logger.info(f'AA-LCR documents ready at {extracted_dir}')
            return extracted_dir
        except Exception as e:
            raise ValueError(
                f'Failed to download or extract AA-LCR documents: {e}. '
                'You can also manually download and set extra_params["text_dir"].'
            ) from e
        finally:
            # Best-effort cleanup of the zip file
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except Exception:
                pass

    def _get_context(self, record: Dict[str, Any]) -> str:
        doc_folder = self.text_dir / record['document_category'] / record['document_set_id']

        # Check if the document folder exists
        if not doc_folder.exists() or not doc_folder.is_dir():
            logger.warning(f'Document folder not found: {doc_folder}. Returning empty context.')
            return ''

        doc_blocks = []
        try:
            for file_path in doc_folder.iterdir():
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8').strip()
                        if content:
                            doc_blocks.append(content)
                    except (IOError, UnicodeDecodeError) as e:
                        logger.warning(f'Could not read file {file_path}, skipping: {e}')
        except OSError as e:
            logger.warning(f'Could not access document folder {doc_folder}: {e}')
            return f"ERROR: Could not read documents for {record['document_category']}/{record['document_set_id']}"

        documents_text = '\n\n'.join(
            f'BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}' for i, doc in enumerate(doc_blocks)
        )
        return documents_text

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a record to a Sample with long-context prompt."""
        context = self._get_context(record)
        prompt = self.prompt_template.format(documents_text=context, question=record['question'])

        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target=record['answer'],
            metadata={
                'index': record.get('index'),          # required by sample_filter
                'question': record['question'],
                'data_source_urls': record['data_source_urls'],
                'input_tokens': record.get('input_tokens', 0),
            }
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        judge_prompt = JUDGE_PROMPT.format(
            question=task_state.metadata['question'], correct_answer=reference, response=filtered_prediction
        )

        # Request judge and obtain score
        judge_response = self.llm_judge.judge(prompt=judge_prompt)

        # Parse judge response to get accuracy score
        # Use word boundaries to avoid matching "CORRECT" within "INCORRECT"
        is_correct = bool(re.search(r'\bCORRECT\b', judge_response, re.IGNORECASE))
        score.value = {
            'acc': 1.0 if is_correct else 0.0,
        }
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
        }
        score.main_score_name = 'acc'
        return score
