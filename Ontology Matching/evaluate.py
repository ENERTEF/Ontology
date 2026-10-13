"""Evaluate updated_OM final matching results against ground-truth answers."""

import argparse
import os
import sys

_UPDATED_OM_DIR = os.path.dirname(os.path.abspath(__file__))
_OM_DIR = os.path.dirname(_UPDATED_OM_DIR)
_PROJECT_ROOT = os.path.dirname(_OM_DIR)
for _path in (_PROJECT_ROOT, _OM_DIR, _UPDATED_OM_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from filter import FILTER_MODES
from om_evaluate import ANSWERS_DIR, F1Score, discover_eval_configs
from run_matching import final_results_dir


def run_evaluation(filter_modes=None):
    """
    Evaluate merged results under updated_OM/Results/final/<mode>/.

    @param {iterable|None} filter_modes - Default both boost and weighted
    """
    if filter_modes is None:
        filter_modes = list(FILTER_MODES)

    eva = F1Score()

    for filter_mode in filter_modes:
        result_dir = final_results_dir(filter_mode)
        eval_output_root = os.path.join(_UPDATED_OM_DIR, 'Evaluation', filter_mode)
        os.makedirs(eval_output_root, exist_ok=True)

        eval_configs = discover_eval_configs(answers_dir=ANSWERS_DIR, result_dir=result_dir)
        if not eval_configs:
            print(f'\n[{filter_mode}] No eval configs found under: {result_dir}')
            print('Run matching first, e.g.:')
            print('  python "Ontology Matching/updated_OM/run_matching.py" --filter-mode both')
            continue

        print(f'\n===== [updated_OM / {filter_mode}] {len(eval_configs)} answer file(s) =====')
        print(f'  predictions: {result_dir}')
        print(f'  eval output: {eval_output_root}')

        for answer_path, result_basename, output_subdir in eval_configs:
            print(f'\n=== [{filter_mode}] Evaluating {output_subdir} ===')
            print(f'  answer: {answer_path}')
            print(f'  predictions: {result_basename}_{{threshold}}.csv')
            summary = eva.evaluate_thresholds(
                answer_path=answer_path,
                result_basename=result_basename,
                output_subdir=output_subdir,
                result_dir=result_dir,
                eval_output_root=eval_output_root,
            )
            if not summary.empty:
                print(summary.to_string(index=False))


def _parse_args():
    """@returns {argparse.Namespace} CLI options"""
    parser = argparse.ArgumentParser(description='Evaluate updated_OM filter-mode results')
    parser.add_argument(
        '--filter-mode',
        choices=list(FILTER_MODES) + ['both'],
        default='both',
        help='Which filter-mode result folders to evaluate (default: both)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    modes = list(FILTER_MODES) if args.filter_mode == 'both' else [args.filter_mode]
    run_evaluation(filter_modes=modes)
