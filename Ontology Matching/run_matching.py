"""Run updated_OM matching: filter classes only, keep properties from name-based."""

import argparse
import os
import sys

import numpy as np

_OM_UPDATED_DIR = os.path.dirname(os.path.abspath(__file__))
_OM_DIR = os.path.dirname(_OM_UPDATED_DIR)
_PROJECT_ROOT = os.path.dirname(_OM_DIR)
for _path in (_PROJECT_ROOT, _OM_DIR, _OM_UPDATED_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.chdir(_PROJECT_ROOT)

from energy_model import BertEnergy
from filter import DEFAULT_CLASS_WEIGHTS, FILTER_MODES, Filter
from hierarchy_based import HierarchyMatching
from matching_results import write_matching_csv
from name_based_Matching import get_name_based_pairs, save_name_based_results
from ontology_reading import OntologyReading
from structure_based import StructureMatching

BENCHMARK_PAIRS = [
    ('Knowledge Search/Sargon.owl', 'Knowledge Search/sbeo.rdf'),
    ('Knowledge Search/saref4bldg.rdf', 'Knowledge Search/Sargon.owl'),
    ('Knowledge Search/saref4bldg.rdf', 'Knowledge Search/sbeo.rdf'),
]

NAME_RESULTS_DIR = os.path.join(_OM_UPDATED_DIR, 'Results', 'name_based')
COMBINED_RESULTS_DIR = os.path.join(_OM_UPDATED_DIR, 'Results', 'combined')


def final_results_dir(filter_mode):
    """
    @param {str} filter_mode - ``boost`` or ``weighted``
    @returns {str} Directory for evaluation CSVs of this filter mode
    """
    return os.path.join(_OM_UPDATED_DIR, 'Results', 'final', filter_mode)


def _final_csv_path(onto1, onto2, threshold, filter_mode):
    """
    @param {OntologyReading} onto1 - Source ontology
    @param {OntologyReading} onto2 - Target ontology
    @param {float} threshold - Similarity threshold
    @param {str} filter_mode - Filter mode name
    @returns {str} Final merged CSV path used for evaluation
    """
    out_dir = final_results_dir(filter_mode)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(
        out_dir,
        f'Matching_Results_{onto1.name}_and_{onto2.name}_{threshold}.csv',
    )


def run_pipeline(
    threshold=0.7,
    structure_weight=(0.3, 0.6, 0.1),
    run_structure=True,
    filter_mode='boost',
    class_weights=DEFAULT_CLASS_WEIGHTS,
    model=None,
    hierarchy_cache=None,
    name_cache=None,
):
    """
    Run name + hierarchy matching, filter classes only, keep properties unfiltered.

    @param {float} threshold - Similarity threshold (also fuse_threshold for weighted)
    @param {tuple} structure_weight - Structure matcher component weights
    @param {bool} run_structure - Whether to run optional structure-based matching
    @param {str} filter_mode - ``boost`` or ``weighted``
    @param {tuple} class_weights - (name_weight, hierarchy_weight)
    @param {BertEnergy|None} model - Shared embedding model
    @param {dict|None} hierarchy_cache - Reuse hierarchy pairs per (pair, threshold)
    @param {dict|None} name_cache - Reuse name sections per (pair, threshold)
    @returns {list[str]} Paths to final evaluation CSV files
    """
    if filter_mode not in FILTER_MODES:
        raise ValueError(f'Unsupported filter_mode={filter_mode}')

    model = model or BertEnergy()
    hierarchy_matcher = HierarchyMatching(model)
    structure_matcher = StructureMatching(model) if run_structure else None
    result_filter = Filter()
    hierarchy_cache = hierarchy_cache if hierarchy_cache is not None else {}
    name_cache = name_cache if name_cache is not None else {}

    os.makedirs(COMBINED_RESULTS_DIR, exist_ok=True)
    final_paths = []

    for path1, path2 in BENCHMARK_PAIRS:
        onto1 = OntologyReading(path1)
        onto2 = OntologyReading(path2)
        pair_name = f'{onto1.name}_{onto2.name}'
        pair_folder = os.path.join(COMBINED_RESULTS_DIR, pair_name, filter_mode)
        os.makedirs(pair_folder, exist_ok=True)
        cache_key = (pair_name, threshold)

        print(
            f'\n=== Processing {pair_name} '
            f'(threshold={threshold}, filter={filter_mode}, weights={class_weights}) ==='
        )

        if cache_key in name_cache:
            name_sections = name_cache[cache_key]
        else:
            name_sections = get_name_based_pairs(onto1, onto2, threshold, model=model)
            save_name_based_results(onto1, onto2, threshold, name_sections)
            name_cache[cache_key] = name_sections

        if cache_key in hierarchy_cache:
            hierarchy_pairs = hierarchy_cache[cache_key]
            hierarchy_path = os.path.join(pair_folder, 'hierarchy')
        else:
            hierarchy_pairs = hierarchy_matcher.get_hierarchy_matching_classes_pairs(
                onto1, onto2, threshold
            )
            hierarchy_path = hierarchy_matcher.save_results(
                hierarchy_pairs,
                onto1.name,
                onto2.name,
                os.path.join(COMBINED_RESULTS_DIR, pair_name, 'hierarchy'),
            )
            hierarchy_cache[cache_key] = hierarchy_pairs

        fuse_threshold = threshold if filter_mode == 'weighted' else None
        filtered_classes = result_filter.filter_classes(
            name_sections['classes'],
            hierarchy_pairs,
            mode=filter_mode,
            weights=class_weights,
            fuse_threshold=fuse_threshold,
        )
        class_filter_path = result_filter.save_filter_results(
            filtered_classes,
            onto1.name,
            onto2.name,
            os.path.join(pair_folder, 'class_filter'),
        )

        final_csv = _final_csv_path(onto1, onto2, threshold, filter_mode)
        write_matching_csv(
            final_csv,
            onto1.name,
            onto2.name,
            filtered_classes,
            name_sections['object_properties'],
            name_sections['data_properties'],
        )
        final_paths.append(final_csv)

        structure_object_path = None
        structure_data_path = None
        if structure_matcher is not None:
            object_struct1 = onto1.get_property_structure_triple('object_property')
            object_struct2 = onto2.get_property_structure_triple('object_property')
            data_struct1 = onto1.get_property_structure_triple('data_property')
            data_struct2 = onto2.get_property_structure_triple('data_property')

            object_matches = structure_matcher.structure_matching_pairs(
                'object_property', object_struct1, object_struct2, threshold, weight=structure_weight
            )
            data_matches = structure_matcher.structure_matching_pairs(
                'data_property', data_struct1, data_struct2, threshold, weight=structure_weight
            )

            structure_dir = os.path.join(COMBINED_RESULTS_DIR, pair_name, 'structure')
            structure_object_path = structure_matcher.save_structure_matching_result(
                object_matches,
                'object_property',
                onto1.name,
                onto2.name,
                structure_dir,
            )
            structure_data_path = structure_matcher.save_structure_matching_result(
                data_matches,
                'data_property',
                onto1.name,
                onto2.name,
                structure_dir,
            )

        print(f'  name-based raw: {os.path.join(NAME_RESULTS_DIR, os.path.basename(final_csv))}')
        print(f'  hierarchy: {hierarchy_path}')
        print(f'  class filter ({filter_mode}): {class_filter_path}')
        print(f'  final eval csv: {final_csv}')
        if structure_object_path:
            print(f'  structure object: {structure_object_path}')
        if structure_data_path:
            print(f'  structure data: {structure_data_path}')

    return final_paths


def run_all_thresholds(
    thresholds=None,
    structure_weight=(0.3, 0.6, 0.1),
    run_structure=True,
    filter_modes=None,
    class_weights=DEFAULT_CLASS_WEIGHTS,
):
    """
    Run matching for all thresholds and selected filter modes.

    Name/hierarchy matching is computed once per threshold and reused across modes.

    @param {iterable|None} thresholds - Default 0.60..0.90 step 0.05
    @param {iterable|None} filter_modes - Default both ``boost`` and ``weighted``
    @param {tuple} class_weights - (name_weight, hierarchy_weight)
    @returns {list[str]} All final CSV paths
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.60, 0.95, 0.05)]
    if filter_modes is None:
        filter_modes = list(FILTER_MODES)

    model = BertEnergy()
    name_cache = {}
    hierarchy_cache = {}
    all_paths = []

    for threshold in thresholds:
        for filter_mode in filter_modes:
            all_paths.extend(
                run_pipeline(
                    threshold=threshold,
                    structure_weight=structure_weight,
                    # structure is independent of class filter; run once on first mode
                    run_structure=run_structure and filter_mode == filter_modes[0],
                    filter_mode=filter_mode,
                    class_weights=class_weights,
                    model=model,
                    hierarchy_cache=hierarchy_cache,
                    name_cache=name_cache,
                )
            )
    return all_paths


def _parse_args():
    """@returns {argparse.Namespace} CLI options"""
    parser = argparse.ArgumentParser(description='Run updated_OM matching with class filter modes')
    parser.add_argument(
        '--filter-mode',
        choices=list(FILTER_MODES) + ['both'],
        default='both',
        help='Class filter strategy (default: both)',
    )
    parser.add_argument(
        '--name-weight',
        type=float,
        default=DEFAULT_CLASS_WEIGHTS[0],
        help='Weight for name-based class scores',
    )
    parser.add_argument(
        '--hierarchy-weight',
        type=float,
        default=DEFAULT_CLASS_WEIGHTS[1],
        help='Weight for hierarchy-based class scores',
    )
    parser.add_argument(
        '--no-structure',
        action='store_true',
        help='Skip structure-based property matching',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    modes = list(FILTER_MODES) if args.filter_mode == 'both' else [args.filter_mode]
    run_all_thresholds(
        filter_modes=modes,
        class_weights=(args.name_weight, args.hierarchy_weight),
        run_structure=not args.no_structure,
    )
