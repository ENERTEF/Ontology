"""Name-based ontology matching using EnergyBERT."""

import os
import sys

import numpy as np

_OM_UPDATED_DIR = os.path.dirname(os.path.abspath(__file__))
_OM_DIR = os.path.dirname(_OM_UPDATED_DIR)
_PROJECT_ROOT = os.path.dirname(_OM_DIR)
for _path in (_PROJECT_ROOT, _OM_DIR, _OM_UPDATED_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from energy_model import BertEnergy
from matching_results import write_matching_csv
from ontology_reading import OntologyReading

RESULTS_DIR = os.path.join(_OM_UPDATED_DIR, 'Results', 'name_based')


def _result_csv_path(onto1, onto2, sim_threshold):
    """
    @param {OntologyReading} onto1 - First ontology
    @param {OntologyReading} onto2 - Second ontology
    @param {float} sim_threshold - Similarity threshold
    @returns {str} Output CSV path
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(
        RESULTS_DIR,
        f'Matching_Results_{onto1.name}_and_{onto2.name}_{sim_threshold}.csv',
    )


def _pairs_from_lists(src_list, tgt_list, sim_list):
    """@returns {list[tuple]} (source, target, similarity) rows"""
    return list(zip(src_list, tgt_list, sim_list))


def bert_matching(model, list1, list2, sim_threshold):
    """
    @param {BertEnergy} model - EnergyBERT wrapper
    @param {list[str]} list1 - Source terms
    @param {list[str]} list2 - Target terms
    @param {float} sim_threshold - Minimum similarity
    @returns {list[tuple]} (source, target, similarity) pairs
    """
    pairs = []
    sim_tensor = model.sim(list1, list2)
    for i in range(sim_tensor.shape[0]):
        for j in range(sim_tensor.shape[1]):
            sim = sim_tensor[i][j]
            if sim > sim_threshold:
                pairs.append((list1[i], list2[j], round(float(sim), 3)))
    return pairs


def get_name_based_pairs(onto1, onto2, sim_threshold, model=None):
    """
    Run name-based matching and return sectioned pair lists.

    @param {OntologyReading} onto1 - Source ontology
    @param {OntologyReading} onto2 - Target ontology
    @param {float} sim_threshold - Similarity threshold
    @param {BertEnergy|None} model - Shared model instance
    @returns {dict} classes, object_properties, data_properties pair lists
    """
    model = model or BertEnergy()
    print(f'[Name-based] Matching {onto1.name} <-> {onto2.name}, threshold={sim_threshold}')

    classes = bert_matching(model, onto1.get_classes(), onto2.get_classes(), sim_threshold)
    object_properties = bert_matching(
        model, onto1.get_object_properties(), onto2.get_object_properties(), sim_threshold
    )
    data_properties = bert_matching(
        model, onto1.get_data_properties(), onto2.get_data_properties(), sim_threshold
    )

    print(f'  classes: {len(classes)}, object properties: {len(object_properties)}, '
          f'data properties: {len(data_properties)}')
    return {
        'classes': classes,
        'object_properties': object_properties,
        'data_properties': data_properties,
    }


def save_name_based_results(onto1, onto2, sim_threshold, sections):
    """
    @param {OntologyReading} onto1 - Source ontology
    @param {OntologyReading} onto2 - Target ontology
    @param {float} sim_threshold - Similarity threshold
    @param {dict} sections - Output of get_name_based_pairs
    @returns {str} Saved CSV path
    """
    csv_path = _result_csv_path(onto1, onto2, sim_threshold)
    write_matching_csv(
        csv_path,
        onto1.name,
        onto2.name,
        sections['classes'],
        sections['object_properties'],
        sections['data_properties'],
    )
    return csv_path


def onto_matching(onto1, onto2, sim_threshold, model=None):
    """
    Run full name-based matching for one ontology pair and threshold.

    @param {OntologyReading} onto1 - Source ontology
    @param {OntologyReading} onto2 - Target ontology
    @param {float} sim_threshold - Similarity threshold
    @param {BertEnergy|None} model - Shared model instance
    @returns {dict} Sectioned name-based pair lists
    """
    sections = get_name_based_pairs(onto1, onto2, sim_threshold, model=model)
    save_name_based_results(onto1, onto2, sim_threshold, sections)
    return sections


def run_all_pairs(thresholds=None):
    """
    Run name-based matching for benchmark ontology pairs.

    @param {iterable|None} thresholds - Default 0.60..0.90 step 0.05
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.60, 0.95, 0.05)]

    os.chdir(_PROJECT_ROOT)
    model = BertEnergy()

    pairs = [
        (OntologyReading('Knowledge Search/Sargon.owl'), OntologyReading('Knowledge Search/sbeo.rdf')),
        (OntologyReading('Knowledge Search/saref4bldg.rdf'), OntologyReading('Knowledge Search/Sargon.owl')),
        (OntologyReading('Knowledge Search/saref4bldg.rdf'), OntologyReading('Knowledge Search/sbeo.rdf')),
    ]

    for onto1, onto2 in pairs:
        for sim_threshold in thresholds:
            onto_matching(onto1, onto2, sim_threshold, model=model)


if __name__ == '__main__':
    run_all_pairs()
