"""Load and write sectioned ontology matching CSV files."""

import os

import pandas as pd

SECTION_LABELS = {
    'Classes': 'classes',
    'Object Properties': 'object_properties',
    'Data Properties': 'data_properties',
}


def load_matching_sections(csv_path):
    """
    Parse a matching CSV into class and property pair lists.

    @param {str} csv_path - Path to Matching_Results CSV
    @returns {dict} keys: classes, object_properties, data_properties
    """
    sections = {key: [] for key in SECTION_LABELS.values()}
    current = 'classes'

    if not os.path.isfile(csv_path):
        return sections

    with open(csv_path, encoding='utf-8') as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(',')]
            if len(parts) < 2 or parts[0] == '':
                continue
            if parts[0] in SECTION_LABELS:
                current = SECTION_LABELS[parts[0]]
                continue
            if parts[1] == '':
                continue
            sim = float(parts[2]) if len(parts) > 2 and parts[2] != '' else 0.0
            sections[current].append((parts[0], parts[1], sim))

    return sections


def write_matching_csv(csv_path, onto1_name, onto2_name, class_pairs, object_pairs, data_pairs):
    """
    Write class and property sections in om_evaluate-compatible CSV format.

    @param {str} csv_path - Output CSV path
    @param {str} onto1_name - Source ontology column name
    @param {str} onto2_name - Target ontology column name
    @param {list[tuple]} class_pairs - (source, target, similarity)
    @param {list[tuple]} object_pairs - Object property pairs
    @param {list[tuple]} data_pairs - Data property pairs
    """
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    df = pd.DataFrame({'Classes': ['']})
    df.to_csv(csv_path, mode='w', header=False, index=False)
    _append_section(csv_path, onto1_name, onto2_name, class_pairs)

    df = pd.DataFrame({'Object Properties': ['']})
    df.to_csv(csv_path, mode='a', header=False, index=False)
    _append_section(csv_path, onto1_name, onto2_name, object_pairs)

    df = pd.DataFrame({'Data Properties': ['']})
    df.to_csv(csv_path, mode='a', header=False, index=False)
    _append_section(csv_path, onto1_name, onto2_name, data_pairs)


def _append_section(csv_path, onto1_name, onto2_name, pairs):
    """@param {list[tuple]} pairs - (source, target, similarity) rows"""
    if not pairs:
        return
    df = pd.DataFrame({
        onto1_name: [p[0] for p in pairs],
        onto2_name: [p[1] for p in pairs],
        'BERT_Sim': [round(float(p[2]), 3) for p in pairs],
    })
    df.to_csv(csv_path, mode='a', header=False, index=False)
