"""Hierarchy-based ontology matching using EnergyBERT."""

import os
import sys

import pandas as pd

_OM_UPDATED_DIR = os.path.dirname(os.path.abspath(__file__))
_OM_DIR = os.path.dirname(_OM_UPDATED_DIR)
_PROJECT_ROOT = os.path.dirname(_OM_DIR)
for _path in (_PROJECT_ROOT, _OM_DIR, _OM_UPDATED_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from energy_model import BertEnergy
from ontology_reading import OntologyReading
from utilizing import get_file_path, get_cosine_similarity


class HierarchyMatching(object):
    """Match ontology classes via subClassOf hierarchy sentence similarity."""

    def __init__(self, transformer_model=None):
        """
        @param {BertEnergy|None} transformer_model - Default EnergyBERT wrapper
        """
        self.model = transformer_model or BertEnergy()

    def triple_to_sentence(self, triples):
        """
        Convert hierarchy triples into natural-language sentences.

        @param {iterable[tuple]} triples - (child, relation, parent) tuples
        @returns {list[str]} Sentences for embedding
        """
        return [f"Class {child} is a subclass of {parent}" for (child, rel, parent) in triples]

    def hierarchy_matching_pairs(self, onto1, onto2, threshold):
        """
        @param {OntologyReading} onto1 - Source ontology
        @param {OntologyReading} onto2 - Target ontology
        @param {float} threshold - Minimum cosine similarity
        @returns {list[tuple]} (sentence1, sentence2, similarity)
        """
        hierarchy1 = onto1.get_onto_hierarchy_triples(onto1.name)
        hierarchy2 = onto2.get_onto_hierarchy_triples(onto2.name)

        hierarchy1_list = self.triple_to_sentence(hierarchy1)
        hierarchy2_list = self.triple_to_sentence(hierarchy2)

        if not hierarchy1_list or not hierarchy2_list:
            return []

        cos_sim = get_cosine_similarity(self.model, hierarchy1_list, hierarchy2_list)
        matches = []
        for idx1, row in enumerate(cos_sim):
            for idx2, sim in enumerate(row):
                sim_value = float(sim)
                if sim_value > threshold:
                    matches.append((hierarchy1_list[idx1], hierarchy2_list[idx2], sim_value))
        return matches

    def get_hierarchy_matching_classes_pairs(self, onto1, onto2, threshold=0.7):
        """
        Derive class-level alignments from matched hierarchy sentences.

        @param {OntologyReading} onto1 - Source ontology
        @param {OntologyReading} onto2 - Target ontology
        @param {float} threshold - Minimum sentence similarity
        @returns {list[tuple]} (class1, class2, similarity)
        """
        matches = self.hierarchy_matching_pairs(onto1, onto2, threshold)
        class_pairs = set()

        for item in matches:
            if 'is a subclass of' not in item[0] or 'is a subclass of' not in item[1]:
                continue

            subclass_1, parent_class_1 = item[0].split(' is a subclass of ')
            subclass_1 = subclass_1.replace('Class ', '', 1).strip()
            parent_class_1 = parent_class_1.strip()

            subclass_2, parent_class_2 = item[1].split(' is a subclass of ')
            subclass_2 = subclass_2.replace('Class ', '', 1).strip()
            parent_class_2 = parent_class_2.strip()
            similarity = item[2]

            class_pairs.add((parent_class_1, parent_class_2, similarity))
            class_pairs.add((subclass_1, subclass_2, similarity))

        class_pairs_dict = {}
        for item in class_pairs:
            key = (item[0], item[1])
            if key not in class_pairs_dict or item[2] > class_pairs_dict[key][2]:
                class_pairs_dict[key] = item

        result = list(class_pairs_dict.values())
        print('Hierarchy based matching is finished!')
        return result

    def save_results(self, class_pairs, onto1_name, onto2_name, result_folder):
        """
        @param {list[tuple]} class_pairs - Alignment rows with similarity
        @param {str} onto1_name - Source ontology name
        @param {str} onto2_name - Target ontology name
        @param {str} result_folder - Output directory
        @returns {str} Saved file path
        """
        match_df = pd.DataFrame(class_pairs, columns=[onto1_name, onto2_name, 'Cosine Similarity'])
        suffix = '_hierarchy_based_results.xlsx'
        output_path = get_file_path(result_folder, onto1_name + '_' + onto2_name, suffix)
        match_df.to_excel(output_path, index=False)
        return output_path


if __name__ == '__main__':
    os.chdir(_PROJECT_ROOT)

    onto1 = OntologyReading('Knowledge Search/saref4bldg.rdf')
    onto2 = OntologyReading('Knowledge Search/Sargon.owl')
    result_folder = os.path.join(_OM_UPDATED_DIR, 'Results', 'hierarchy')

    matcher = HierarchyMatching(BertEnergy())
    hierarchy_pairs = matcher.get_hierarchy_matching_classes_pairs(onto1, onto2, 0.7)
    print(f'Found {len(hierarchy_pairs)} class pairs')
    if hierarchy_pairs:
        print(hierarchy_pairs[:5])
    matcher.save_results(hierarchy_pairs, onto1.name, onto2.name, result_folder)
