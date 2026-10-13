"""Structure-based property matching using EnergyBERT."""

import os
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_OM_UPDATED_DIR = os.path.dirname(os.path.abspath(__file__))
_OM_DIR = os.path.dirname(_OM_UPDATED_DIR)
_PROJECT_ROOT = os.path.dirname(_OM_DIR)
for _path in (_PROJECT_ROOT, _OM_DIR, _OM_UPDATED_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from energy_model import BertEnergy
from utilizing import get_file_path


class StructureMatching(object):
    """Match properties by (domain, property, range) structure similarity."""

    def __init__(self, transformer_model=None):
        """
        @param {BertEnergy|None} transformer_model - Default EnergyBERT wrapper
        """
        self.model = transformer_model or BertEnergy()

    def encode_structure_separate(self, data_type, structure_list):
        """
        Encode domain, property, and range components separately.

        @param {str} data_type - ``data_property`` or ``object_property``
        @param {list[tuple]} structure_list - (domain, property, range) tuples
        @returns {list} Encoded structures for similarity scoring
        """
        encode_structures = []
        if data_type == 'data_property':
            for domain, property_, range_ in structure_list:
                domain_emb = self.model.embedding([f'Domain: {domain}'])[0]
                property_emb = self.model.embedding([f'Property: {property_}'])[0]
                if range_ == 'int' or range_ is None:
                    range_ = 'float'
                encode_structures.append((domain_emb, property_emb, range_))

        elif data_type == 'object_property':
            for domain, property_, range_ in structure_list:
                domain_emb = self.model.embedding([f'Domain: {domain}'])[0]
                property_emb = self.model.embedding([f'Property: {property_}'])[0]
                range_emb = self.model.embedding([f'Range: {range_}'])[0]
                encode_structures.append((domain_emb, property_emb, range_emb))
        return encode_structures

    def encode_structure_overall(self, data_type, structure_list):
        """
        Encode each structure triple as one sentence.

        @param {str} data_type - ``data_property`` or ``object_property``
        @param {list[tuple]} structure_list - (domain, property, range) tuples
        @returns {list} Whole-structure embeddings
        """
        encode_structures = []
        if data_type == 'data_property':
            for structure in structure_list:
                domain, property_, range_ = structure
                if range_ == 'int':
                    range_ = 'float'
                sentence = f'Domain: {domain}; Property: {property_}; Range: {range_}'
                encode_structures.append(self.model.embedding([sentence])[0])
        elif data_type == 'object_property':
            for domain, property_, range_ in structure_list:
                sentence = f'Domain: {domain}; Property: {property_}; Range: {range_}'
                encode_structures.append(self.model.embedding([sentence])[0])
        return encode_structures

    @staticmethod
    def calculate_weighted_similarity(data_type, emb1, emb2, weight=None):
        """
        @param {str} data_type - Property type
        @param {tuple|object} emb1 - First encoded structure
        @param {tuple|object} emb2 - Second encoded structure
        @param {tuple|None} weight - Optional component weights
        @returns {float} Similarity score
        """
        if weight is None:
            return float(cosine_similarity([emb1], [emb2])[0][0])

        if data_type == 'data_property':
            domain_sim = float(cosine_similarity([emb1[0]], [emb2[0]])[0][0])
            property_sim = float(cosine_similarity([emb1[1]], [emb2[1]])[0][0])
            return weight[0] * domain_sim + weight[1] * property_sim

        domain_sim = float(cosine_similarity([emb1[0]], [emb2[0]])[0][0])
        property_sim = float(cosine_similarity([emb1[1]], [emb2[1]])[0][0])
        range_sim = float(cosine_similarity([emb1[2]], [emb2[2]])[0][0])
        return weight[0] * domain_sim + weight[1] * property_sim + weight[2] * range_sim

    def structure_matching_pairs(self, data_type, structure1_list, structure2_list, threshold, weight=None):
        """
        @param {str} data_type - ``data_property`` or ``object_property``
        @param {list[tuple]} structure1_list - Source property structures
        @param {list[tuple]} structure2_list - Target property structures
        @param {float} threshold - Minimum similarity
        @param {tuple|None} weight - Optional component weights
        @returns {list[tuple]} (property1, property2, similarity)
        """
        if weight is None:
            structure1_emb = self.encode_structure_overall(data_type, structure1_list)
            structure2_emb = self.encode_structure_overall(data_type, structure2_list)
        else:
            structure1_emb = self.encode_structure_separate(data_type, structure1_list)
            structure2_emb = self.encode_structure_separate(data_type, structure2_list)

        matches = []
        if data_type == 'data_property':
            for idx1, emb1 in enumerate(structure1_emb):
                for idx2, emb2 in enumerate(structure2_emb):
                    range1 = structure1_list[idx1][2]
                    range2 = structure2_list[idx2][2]
                    if range1 == 'int':
                        range1 = 'float'
                    if range2 == 'int':
                        range2 = 'float'
                    if range1 != range2:
                        continue
                    sim = self.calculate_weighted_similarity(data_type, emb1, emb2, weight)
                    if sim > threshold:
                        matches.append((structure1_list[idx1], structure2_list[idx2], sim))
        elif data_type == 'object_property':
            for idx1, emb1 in enumerate(structure1_emb):
                for idx2, emb2 in enumerate(structure2_emb):
                    sim = self.calculate_weighted_similarity(data_type, emb1, emb2, weight)
                    if sim > threshold:
                        matches.append((structure1_list[idx1], structure2_list[idx2], sim))

        return [(item[0][1], item[1][1], item[2]) for item in matches]

    def save_structure_matching_result(self, data, data_type, name1, name2, result_folder):
        """
        @param {list[tuple]} data - Matched property pairs
        @param {str} data_type - Property type label
        @param {str} name1 - Source ontology name
        @param {str} name2 - Target ontology name
        @param {str} result_folder - Output directory
        @returns {str} Saved file path
        """
        df = pd.DataFrame(data, columns=[name1 + '_' + data_type, name2 + '_' + data_type, 'Similarity'])
        suffix = '_structure_based_result.xlsx'
        output_path = get_file_path(result_folder, name1 + '_' + name2 + '_' + data_type, suffix)
        df.to_excel(output_path, index=False)
        return output_path


if __name__ == '__main__':
    os.chdir(_PROJECT_ROOT)

    from ontology_reading import OntologyReading

    onto1 = OntologyReading('Knowledge Search/saref4bldg.rdf')
    onto2 = OntologyReading('Knowledge Search/Sargon.owl')
    result_folder = os.path.join(_OM_UPDATED_DIR, 'Results', 'structure')

    object_list1 = onto1.get_property_structure_triple('object_property')
    object_list2 = onto2.get_property_structure_triple('object_property')
    data_list1 = onto1.get_property_structure_triple('data_property')
    data_list2 = onto2.get_property_structure_triple('data_property')

    matcher = StructureMatching(BertEnergy())
    weight = (0.3, 0.6, 0.1)

    object_matches = matcher.structure_matching_pairs(
        'object_property', object_list1, object_list2, 0.5, weight=weight
    )
    data_matches = matcher.structure_matching_pairs(
        'data_property', data_list1, data_list2, 0.5, weight=weight
    )

    print(f'Object property matches: {len(object_matches)}')
    print(f'Data property matches: {len(data_matches)}')
    matcher.save_structure_matching_result(object_matches, 'object_property', onto1.name, onto2.name, result_folder)
    matcher.save_structure_matching_result(data_matches, 'data_property', onto1.name, onto2.name, result_folder)
