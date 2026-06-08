import numpy as np
import torch
import pandas as pd
import os
import sys
from owlready2 import get_ontology
from rdflib import *
from rdflib.namespace import split_uri

# Local wordembedding module lives in Ontology Matching/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Ontology Matching'))
from wordembedding import bert_Energy_tsdae


def get_oeo_class_labels(path):
    """
    Read OEO owl:Class rdfs:label values (OBO-style IDs are not BERT-friendly).

    @param path - Path to oeo.owl
    @returns List of English labels; falls back to URI local name if label is missing
    """
    g = Graph()
    g.parse(path)
    labels = []
    for cls_uri in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls_uri, URIRef):
            continue
        label = None
        for lit in g.objects(cls_uri, RDFS.label):
            lang = getattr(lit, "language", None)
            if lang in (None, "en"):
                label = str(lit).strip()
                break
        if not label:
            label = split_uri(cls_uri)[-1]
        if label:
            labels.append(label)
    return labels


def get_ontoclass(path):
    """@param path - Ontology file path (.ttl, .owl, .rdf)"""
    if os.path.basename(path).lower() == "oeo.owl":
        return get_oeo_class_labels(path)

    lst = []
    if path.endswith('.ttl'):
        g = Graph()
        g.parse(path)
        for s, p, o in g.triples((None, RDF.type, OWL.Class)):
            if isinstance(s, URIRef):
                class_name = split_uri(s)[-1]
                lst.append(class_name)
    else:
        onto = get_ontology(path).load()
        for i in list(onto.classes()):
            class_name = str(i).split('.')[-1]
            lst.append(class_name)
    return lst


def get_bertsim(path, sim_min=0.6):
    print("Ontology file name:", os.path.basename(path))
    nodes_lst = get_ontoclass(path)
    print('Number of classes:', len(nodes_lst))

    word_chain = [
        'Building', 'BuildingOperation', 'ElectricVehicle', 'Device', 'PhotovoltaicDevice',
        'SmartMeteringObservation', 'PhotovoltaicMeasurement', 'StorageBatteryMeasurement',
        'Battery', 'StorageBatteryDevice', 'EnergyConsumption', 'Energy', 'System', 'Weather',
        'ManufacturingProcess',
    ]

    bert = bert_Energy_tsdae()
    sim_tensor = bert.sim(nodes_lst, word_chain)

    threshold = sim_tensor > sim_min
    sum_bert = sim_tensor[threshold].sum().item()

    print(f'BERT similarity is:', sum_bert)

    return round(sum_bert, 3)


if __name__ == '__main__':
    # Ontology files under Knowledge Search/ (exclude catalog-v001.xml; one file per ontology)
    ONTOLOGY_ENTRIES = [
        ("Knowledge Search/saref4bldg.rdf", "SAREF4BLDG"),
        ("Knowledge Search/saref4ener.ttl", "SAREF4ENER"),
        ("Knowledge Search/saref4grid.ttl", "SAREF4GRID"),
        ("Knowledge Search/SARGON.ttl", "SARGON"),
        ("Knowledge Search/sbeo.rdf", "SBEO"),
        ("Knowledge Search/em-kpi.ttl", "EM-KPI"),
        ("Knowledge Search/fso.ttl", "FSO"),
        ("Knowledge Search/SSN.ttl", "SSN"),
        ("Knowledge Search/BIMERR Building Ontology.ttl", "BIMERR-Building"),
        ("Knowledge Search/BIMERR Weather Ontology.ttl", "BIMERR-Weather"),
        ("Knowledge Search/BIMERR Sensor Data Ontology.ttl", "BIMERR-Sensor"),
        ("Knowledge Search/BED.owl", "BED"),
        ("Knowledge Search/oeo.owl", "OEO"),
    ]

    threshold_range = np.arange(0.60, 0.90, 0.05)
    print('Sim_threshold chosen:', threshold_range)
    for sim_threshold in threshold_range:
        sim_threshold = round(sim_threshold, 2)
        print("Similarity threshold value:", sim_threshold)
        onto_list = [name for _, name in ONTOLOGY_ENTRIES]
        sim_list = [get_bertsim(path, sim_threshold) for path, _ in ONTOLOGY_ENTRIES]

        df = pd.DataFrame({'Ontology': onto_list, 'BERT_Sim': sim_list})
        df.to_csv(f'Ontology Selection/Results/BERTSIMforOntologySelection_{sim_threshold}.csv')
