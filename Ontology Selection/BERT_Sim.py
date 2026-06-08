import numpy as np
import torch
import pandas as pd
import os
from owlready2 import get_ontology
from rdflib import *
from rdflib.namespace import split_uri
from STEM.app.Matcher.Corpus_based_methods.wordembedding import bert_Energy_tsdae


# Get all the class names of an ontology
def get_ontoclass(path):
    lst = []
    if path.endswith('.ttl'):
        g = Graph()
        g.parse(path)
        lst = []
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
    print('The classes of ontology_nodes:', nodes_lst)
    print('Number of classes:', len(nodes_lst))

    word_chain = ['Building', 'BuildingOperation', 'Room', 'Device', 'PhotovoltaicDevice', 'SmartMeteringObservation',
                  'PhotovoltaicMeasurement', 'StorageBatteryMeasurement', 'Battery', 'StorageBatteryDevice',
                  'EnergyConsumption', 'Energy']  # word chain generated from energy, building, device domains

    bert = bert_Energy_tsdae()
    sim_tensor = bert.sim(nodes_lst, word_chain)

    threshold = sim_tensor > sim_min
    sum_bert = sim_tensor[threshold].sum().item()

    print(f'BERT similarity is:', sum_bert)

    return round(sum_bert, 3)


if __name__ == '__main__':
    threshold_range = np.arange(0.8, 0.91, 0.05)
    print('Sim_threshold chosen:', threshold_range)
    for sim_threshold in threshold_range:
        sim_threshold = round(sim_threshold, 2)
        print("Similarity threshold value:", sim_threshold)
        sim_list = [get_bertsim("Ontologies/saref4bldg.rdf", sim_threshold),
                    get_bertsim("Ontologies/saref4ener.ttl", sim_threshold),
                    get_bertsim("Ontologies/Sargon.ttl", sim_threshold),
                    get_bertsim("Ontologies/bonsai.ttl", sim_threshold),
                    get_bertsim("Ontologies/BCOM.rdf", sim_threshold),
                    get_bertsim("Ontologies/sbeo.rdf", sim_threshold),
                    get_bertsim("Ontologies/BIMERR.ttl", sim_threshold),
                    get_bertsim("Ontologies/em-kpi.ttl", sim_threshold),
                    get_bertsim("Ontologies/FSO.ttl", sim_threshold),
                    get_bertsim("Ontologies/respond.owl", sim_threshold),
                    get_bertsim("Ontologies/SSN.ttl", sim_threshold),
                    get_bertsim("Ontologies/SOSA.ttl", sim_threshold),
                    get_bertsim("Ontologies/Building Topology.ttl", sim_threshold)]

        onto_list = ['SAREF4BLDG', 'SAREF4ENER', 'SARGON', 'BONSAI', 'BCOM', 'SBEO', 'BIMERR', 'EM-KPI',
                     'FSO', 'RESPOND', 'SSN', 'SOSA', 'BTO']

        df = pd.DataFrame({'Ontology': onto_list, 'BERT_Sim': sim_list})
        df.to_csv(f'Ontology Selection/Results/BERTSIMforOntologySelection_{sim_threshold}.csv')
