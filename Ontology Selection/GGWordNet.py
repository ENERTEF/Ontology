# -*- coding = utf-8 -*-
# @Time: 2023/10/15 12:05
# @Author: Outing Gao, Lei Zhao
# @File: getset.py
# @Software: PyCharm
from numpy import *
import pandas as pd
import os
from nltk.corpus import wordnet as wn
from owlready2 import get_ontology
from rdflib import *
from rdflib.namespace import split_uri


# Get all the class names of an ontology
def get_ontoclass(path):
    lst = []
    if path.endswith('.ttl'):
        g = Graph()
        g.parse(path)
        lst = []
        for s, p, o in g.triples((None, RDF.type, OWL.Class)):
            if isinstance(s, URIRef):
                class_name = split_uri(s)[1]
                lst.append(class_name)
    else:
        onto = get_ontology(path).load()
        for i in list(onto.classes()):
            i = str(i).split('.')[1]
            lst.append(i)
    return lst


# Get Synonyms and hyponyms of a wordlist
def get_Synonymsandhyponyms(lst):
    lemma_list = []
    for word in lst:
        if wn.synsets(word):
            for synset in wn.synsets(word):
                if synset.pos() == 'n':  # Guarantee the synset to be nouns
                    for hyponym in synset.hyponyms():
                        lemma_list.extend(hyponym.lemma_names())
                    lemma_list.extend(synset.lemma_names())
                    lemma_list.append(word)  # Including the word itself
    return sorted(set(lemma_list), key=lemma_list.index)


class Word_net(object):
    def __init__(self, str_a, str_b):
        self.str_a = str_a
        self.str_b = str_b

    # Get the max path distance similarity of synsets from two given words
    def path_sim(self):
        synsets1 = wn.synsets(self.str_a)
        synsets2 = wn.synsets(self.str_b)
        path_sim = 0

        for synset1 in synsets1:
            for synset2 in synsets2:
                if synset1.pos() == 'n' and synset2.pos() == 'n':
                    try:
                        if synset1.path_similarity(synset2) >= path_sim:
                            path_sim = synset1.path_similarity(synset2)
                    except Exception as e:
                        print(synset1, synset2)
                        print("path: " + str(e))

        return round(path_sim, 3)

    # Get the pair of synsets with max path distance similarity (min 0.5) from two given words
    def path_simsyn(self):
        synsets1 = wn.synsets(self.str_a)
        synsets2 = wn.synsets(self.str_b)
        path_sim = 0.5
        list_a = []
        list_b = []
        match_flag = False

        for synset1 in synsets1:
            for synset2 in synsets2:
                if synset1.pos() == 'n' and synset2.pos() == 'n':
                    try:
                        if synset1.path_similarity(synset2) >= path_sim:
                            match_flag = True
                            path_sim = synset1.path_similarity(synset2)
                            max_a = synset1
                            max_b = synset2
                    except Exception as e:
                        print(synset1, synset2)
                        print("path: " + str(e))
        if match_flag:
            list_a.append(max_a)
            list_b.append(max_b)

        return list_a, list_b


# Generate the pair of words whose path_sim higher than 0.5
def get_simSN(lst1, lst2):
    for word_a in lst1:
        if not wn.synsets(word_a):
            lst1.remove(word_a)
    for word_b in lst2:
        if not wn.synsets(word_b):
            lst2.remove(word_b)

    simSN = []
    simRef = []
    for word_a in lst1:
        for word_b in lst2:
            if Word_net(word_a, word_b).path_sim() >= 0.5:
                simSN.append(word_a)
                simRef.append(word_b)

    return sorted(set(simSN), key=simSN.index), sorted(set(simRef), key=simRef.index)


def get_SSG(lst):
    poly = []
    SSG = 0

    for word in lst:
        if wn.synsets(word):
            poly.append(len(wn.synsets(word)))
        else:
            lst.remove(word)

    for i in poly:
        if i != 0:
            SSG += 1 / i
    return round(SSG, 3)


def get_SG(list_a, list_b):
    depth_LCS = []
    path_list = []
    a = 0.2
    b = 0.6
    sg_list = []
    for i in list_a:
        for j in list_b:
            var = i.shortest_path_distance(j)
            if var is not None:
                length = var
                path_list.append(length)
                for hyper in i.lowest_common_hypernyms(j):
                    depth = hyper.min_depth()
                    depth_LCS.append(depth)
                    c = exp(-(a * length))
                    f = exp(b * depth) - exp(-b * depth)
                    g = exp(b * depth) + exp(-b * depth)
                    h = c * (f / g)
                    sg_list.append(h)
                    sg = sum(sg_list)

    return round(sg, 3)


# Get two word lists, screening most related pair synset of each pair of words from word lists
def get_simSN_in_synsets(lst1, lst2):
    for word_a in lst1:
        if not wn.synsets(word_a):
            lst1.remove(word_a)
    for word_b in lst2:
        if not wn.synsets(word_b):
            lst2.remove(word_b)

    result_a = []
    result_b = []
    for word_a in lst1:
        for word_b in lst2:
            list_a, list_b = Word_net(word_a, word_b).path_simsyn()
            if len(list_a) > 0:
                result_a.append(list_a[0])
                result_b.append(list_b[0])
    return result_a, result_b


def get_GG(path, chain):
    print("Ontology file name:", os.path.basename(path))
    nodes_lst = get_ontoclass(path)
    print('The classes of ontology_nodes:', nodes_lst)
    simSN = get_simSN(chain, nodes_lst)
    print('The sim Semantic Network of two lists:', simSN)
    ssg = get_SSG(simSN[1])
    print('SSG of two lists:', ssg)
    sn_keywords, sn_onto = get_simSN_in_synsets(chain, nodes_lst)
    print(sn_keywords)
    print(sn_onto)
    sg = get_SG(sn_keywords, sn_onto)
    print('SG of two lists:', sg)
    gg = ssg + sg
    print('GG of two lists:', gg)
    return round(gg, 3)


if __name__ == '__main__':
    # word chain generated from the information of energy, building, device domains
    word_chain = ['Building', 'BuildingOperation', 'Room', 'Device', 'PhotovoltaicDevice', 'SmartMeteringObservation',
                  'PhotovoltaicMeasurement', 'StorageBatteryMeasurement', 'Battery', 'StorageBatteryDevice',
                  'EnergyConsumption', 'Energy']
    word_chain_new = get_Synonymsandhyponyms(word_chain)
    print('The synonyms and hyponyms of the word chain:', word_chain_new)

    gg_list = [get_GG("Ontologies/saref4bldg.rdf", word_chain_new),
               get_GG("Ontologies/saref4ener.ttl", word_chain_new),
               get_GG("Ontologies/Sargon.ttl", word_chain_new),
               get_GG("Ontologies/bonsai.ttl", word_chain_new),
               get_GG("Ontologies/BCOM.rdf", word_chain_new),
               get_GG("Ontologies/sbeo.rdf", word_chain_new),
               get_GG("Ontologies/BIMERR.ttl", word_chain_new),
               get_GG("Ontologies/em-kpi.ttl", word_chain_new),
               get_GG("Ontologies/FSO.ttl", word_chain_new),
               get_GG("Ontologies/respond.ttl", word_chain_new),
               get_GG("Ontologies/SSN.ttl", word_chain_new),
               get_GG("Ontologies/SOSA.ttl", word_chain_new),
               get_GG("Ontologies/Building Topology.ttl", word_chain_new)]

    onto_list = ['SAREF4BLDG', 'SAREF4ENER', 'SARGON', 'BONSAI', 'BCOM', 'SBEO', 'BIMERR', 'EM-KPI',
                 'FSO', 'RESPOND', 'SSN', 'SOSA', 'BTO']

    df = pd.DataFrame({'Ontology': onto_list, 'GG': gg_list})
    df.to_csv('Ontology Selection/GGforOntologySelection.csv')
