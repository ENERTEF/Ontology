# Ontology Development

This repository is structured according to the main steps of the ontology development framework, which are as follows:

1. [Domain and Scope Determination](#domain-and-scope-determination)
2. [Requirement Specification](#requirement-specification)
3. [Knowledge Search](#knowledge-search)
4. [Ontology Selection](#ontology-selection)
5. [Ontology Matching](#ontology-matching)
6. [Ontology Merging](#ontology-merging)
7. [Conceptualization](#conceptualization)
8. [Evaluation](#evaluation)

---

## Domain and Scope Determination 
The first step in ontology development is to clearly define its purpose, target users, use cases, and requirements. Following the guidelines in the Ontology Requirement Specification Document (ORSD), we focus on domains such as energy, building, weather.

## Requirement Specification 
The ORSD outlines two types of ontology requirements:
- **Non-functional Requirements:** General criteria the ontology must meet.
- **Functional Requirements:** Specific content-based needs, expressed as competency questions (CQs) along with their answers.
Keywords extracted from the CQs are used for knowledge search.

## Knowledge Search 
This phase involves identifying existing ontologies and data models relevant to target domain.

### Knowledge Search Results
| Ontology / Standard                   | Domain                            |
| ------------------------------------- | --------------------------------- |
| SAREF4BLDG                            | Energy, Building, Device          |
| SAREF4ENER                            | Energy, Device                    |
| SAREF4GRID                            | Energy, Device                    |
| SARGON                                | Energy, Building, Device          |
| Smart Building Energy Ontology (SBEO) | Building                          |
| EM-KPI                                | Energy, Building, Device, Weather |
| Flow System Ontology (FSO)            | Energy, Device                    |
| Semantic Sensor Network (SSN)         | Device                            |
| Building Ontology                     | Building                          |
| Weather Ontology                      | Weather                           |
| Sensor Ontology                       | Device                            |
| Building Energy Domain Ontology (BED) | Energy, Building, Device, Weather |
| Open Energy Ontology (OEO)            | Energy, Device, Weather           |
| FIWARE                                | Energy                            |
| CIM                                   | Energy, Device                    |


## Ontology Selection 
Based on semantic similarity scores from EnergyBERT, the ontologies for next steps are selected. 

