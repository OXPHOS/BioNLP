# BacReader (WIP)
#### For BioNLP-OST 2019 task [Bacteria Biotope](https://sites.google.com/view/bb-2019/home)
## Table of content
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Data](#data)
    - [Dictionaries](#dictionaries)
    - [Free text and entities](#free-text-and-entities)
- [Other resources](#other-resources)
    - [Ab3P](#ab3p)
    - [Word Vector Space Model](#word-vector-space-model)
- [Dependencies](#dependencies)
- [Run instructions](#run-instructions)
- [Results](#results)

### Overview
The project aims to provide solutions for microorganism-related free-text entity nomralization
 /linking as suggested by the [task](https://sites.google.com/view/bb-2019/task-description).
 
To be more specific, pre-annotated microorganism-related entities, as well as different types of dictionaries
were provided. The system links the pre-annotated entities to standard concepts in dictionaries
with a 2-step method combining a perfect-match module and an ensemble shallow CNN module by converting entities 
to pre-trained word vectors.

**NOTE**: species names were normalized with ensemble editing distance method, which was not present here. 

### Pipeline
![Figure 1. Model Architecture Overview](https://github.com/OXPHOS/BioNLP/blob/master/elements/flowchart.png)

#### [generate_table.py]()
Convert input to `dataframe` format for easier downstream processing. For training and dev datasets, labels were added.

Abbreviations were extracted with [Ab3P](#ab3p).

#### [generate_vsm.py](https://github.com/OXPHOS/BioNLP/blob/master/src/generate_vsm.py)
Convert entities to word vectors with [pre-trained word vector space model](#word-vector-space-model) for downstream system.

Different conversion rules were tested. Most of them were commented out except for the one 
used for ensemble method as described in the paper(`generate_five_fold_dataset(prediction=True)`)  

#### [main.py](https://github.com/OXPHOS/BioNLP/blob/master/src/main.py)

- **perfect match module**
  
  Link free-text entities to standard concepts by exact-matching rules.
  
  The rules included:
  - Hyphens were replaced with spaces.
  - Characters except alphabetic letters and spaces were removed.
  - Case-insensitive string matching was performed between the free text entities and standard entities.
  - All types of 'xx cheese' was linked to the corresponding cheese category by string matching
  
- **CNN module**
   
   Input: 8x200, output: 1x139
   ```        
    self.model.add(Conv1D(filters=arg, kernel_size=4, padding='same',
                          input_shape=(entity_embedding_size, vector_len)))
    self.model.add(MaxPooling1D(entity_embedding_size))
    self.model.add(Dense(139))
    self.model.compile(loss='cosine_proximity', optimizer=SGD())
   ```

- **Ensemble voting module**

  - Identify the most similar standard concepts via cosine similarity
  - Return the standard concepts with the majority votes

#### Evaluation
The results was formatted as the given label files ([`.a2` file](#free-text-and-entities)) with [`online_test_output.py`]
(https://github.com/OXPHOS/BioNLP/blob/master/src/online_test_output.py).

The test suite could be found [here](https://sites.google.com/view/bb-2019/evaluation-results#h.p_ru-Q1Kt6ssyr)

### Data
#### Dictionaries
Two types of entities were involved in the task: phenotype, which describes microbial characteristics, and habitat, which describes physical places where microorganisms could be observed. Dictionary with 3602 standard concepts was also provided by the task. 

 ![standard dictionary example](https://github.com/OXPHOS/BioNLP/blob/master/elements/obo.png)
 
 As can be seen, in the original dictionary, each concept is assigned to a unique ID, while its hierarchical information of its direct parents is also listed. In our model, the hierarchical information is omitted.

 ![standard dictionary after flatten](https://github.com/OXPHOS/BioNLP/blob/master/elements/obo2.png)

#### Free text and entities

The statistics of all available entities are listed below.

|       |          |               Total Number         |     Number after de-duplication    |
|-------|----------|-------------|------------|---------|-------------|------------|---------|
|       | Article# | Habitat     | Phenotype  | H+P     | Habitat     | Phenotype  | H+P    
|-------|----------|-------------|------------|---------|-------------|------------|---------|
| Train | 133      | 1118        | 369        | 1487    | 627         | 176        | 803     |
|-------|----------|-------------|------------|---------|-------------|------------|---------|
| Dev   | 66       | 610         | 161        | 771     | 348         | 97         | 445     |
|-------|----------|-------------|------------|---------|-------------|------------|---------|
| Test  | 97       | 924         | 252        | 1176    | 596         | 148        | 744     |

`.a1` files include different pieces of literature. Numbering, type of the row, start and end positions, as well as the 
 corresponding text were presented.
 
 ![`.a1` file example](https://github.com/OXPHOS/BioNLP/blob/master/elements/a1.png)
 
 `.a2` files include the label to each annotated text in `.a1` file. Entity type as well as the unique code in the corresponding
 standard dictionary was provided.
 
 ![`.a2` file example](https://github.com/OXPHOS/BioNLP/blob/master/elements/a2.png)
 

### Other resources
#### Ab3P

[Ab3P](https://github.com/ncbi-nlp/Ab3P) is an abbreviation detection tool developed specifically for biomedical concepts. 
It reached 96.5% precision and 83.2% recall on 1250 randomly selected MEDLINE records as suggested by Sohn et al (Sohn S, Comeau DC, Kim W, Wilbur WJ. (2008) Abbreviation definition identification based on automatic precision estimates. BMC Bioinformatics.  25;9:402. PubMed ID: 1881755).

Ab3P-detected abbreviations were provided as [separate input files](https://sites.google.com/view/bb-2019/supporting-resources) 
by the task organizers.

#### Word Vector Space Model
A set of word vectors induced on a combination of PubMed and PMC texts with texts extracted from a recent English Wikipedia dump
The 4GB vectors can be downloaded [here](http://bio.nlplab.org/) as `wikipedia-pubmed-and-PMC-w2v.bin`
(Sampo Pyysalo, Filip Ginter, Hans Moen, Tapio Salakoski and Sophia Ananiadou. Distributional Semantics Resources for Biomedical Text Processing.  LBM 2013.)

### Dependencies
```
keras==2.2.4
scikit-learn==0.21.2
gensim==3.4.0
pandas==0.25.1
nummpy==1.16.5
```
The detailed description of the dependencies could be found [here](https://github.com/OXPHOS/BioNLP/blob/master/environment.yml)

### Run instructions
Rename `/sample_data` to `/input_data` and change the corresponding paths if necessary.

Datasets could be downloaded at the [Bacteria Biotope website](https://sites.google.com/view/bb-2019/dataset)

Then, run the scripts as the pipeline suggested.


### Results
| Team | Habitat | Phenotype |
|---------|---------|-----------|
| PADIA_BacReader    | 0.684    | 0.758     |
|---------|---------|-----------|
| Challenge-provided baseline    | 0.559    | 0.581     |
|---------|---------|-----------|
| AmritaCen_healthcare    | 0.522    | 0.646     |
|---------|---------|-----------|
| BLARI_GMU    | 0.615   | 0.646     |
|---------|---------|-----------|
| BOUN-ISIK    | 0.687    | 0.566     |