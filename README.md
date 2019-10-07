#BacReader (WIP)
#### For BioNLP-OST 2019 task [Bacteria Biotope](https://sites.google.com/view/bb-2019/home)
## Table of content
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Dependencies](#dependencies)
- [Data](#data)
- [Other resources](#other-resources)
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
[Figure 1. Model Architecture Overview]()

#### [generate_table.py]()
Convert input to `dataframe` format for easier downstream processing.

For training and dev datasets, labels were added.

Abbreviations were extracted with [Ab3P]().

#### [generate_vsm.py]()
Convert entities to word vectors with [pre-trained word vector space model]().

Different conversion rules were tested. They were commented out except for the one 
used for ensemble method as described in the paper()`generate_five_fold_dataset(prediction=True)`)  

### Dependencies
`pandas==0.25.1`
`keras==2.2.4`
The detailed description of the dependencies could be found [here]()

### Data


|       |          |               Total Number         |     Number after de-duplication    |
|-------|----------|-------------|------------|---------|-------------|------------|---------|
|       | Article# | Habitat     | Phenotype  | H+P     | Habitat     | Phenotype  | H+P    
|-------|----------|-------------|------------|---------|-------------|------------|---------|
| Train | 133      | 1118        | 369        | 1487    | 627         | 176        | 803     |
|-------|----------|-------------|------------|---------|-------------|------------|---------|
| Dev   | 66       | 610         | 161        | 771     | 348         | 97         | 445     |
|-------|----------|-------------|------------|---------|-------------|------------|---------|
| Test  | 97       | 924         | 252        | 1176    | 596         | 148        | 744     |





### Other resources
#### Ab3P

#### WordVector

### Run instructions
Rename `/sample data` to `/input data`.

### Results
