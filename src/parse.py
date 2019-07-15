import pandas as pd
import os


def parse_biotope_dict():
    return pd.read_csv(os.path.join(os.getcwd(), '../input_data/OBT.txt'), sep='|')


def parse_a1_files():
    pass


def parse_entity_and_label_table(tablename):
    data_in = pd.read_csv(os.path.join(os.getcwd(),
                                       '../input_data/wide_tables/%s' %tablename), sep='\t')
    data_in = data_in[~data_in.dict_id.isna()]
    phe_hab = data_in[data_in.category.isin(['Phenotype', 'Habitat'])][['name', 'dict_name']].reset_index()
    names, labels = phe_hab.name, phe_hab.dict_name
    return names, labels


def parse_word_pair():
    pass
