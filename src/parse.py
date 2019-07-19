import pandas as pd
import os


def parse_biotope_dict():
    return pd.read_csv(os.path.join(os.getcwd(), '../input_data/OBT.txt'), sep='|')


def parse_a1_files():
    pass


def parse_entity_and_label_table(tablename, return_id=False):
    data_in = pd.read_csv(os.path.join(os.getcwd(),
                                       '../input_data/wide_tables/%s' % tablename), sep='\t')
    data_in = data_in[~data_in.dict_id.isna()]
    if return_id:
        phe_hab = data_in[data_in.category.isin(['Phenotype', 'Habitat'])][['dict_id', 'dict_name']].reset_index()
    else:
        phe_hab = data_in[data_in.category.isin(['Phenotype', 'Habitat'])][['text_id', 'name', 'dict_name']]\
            .reset_index()
    return phe_hab


def parse_word_pair():
    pass
