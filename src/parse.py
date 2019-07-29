"""
Read files and return pd.DataFrame

@ Date: 2019-07-12
@ Author: OXPHOS
"""

import pandas as pd
import os


def parse_biotope_dict():
    """
    :return: Biotope Dictionary in DataFrame 
    """
    return pd.read_csv(os.path.join(os.getcwd(), '../input_data/OBT.txt'), sep='|')


def parse_a1_files():
    pass


def parse_entity_table(tablename):
    """
    Extract phenotype and habitat entities from given table
    NOTE: Adding ab3p prefix to all tables reading in
    
    :param tablename: the table to process
    :return: DataFrame of entities with category of phenotype or habitat
    """
    data_in = pd.read_csv(os.path.join(os.getcwd(),
                                       '../input_data/wide_tables/ab3p_%s' % tablename), sep='\t')
    phe_hab = data_in[data_in.category.isin(['Phenotype', 'Habitat'])][['text_id', 'entity_id', 'name']]\
        .reset_index()
    return phe_hab


def parse_entity_and_label_table(tablename):
    """
    Extract phenotype and habitat entities from given table, with label information of give entities
    NOTE: Adding ab3p prefix to all tables reading in

    :param tablename: the table to process
    :return: DataFrame of entities with category of phenotype or habitat
    """
    data_in = pd.read_csv(os.path.join(os.getcwd(),
                                       '../input_data/wide_tables/ab3p_%s' % tablename), sep='\t')
    # Remove title and paragraph
    data_in = data_in[~data_in.dict_id.isna()]

    phe_hab = data_in[data_in.category.isin(['Phenotype', 'Habitat'])]
    phe_hab = phe_hab[['text_id', 'entity_id', 'name', 'dict_name', 'dict_id']].reset_index()
    return phe_hab
