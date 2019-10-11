# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:03:22 2019

@author: chenhaipeng
@author: OXPHOS
"""

import pandas as pd
import os
from tqdm import tqdm


def judge_type(a):
    """
    Return entity type
    :param a: the entity to be determined
    :return: entity type
    """
    if a == 'Microorganism':
        return 'NCBI_Taxonomy'
    else:
        return 'OntoBiotope'


def solve_data(a1):
    """
    further process .a1 file: extract category information, and remove title and paragraph
    :param a1: the .a1 file
    :return: trimmed .a1 file info
    """
    # extract category information
    df = a1['category+start+end'].str.split(' ', expand=True)
    a1 = a1.drop('category+start+end', axis=1)
    a1.insert(1, 'category', df[0])

    # remove title and paragraph if exist
    a1 = a1[~(a1['category'].isin(['Title']) | a1['category'].isin(['Paragraph']))]

    # category normalization (to dictionary type)
    a1['dict_type'] = a1.apply(lambda x: judge_type(x['category']), axis=1)

    return a1


def match_dict_id(a2, a1, obt):
    """
    Fill in predicted normalized entity id to .a2 file with entity_id
    :param a2: the .a2 dataframe
    :param a1: the .a1 dataframe
    :param obt: prediction result
    :return: filled .a2 file
    """
    for i in a2.index:
        if a2.loc[i, 'dict_type'] == 'OntoBiotope':
            a2.loc[i, 'dict_id'] = 'Referent:' + str(
                obt[obt['entity_id'] == a1.loc[i, 'a1_id']].matched_id.iloc[0])[4:]
        else:
            a2.loc[i, 'dict_id'] = 'Referent:' + str('-1')
    return a2


def match_data(path, file_name, obt_result):
    """
    Generate corresponding .a2 file for each .a1 file
    
    :param path: path to .a1 and .a2 files
    :param file_name: the .a1 file to process
    :param obt_result: prediction result
    """

    # read and parse .a1 file
    a1 = pd.read_csv(path + '/' + file_name, header=None, sep='\t')
    # multiple sep symbols (tab and space) were used. Process with solve_data()
    a1.columns = ['a1_id', 'category+start+end', 'entity']
    a1 = solve_data(a1)

    # fill in content to .a2 file
    result = pd.DataFrame(columns=('id', 'dict_type', 'a1_id', 'dict_id'))
    result['a1_id'] = 'Annotation:' + a1['a1_id']
    result['dict_type'] = a1['dict_type']
    for i in range(len(result['a1_id'])):
        result.iloc[i, 0] = 'N' + str(i+1)
    result = match_dict_id(result, a1, obt_result[obt_result['text_id'] == file_name[8:-3]])

    # output .a2 file
    f_handler = open('../output/' + file_name[:-3] + '.a2', 'w')
    for i in range(len(result)):
        f_handler.write(result.iloc[i, 0]+'\t'+result.iloc[i, 1]+' '+result.iloc[i, 2]+' '+result.iloc[i, 3]+'\n')


if __name__=="__main__":
    if not os.path.exists('../output/'):
        os.makedirs('../output')

    # Read in the prediction result
    obt_result = pd.read_csv('OntoBiotope_result.tsv', sep='\t', encoding='utf-8')\
        .reset_index(drop=True)

    # Generate corresponding .a2 file for each .a1 file
    g = os.walk(r"../input_data/BioNLP-OST-2019_BB-norm_test")
    for path, dir_list, f_list in g:
        file_list = tqdm(f_list)
        for file_name in file_list:
            if file_name.endswith('.a1'):
                match_data(path, file_name, obt_result)
