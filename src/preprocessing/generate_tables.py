# !/usr/bin/env python3.6
# -*- coding:utf-8 -*-

"""
Merge training and validation data and labels to wide tables for sample analysis

@ Date: 2019-07-05
@ Author: OXPHOS
"""

import os
import re
import pandas as pd


def check_input_data(file_path, input_files):
    """
    Check if every input file (.a1) has a corresponding label file (.a2)
    
    :param file_path: absolute path to directory of the files
    :param input_files: .a1 file names
    :return: input file names
    """
    for f in input_files:
        assert os.path.isfile(os.path.join(file_path, f[:-1]+'2'))


def get_reference():
    """
    Parse OBT dict and TAX dict
    
    :return: OBT dataframe and TAX dataframe
    """
    ref_obt = pd.read_csv(os.path.join(os.getcwd(), '../../input_data/OBT.txt'), sep='|')
    ref_obt.columns = ['dict_id', 'dict_name']

    # For the OBT dict with id of format: xxxxxx(number, no 'OBT:' prefix)
    # for i in range(ref_obt.shape[0]):
    #     ref_obt.dict_id[i] = 'OBT:' + '0' * (6 - len(str(ref_obt.dict_id[i]))) + str(ref_obt.dict_id[i])

    ref_tax_raw = pd.read_csv(os.path.join(os.getcwd(), '../../input_data/TAX1_trim.txt'), sep='|')
    ref_tax = ref_tax_raw.groupby('tax_id').agg(lambda x: x.tolist())
    ref_tax.reset_index(inplace=True)
    ref_tax.columns = ['dict_id', 'dict_name', 'dict_name_class']
    ref_tax.dict_id = ref_tax.dict_id.astype(int).astype(str)

    return ref_obt, ref_tax


def merge_input_data(file_path, input_files, filename, abbrfile=None):
    """
    Generate a wide table with data from all input files
     
    :param file_path: absolute path to directory of the files
    :param input_files: input file name list
    :param filename: the output filename
    :param abbrfile: the path to the abbreviation extraction file
    """
    if filename:
        f_handler = open(os.path.join('../../input_data/wide_tables', filename), 'w')

    if abbrfile:
        abbr = pd.read_csv(abbrfile, sep='\t', header=None)
        abbr.columns = ['text_id', 'abbr', 'abbr_pos', 'fullname', 'fullname_pos']
        abbr = abbr.drop(columns=['fullname_pos']).drop_duplicates()
        abbr.abbr_pos = abbr.abbr_pos.str.replace('-', ' ')

    pattern = '.*norm-(.*)\..*'
    header = True
    for f in input_files:
        f_id = re.search(pattern, f).group(1)

        # Parse input files, with format of: label \t category \s [start \s end]rep  \t name
        f_in = pd.read_csv(os.path.join(file_path, f), sep='\t', header=None)
        f_in.columns = ['entity_id', 'mix', 'name']
        f_in[['category', 'positions']] = f_in.mix.str.split(n=1, expand=True)
        f_in.drop(columns=['mix'], inplace=True)
        f_in.insert(0, 'text_id', f_id)

        if abbrfile:
            for _, row in abbr[abbr.text_id.str.match(f.rstrip('.a1'))].iterrows():
                f_in['name'][f_in.positions.str.contains(row.abbr_pos)] = row.fullname

        if filename:
            if header:
                f_in.to_csv(f_handler, index=False, sep='\t')
                header = False
            else:
                f_in.to_csv(f_handler, index=False, header=False, sep='\t')


def merge_input_and_labels(file_path, input_files, filename=None, abbrfile=None):
    """
    Genaerate a wide table with data from all input files and label files

    :param file_path: absolute path to directory of the files
    :param input_files: input file name list
    :param filename: the output filename
    :param abbrfile: the path to the abbreviation extraction file
    :return if filename: tsv table to file
            else: dataframe
    """
    if filename:
        f_handler = open(os.path.join('../../input_data/wide_tables', filename), 'w')
        header = True
    else:
        final_res = pd.DataFrame()

    if abbrfile:
        abbr = pd.read_csv(abbrfile, sep='\t', header=None)
        abbr.columns=['text_id', 'abbr', 'abbr_pos', 'fullname', 'fullname_pos']
        abbr = abbr.drop(columns=['fullname_pos']).drop_duplicates()
        abbr.abbr_pos = abbr.abbr_pos.str.replace('-', ' ')

    pattern = '.*norm-(.*)\..*'
    for f in input_files:
        f_id = re.search(pattern, f).group(1)

        # Parse input files, with format of: label \t category \s [start \s end]rep  \t name
        f_in = pd.read_csv(os.path.join(file_path, f), sep='\t', header=None)
        f_in.columns = ['entity_id', 'mix', 'name']
        f_in[['category', 'positions']] = f_in.mix.str.split(n=1, expand=True)
        f_in.drop(columns=['mix'], inplace=True)
        f_in.insert(0, 'text_id', f_id)
        if abbrfile:
            for _, row in abbr[abbr.text_id.str.match(f.rstrip('.a1'))].iterrows():
                f_in['name'][f_in.positions.str.contains(row.abbr_pos)] = row.fullname


        # Parse output files, with format of: label \t category \s entity_lable \s referent
        f_out = pd.read_csv(os.path.join(file_path, f[:-1] + '2'), sep='\t|\s', header=None)
        f_out.columns = ['standard_id', 'dict_type', 'entity_id', 'dict_id']
        f_out.entity_id = f_out.entity_id.str.lstrip('Annotation:')
        f_out.dict_id = f_out.dict_id.str.lstrip('Referent:')
        f_out.insert(0, 'text_id', f_id)

        # Merge input and label files
        res = pd.merge(f_in, f_out, how='outer', on=['text_id', 'entity_id'])

        # Map dict_id in label files to dict_name in dicts
        res = pd.merge(res, ref_obt, how='left', on='dict_id')
        res = pd.merge(res, ref_tax, how='left', on='dict_id', suffixes=('', '_tmp'))
        res['dict_name'] = res.dict_name.fillna(res.dict_name_tmp)
        res.drop(columns=['dict_name_tmp'], inplace=True)

        if filename:
            if header:
                res.to_csv(f_handler, index=False, sep='\t')
                header = False
            else:
                res.to_csv(f_handler, index=False, header=False, sep='\t')
        else:
            final_res = pd.concat([final_res, res])

    if filename:
        return
    else:
        return final_res


def process_by_datasets(data_dir_name, output_to_file=True, check_integrity=True, abbrfile=None):
    """
    Extract information from all targeted files in the input directory,
    and output tables
    
    :param data_dir_name: input dir name 
    :param output_to_file: whether the results will be output to files
    :param check_integrity: whether to check each input file has a matched label file
    :param abbrfile: relative path to the abbreviation extraction file
    :return: if output_to_file: None
             else: dataframe
    """
    file_path = os.path.join(os.getcwd(), os.path.join('../../input_data', data_dir_name))
    input_files = [_ for _ in os.listdir(file_path) if _.endswith('.a1')]
    if check_integrity:
        check_input_data(file_path, input_files)
    if abbrfile:
        abbrfile = os.path.join(os.getcwd(), os.path.join('../../input_data', abbrfile))

    if output_to_file:
        merge_input_data(file_path, input_files, 'ab3p_entity_list_%s.tsv'%data_dir_name, abbrfile=abbrfile)
        # merge_input_and_labels(file_path, input_files, 'ab3p_entity_and_label_list_%s.tsv'%data_dir_name,
        #                        abbrfile=abbrfile)
    else:
        return merge_input_and_labels(file_path, input_files)


def process_by_entities(data_dir_list):
    """
    Generate a table with non-duplicated entities and corresponding labels from all given corpus

    :param data_dir_list: list of input dir names 
    """
    no_dup_entities = pd.DataFrame()

    for data_dir_name in data_dir_list:
        full_result = process_by_datasets(data_dir_name, output_to_file=False)
        full_result=full_result[~full_result.dict_id.isna()] # Remove title and abstract paragraphs
        no_dup_entities = pd.concat([no_dup_entities, full_result])

    # drop the columns with corpus-specific information
    no_dup_entities.drop(columns=['text_id', 'entity_id', 'positions', 'standard_id'], inplace=True)
    no_dup_entities.drop_duplicates(['name', 'dict_id'], inplace=True)

    no_dup_entities.to_csv(os.path.join('../../input_data/wide_tables', 'All_entities_with_labels_nodup.tsv'),
                           index=False, sep='\t')


if __name__ == "__main__":
    # Parse reference dictionaries: OBT dict and NCBI Taxdump
    ref_obt, ref_tax = get_reference()

    process_by_datasets('BioNLP-OST-2019_BB-norm_train',
                        abbrfile='Ab3P/BioNLP-OST-2019_BB-norm_train/abbreviations.txt')
    process_by_datasets('BioNLP-OST-2019_BB-norm_dev')
    # process_by_entities(['BioNLP-OST-2019_BB-norm_train', 'BioNLP-OST-2019_BB-norm_dev'])


