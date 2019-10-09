# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:03:22 2019

@author: chenhaipeng
"""
import pandas as pd
import os
from tqdm import tqdm

# 依照类别区分字典类型
def Judge_type(a):
    if a == 'Microorganism':
        return 'NCBI_Taxonomy'
    else:
        return 'OntoBiotope'

# 将缩写转为全称
def ab3p(df, str):
    a_to_b = pd.read_csv('../input_data/Ab3P/BioNLP-OST-2019_BB-norm_'+ str + '/abbreviations.txt',
                         header=None, sep='\t', names=['1', 'a', 'page1', 'b', 'page2'])
    for i in range(len(a_to_b)):
        for j in range(len(df)):
            if a_to_b.iloc[i, 1] in df.iloc[j, 2]:
                df.iloc[j, 2] = a_to_b.iloc[i, 3]
    # df['word'].loc[df['word'].isin(a_to_b['a'])] = a_to_b['b']
    return df

# 对a1中的数据生成字典类型，并将实体简写转为全称
def SolveData(a1):
    # 删除起点终点信息只保留名词类别
    df = a1['category+start+end'].str.split(' ', expand=True)
    a1 = a1.drop('category+start+end', axis=1)
    a1.insert(1, 'category', df[0])
    # 首先删除类别为Title和Paragraph的行
    a1 = a1[~(a1['category'].isin(['Title']) | a1['category'].isin(['Paragraph']))]
    # 根据分类类别判断字典类型
    a1['dict_type'] = a1.apply(lambda x: Judge_type(x['category']), axis=1)
    a1 = ab3p(a1, 'test')
    return a1

# 依据a1中的实体名寻找预测的dict_id
def Match_ID(df, a1, mic, obt):
    # df.loc[df['dict_type'] == 'NCBI_Taxonomy'].apply(lambda  x: Search(df['id'], mic), axis=1)
    for i in range(len(df)):
        if df.iloc[i, 1] == 'NCBI_Taxonomy':
            df.iloc[i, 3] = mic.loc[mic['word'] == a1.iloc[i, 2]].iloc[0, 3]
            df.iloc[i, 3] = 'Referent:' + str(df.iloc[i, 3])
        else:
            a = obt.loc[obt['entity_id'] == a1.iloc[i, 0]]
            df.iloc[i, 3] = obt.loc[obt['entity_id'] == a1.iloc[i, 0]].iloc[0, 4]
            df.iloc[i, 3] = 'Referent:' + str(df.iloc[i, 3])
    return df

# 读取每一个a1文件信息，使用SolveData函数对文件内容进行删除文章题目、章节，大小写转化，id提取等操作，最后按格式生成a2文件
def MatchData(path, file_path, mic_result, obt_result):
    result = pd.DataFrame(columns=('id', 'dict_type', 'a1_id', 'dict_id'))
    # 编号带F的文件不包含前两行的文章信息，因此读取时无需去除前两行信息
    # if 'F' in file_path:
    #     a1 = pd.read_csv(path + '/' + file_path, header=None, sep='\t')
    #     # 类别与文字起点和终点用空格分隔，稍后再单独处理
    #     a1.columns = ['a1_id', 'category+start+end', 'entity']
    #     a1 = SolveData(a1)
    # else:
    a1 = pd.read_csv(path + '/' + file_path, header=None, sep='\t')
    # 类别与文字起点和终点用空格分隔，稍后再单独处理
    a1.columns = ['a1_id', 'category+start+end', 'entity']
    a1 = SolveData(a1)
    # 按格式生成a2文件
    result['a1_id'] = 'Annotation:' + a1['a1_id']
    result['dict_type'] = a1['dict_type']
    for i in range(len(result['a1_id'])):
        result.iloc[i, 0] = 'N' + str(i+1)
    # 根据实体名在MIC和OBT匹配表中寻找对应的dict_id
    result = Match_ID(result, a1, mic_result, obt_result.loc[obt_result['text_id'] == file_path[8:-3]])
    # 生成的a2文件保存在BioNLP-OST-2019_BB-norm_test_pre文件夹中

    f_handler = open('BioNLP-OST-2019_BB-norm_pre./' + file_path[:-3] + '.a2', 'w')
    for i in range(len(result)):
        f_handler.write(result.iloc[i, 0]+'\t'+result.iloc[i, 1]+' '+result.iloc[i, 2]+' '+result.iloc[i, 3]+'\n')

    # result.to_csv('BioNLP-OST-2019_BB-norm_pre./' + file_path[:-3] + '.a2', sep=' ', header=None, index=False)

# 批量读取文件，记录每一个编号下的a1件传递给MatchData进行处理
def TraverFile(file_path, mic_result, obt_result):
    g = os.walk(file_path)
    for path, dir_list, f_list in g:
        T = 0
        # 以此读取文件，以同个编号下的两个文件为一个单位，读取a1文件中的实体信息进行匹配
        file_list = tqdm(f_list)
        for file_name in file_list:
            if file_name.endswith('.a1'):
                MatchData(path, file_name, mic_result, obt_result)

if __name__=="__main__":
    # 读入两个词典的预测匹配表
    # Microorganism_result中列名为[text_id, word, matched_name, matched_id, score]
    mic_result = pd.read_csv('Microorganism_result.csv', sep='\t', encoding='utf-8')

    # OntoBiotope_result中列名为[text_id, entity_id, word, matched_name, matched_id]
    obt_result = pd.read_csv('OntoBiotope_result.tsv', sep='\t', encoding='utf-8')
    # obt_result.drop_duplicates(subset=['word'], keep='first', inplace=True)
    # obt_result = obt_result.reset_index(drop=True)

    # 根据匹配表生成a2文件
    TraverFile(r"../input_data/BioNLP-OST-2019_BB-norm_test", mic_result, obt_result)