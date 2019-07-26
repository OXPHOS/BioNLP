"""
Convert entities, labels and dictionary to vectors with vector space model
Vector space model: http://bio.nlplab.org/, wikipedia-pubmed-and-PMC-w2v.bin

@ Date: 2019-07-12
@ Author: OXPHOS
"""

import gensim
import numpy as np
from sklearn.decomposition import PCA
from src import entity_embedding_size, context_embedding_size, vector_len
from src.parse import *


def reduce_dimension(vectors):
    pca = PCA(n_components=0.95)
    return pca.fit_transform(vectors)


def averaging_vectors(namelist, default_value=True, fill_na=False):
    if default_value:
        unk = w2v_model.get_vector('unknown')
        vectors = np.array([unk for _ in range(len(namelist))])
    else:
        vectors = np.array([[np.nan] * vector_size for _ in range(len(namelist))])
    i = 0
    for e in namelist:
        name = ''.join(filter(whitelist.__contains__, e.replace('-', ' ')))
        vec = np.zeros(vector_size)
        count = 0
        for w in name.split():
            if w in w2v_model.vocab:
                vec += w2v_model.get_vector(w)
                count += 1
            elif w.lower() in w2v_model.vocab:
                vec += w2v_model.get_vector(w.lower())
                count += 1
        if count:
            vectors[i] = vec / count
        i += 1

    return vectors


def fixed_length_vectors(namelist, embedding_size=entity_embedding_size):
    vectors = np.zeros((len(namelist), embedding_size, vector_size))
    i = 0
    for e in namelist:
        name = ''.join(filter(whitelist.__contains__, e.replace('-', ' ')))
        if len(name.split()) > embedding_size:
            tmp = list()
            for w in name.split():
                if w in w2v_model.vocab:
                    tmp.append(w2v_model.get_vector(w))
                elif w.lower() in w2v_model.vocab:
                    tmp.append(w2v_model.get_vector(w.lower()))
            avg_factor = np.ceil(len(tmp) / embedding_size).astype(int)
            for k in range(0, len(tmp), avg_factor):
                vectors[i][k//avg_factor] = np.mean(tmp[k:k+avg_factor])
        else:
            j = 0
            for w in name.split():
                if w in w2v_model.vocab:
                    vectors[i][j] = w2v_model.get_vector(w)
                elif w.lower() in w2v_model.vocab:
                    vectors[i][j] = w2v_model.get_vector(w.lower())
                j += 1
        i += 1
    return vectors


def weighted_vectors():
    # genia-pos
    pass


def fixed_length_vectors_by_text(names):
    # print(names.groupby('text_id').count())
    names_by_text = names.groupby('text_id').aggregate(lambda x: set(x))
    names_by_text['vec'] = None
    # for idx in names_by_text.index:
    #     tmp = {}
    #     for n in names_by_text.name[idx]:
    #         tmp[n] = fixed_length_vectors(list(n), 8)[0]
    #     names_by_text.name[idx] = tmp
    # print(names_by_text.loc['F-25496341-008', 'name'].values())

    for idx in names_by_text.index:
        tmp = fixed_length_vectors(names_by_text.name[idx], entity_embedding_size)
        tmp = tmp.reshape(-1, tmp.shape[-1])
        tmp = np.lib.pad(tmp, ((0, context_embedding_size-entity_embedding_size-tmp.shape[0]), (0, 0)),
                         'constant', constant_values=(0))
        names_by_text.vec[idx] = tmp
    return names_by_text


def process_biotope_dict():
    ref = parse_biotope_dict()

    vectors = averaging_vectors(ref.name)
    np.save(os.path.join(path, 'OBT_VSM_vectors.npy'), vectors)

    ref['vec'] = list(vectors)
    ref.to_csv(os.path.join(path, 'OBT_VSM.tsv'), sep='\t')


def process_entity_and_label_table(tablename):
    prefix = tablename.split('_', -1)[-1]
    names_and_labels = parse_entity_and_label_table(tablename)
    names_vec = fixed_length_vectors(names_and_labels.name)
    labels_vec = averaging_vectors(names_and_labels.dict_name)
    names_and_labels.to_csv(os.path.join(path, '%s_names_and_labels.tsv' %prefix), sep='\t')
    np.save(os.path.join(path, '%s_names_vectors.npy' %prefix), names_vec)
    np.save(os.path.join(path, '%s_labels_vectors.npy' %prefix), labels_vec)


def generated_normalized_dict_and_labels():
    ref = parse_biotope_dict()
    vectors = averaging_vectors(ref.name)
    vectors = reduce_dimension(vectors)
    np.save(os.path.join(path, 'OBT_VSM_norm.npy'), vectors)
    ref['vec'] = list(vectors)
    ref.to_csv(os.path.join(path, 'OBT_VSM_norm.tsv'), sep='\t')

    for tablename in ['entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv',
                      'entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv']:
        labels_id_and_labels = parse_entity_and_label_table(tablename)
        labels_vec = vectors[list(map(pd.Index(ref.id).get_loc, labels_id_and_labels.dict_id))]
        np.save(os.path.join(path, '%s_labels_vectors_norm.npy' %tablename.split('_', -1)[-1]),
                labels_vec)


def generate_context_entity_list(tablename):
    names_and_labels = parse_entity_and_label_table(tablename)
    names_vec = fixed_length_vectors(names_and_labels.name)

    names_by_text = fixed_length_vectors_by_text(names_and_labels[['text_id', 'name']])
    concat_vec = np.stack(names_by_text.loc[names_and_labels.text_id, 'vec'], axis=0)
    names_vec = np.concatenate((names_vec, concat_vec), axis=1)
#     print(names_vec.shape)
    names_and_labels.to_csv(os.path.join(path, '%s_names_and_labels_with_context.tsv' %tablename.split('_', -1)[-1]), sep='\t')
    np.save(os.path.join(path, '%s_names_vectors_with_context.npy' %tablename.split('_', -1)[-1]), names_vec)


def generate_five_fold_dataset(prediction=False):
    ref = parse_biotope_dict()
    vectors = averaging_vectors(ref.name)
    vectors = reduce_dimension(vectors)
    np.save(os.path.join(path, 'OBT_VSM_norm.npy'), vectors)
    ref['vec'] = list(vectors)
    ref.to_csv(os.path.join(path, 'OBT_VSM_norm.tsv'), sep='\t')

    names_and_labels = parse_entity_and_label_table('entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv')
    names_and_labels = pd.concat([names_and_labels,
                                 parse_entity_and_label_table('entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv')])
    if prediction:
        training_size = len(names_and_labels)
        names_and_labels = pd.concat([names_and_labels,
                                     parse_entity_table('entity_list_BioNLP-OST-2019_BB-norm_test.tsv')])
    names_and_labels.reset_index(drop=True, inplace=True)

    if prediction:
        training_set = names_and_labels.iloc[:training_size]
        test_set = names_and_labels[training_size:]
    else:
        test_set = names_and_labels.sample(frac=0.17)
        training_set = names_and_labels[~names_and_labels.index.isin(test_set.index)]

    for dataset, datatype in [(test_set, 'test'), (training_set, 'train')]:
        dataset.to_csv(os.path.join(path, '5fold_%s.tsv' % datatype), sep='\t')

        names_vec = fixed_length_vectors(dataset.name)
        np.save(os.path.join(path, '5fold_%s_names_vectors.npy' % datatype), names_vec)

        if prediction and datatype=='test':
            pass
        else:
            labels_vec = vectors[list(map(pd.Index(ref.id).get_loc, dataset.dict_id))]
            np.save(os.path.join(path, '5fold_%s_labels_vectors_norm.npy' %datatype), labels_vec)


if __name__=="__main__":
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
       '../input_data/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    vector_size = w2v_model.vector_size
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    path = os.path.join(os.getcwd(), '../input_data/vsm/')

    # process_biotope_dict(default_value=False)
    # process_entity_and_label_table('entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv')
    # process_entity_and_label_table('entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv')
    # generated_normalized_dict_and_labels()
    # generate_context_entity_list('entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv')
    # generate_context_entity_list('entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv')
    generate_five_fold_dataset(prediction=True)
