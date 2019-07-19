import gensim
import numpy as np
from sklearn.decomposition import PCA
from src import embedding_size
from src.parse import *


def reduce_dimension(vectors):
    pca = PCA(n_components=0.95)
    return pca.fit_transform(vectors)


def generate_averaging_vectors(namelist, default_value=True, fill_na=False):
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


def generate_fixed_length_vectors(namelist):
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


def generate_weighted_vectors():
    # genia-pos
    pass


def process_biotope_dict():
    ref = parse_biotope_dict()

    vectors = generate_averaging_vectors(ref.name)
    np.save(os.path.join(path, 'OBT_VSM_vectors.npy'), vectors)

    ref['vec'] = list(vectors)
    ref.to_csv(os.path.join(path, 'OBT_VSM.tsv'), sep='\t')


def process_entity_and_label_table(tablename):
    names_and_labels = parse_entity_and_label_table(tablename, return_id=False)
    names_vec = generate_fixed_length_vectors(names_and_labels.name)
    labels_vec = generate_averaging_vectors(names_and_labels.dict_name)
    np.save(os.path.join(path, '%s_names_vectors.npy' %tablename.split('_', -1)[-1]), names_vec)
    np.save(os.path.join(path, '%s_labels_vectors.npy' %tablename.split('_', -1)[-1]), labels_vec)


def generated_normalized_dict_and_labels():
    ref = parse_biotope_dict()
    vectors = generate_averaging_vectors(ref.name)
    vectors = reduce_dimension(vectors)
    np.save(os.path.join(path, 'OBT_VSM_norm.npy'), vectors)
    ref['vec'] = list(vectors)
    ref.to_csv(os.path.join(path, 'OBT_VSM_norm.tsv'), sep='\t')

    for tablename in ['entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv',
                      'entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv']:
        labels_id_and_labels = parse_entity_and_label_table(tablename, return_id=True)
        labels_vec = vectors[list(map(pd.Index(ref.id).get_loc, labels_id_and_labels.dict_id))]
        np.save(os.path.join(path, '%s_labels_vectors_norm.npy' %tablename.split('_', -1)[-1]),
                labels_vec)


if __name__=="__main__":
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
       '../input_data/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    vector_size = w2v_model.vector_size
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    path = os.path.join(os.getcwd(), '../input_data/vsm/')
    # process_biotope_dict(default_value=False)
    process_entity_and_label_table('entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv')
    process_entity_and_label_table('entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv')
    generated_normalized_dict_and_labels()
