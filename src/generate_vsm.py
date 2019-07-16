import gensim
import numpy as np
from src.parse import *


def normalize():
    pass


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


def generate_fixed_length_vectors(namelist, phrase_size):
    vectors = np.zeros((len(namelist), phrase_size, vector_size))
    i = 0
    for e in namelist:
        name = ''.join(filter(whitelist.__contains__, e.replace('-', ' ')))
        if len(name.split()) > phrase_size:
            tmp = list()
            for w in name.split():
                if w in w2v_model.vocab:
                    tmp.append(w2v_model.get_vector(w))
                elif w.lower() in w2v_model.vocab:
                    tmp.append(w2v_model.get_vector(w.lower()))
            avg_factor = len(tmp) / phrase_size + 1
            for k in range(0, len(tmp), avg_factor):
                vectors[i][k/avg_factor] = np.mean(tmp[k:k+avg_factor])
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
    names, labels = parse_entity_and_label_table(tablename)
    names_vec = generate_fixed_length_vectors(names, 8)
    labels_vec = generate_averaging_vectors(labels)
    np.save(os.path.join(path, 'val_names_vectors.npy'), names_vec)
    np.save(os.path.join(path, 'val_labels_vectors.npy'), labels_vec)


if __name__=="__main__":
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
       '../input_data/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    vector_size = w2v_model.vector_size
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    path = os.path.join(os.getcwd(), '../input_data/vsm/')
    process_biotope_dict(default_value=False)
    process_entity_and_label_table('entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv')
