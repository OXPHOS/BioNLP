import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity

import src.plot as plot
from src.parse import *
from src import entity_embedding_size, context_embedding_size, vector_len


def perfect_match(data):
    correct = 0
    mask = [True]*len(data)
    for i in range(len(data.input)):
        if (ref.name == data.input[i]).any():
            matched_idx = ref.name[ref.name == data.input[i]].index[0]
            data.pred[i] = ref.name[matched_idx]
            data.method[i] = 'Exact match'
            if data.pred[i] == data.labels[i]:
                correct += 1
            mask[i] = False

    # pattern =".*\sCheese"
    # regxObj = re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for i in range(len(data.input)):
        if mask[i]:
            s = data.input[i].split(' ', -1)
            if 'cheese' in s[-1].lower() and (ref.name == s[0]).any():
                matched_idx = ref.name[ref.name == s[0]].index[0]
                data.pred[i] = ref.name[matched_idx]
                data.method[i] = 'Cheeses'
                if data.pred[i] == data.labels[i]:
                    correct += 1
                    mask[i] = False
    print("Total matched:", len(data)-sum(mask), " Correct matched:", correct,
          " Accuracy:", correct/(len(data)-sum(mask)))
    return data, mask


class Model:
    def __init__(self, ref_vec, arg):
        print("Loading reference dictionary =================================>")
        self.ref_vec = ref_vec

        print("Initiating =========================================>")
        self.model = Sequential()
        self.model.add(Conv1D(filters=arg, kernel_size=5, padding='same',
                              input_shape=(context_embedding_size, vector_len)))
        self.model.add(MaxPooling1D(5))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters=arg, kernel_size=5, padding='same'))
        self.model.add(MaxPooling1D(context_embedding_size//5))
        self.model.add(Dense(139))
        self.model.compile(loss='cosine_proximity', optimizer=SGD())
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        print("Training =========================================>")
        # print(x_train.shape, y_train.shape)
        y_train = np.expand_dims(y_train, axis=1)
        history = self.model.fit(x_train, y_train,
                                 validation_split=0.2,
                                 epochs=50, batch_size=2,
                                 callbacks=[EarlyStopping()],
                                 verbose=0)
        plot.plot_model(history, "Training loss")
        print("*Training loss: ", history.history['loss'][-1])
        print("*Cross-val loss: ", history.history['val_loss'][-1])

        if isinstance(x_val, np.ndarray) and isinstance(y_val, np.ndarray):
            print("Validating =========================================>")
            y_val = np.expand_dims(y_val, axis=1)
            eval_loss = self.model.evaluate(x_val, y_val, verbose=0)
            print("*Eval loss: ", eval_loss)
        return history.history['loss'][-1], history.history['val_loss'][-1], eval_loss

    def eval(self, x, data, mask, filename):
        print("Evaluating ============================>" )

        y_pred = self.model.predict(x)
        y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[2]))
        idx_pred = np.argmax(cosine_similarity(y_pred, np.nan_to_num(self.ref_vec)), axis=1)
        data.pred[mask] = ref.name[idx_pred[mask]]
        data.method[mask] = 'CNN'

        result = data[['input', 'labels', 'pred', 'method']].sort_values('input')
        # result.to_csv(filename, sep='\t')

        accuracy_cnn = sum(result[result.method=='CNN'].labels==result[result.method=='CNN'].pred) \
                       / len(result[result.method=='CNN'])
        accuracy_total = sum(result.labels==result.pred) / len(result)
        print("CNN accuracy:", accuracy_cnn, " Total accuracy:", accuracy_total)
        return accuracy_cnn



    '''
    def cos_distance(y_true, y_pred):
        def l2_normalize(x, axis):
            norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
            return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())
        y_true = l2_normalize(y_true, axis=-1)
        y_pred = l2_normalize(y_pred, axis=-1)
        return -K.mean(y_true * y_pred, axis=-1)
    '''


def generate_data(x, y, lookup_table):
    names_and_labels = parse_entity_and_label_table(lookup_table)
    assert (len(names_and_labels.dict_name) == len(y))
    result = pd.DataFrame([names_and_labels.name.values, x, names_and_labels.dict_name.values, y]).T
    result.columns = ['input', 'input_vecs', 'labels', 'labels_vecs']
    result['pred'] = None
    result['method'] = None
    return result


def main():
    X = np.load(os.path.join(os.getcwd(), "../input_data/vsm/train.tsv_names_vectors_with_context.npy"),
                allow_pickle=True)
    Y = np.load(os.path.join(os.getcwd(), "../input_data/vsm/train.tsv_labels_vectors_norm.npy"),
                allow_pickle=True)
    X_val = np.load(os.path.join(os.getcwd(), "../input_data/vsm/dev.tsv_names_vectors_with_context.npy"),
                    allow_pickle=True)
    Y_val = np.load(os.path.join(os.getcwd(), "../input_data/vsm/dev.tsv_labels_vectors_norm.npy"),
                    allow_pickle=True)
    ref_vec = np.load(os.path.join(os.getcwd(), "../input_data/vsm/OBT_VSM_norm.npy"),
                      allow_pickle=True)
    train_data = generate_data(X, Y, 'entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv')
    val_data = generate_data(X_val, Y_val, 'entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv')

    train_match_result, train_match_mask = perfect_match(train_data)
    val_match_result, val_match_mask = perfect_match(val_data)

    params = pd.DataFrame(columns=['value', 'Training_loss', 'Cross_val_loss', 'Validation_loss',
                                   'Training_accuracy', 'Validation_accuracy'])
    for i in range(1000, 10001, 1000):
        print("----", i, "----")
        model = Model(ref_vec, i)
        loss = model.train(X[train_match_mask], Y[train_match_mask], X_val[val_match_mask], Y_val[val_match_mask])
        acc_train = model.eval(X, train_match_result, train_match_mask, 'pred_result_train.tsv')
        acc_val = model.eval(X_val, val_match_result, val_match_mask, 'pred_result_dev.tsv')
        params.loc[i] = [i]+list(loss)+[acc_train, acc_val]
    # params.to_csv('Param_Search_Filter_50-5000_8x200_pac0.95.csv')
    # print(params.loc[params.Validation_accuracy.idxmax()])


if __name__=="__main__":
    np.random.seed(42)
    ref = parse_biotope_dict()
    main()

