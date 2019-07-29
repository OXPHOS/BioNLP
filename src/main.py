"""
Train a CNN model to predict normalized habitats or phenotypes

@ Date: 2019-07-15
@ Author: OXPHOS
"""

from collections import Counter
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity

import src.plot as plot
from src.parse import *
from src import entity_embedding_size, context_embedding_size, vector_len


def perfect_match(data):
    """
    Predict normalized name with pattern matching
    :param data: DataFrame with input entity name.
                 DataFrame has columns: ['text_id', 'entity_id', 'input', 'input_vecs', 'labels', 'labels_vecs',
                                         'pred', 'pred_obt', 'method']
    :return: data: updated DataFrame with matched ['pred', 'pred_obt', 'method'] filled
             mask: unmatched index list
    """
    correct = 0 # For evaluation purpose only

    # char-to-char matching
    for i in data.index:
        if (ref.name == data.input[i]).any():
            matched_idx = ref.name[ref.name == data.input[i]].index[0]
            data.pred[i] = ref.name[matched_idx]
            data.pred_obt[i] = ref.id[matched_idx]
            data.method[i] = 'Exact match'
            if data.pred[i] == data.labels[i]:
                correct += 1

    # match all kind of cheeses
    for i in data.index:
            s = data.input[i].split(' ', -1)
            if (not data.method[i]) and 'cheese' in s[-1].lower() \
                    and (ref.name == s[0]).any():
                matched_idx = ref.name[ref.name == s[0]].index[0]
                data.pred[i] = ref.name[matched_idx]
                data.method[i] = 'Cheeses'
                data.pred_obt[i] = ref.id[matched_idx]
                if data.pred[i] == data.labels[i]:
                    correct += 1

    # get unmatched entity index
    mask = data.index[data.method.isna()]

    print("Total matched:", len(data)-len(mask), " Correct matched:", correct,
          " Accuracy:", correct/(len(data)-len(mask)))
    return data, mask


class Model:
    """
    CNN prediction model
    """

    def __init__(self, arg, fname=None):
        """
        :param arg: the param to be searched. Currently searching for different filter number 
        :param fname: the file name to save the trained models
        """
        print("Initiating =========================================>")
        self.model = Sequential()
        self.model.add(Conv1D(filters=arg, kernel_size=4, padding='same',
                              input_shape=(entity_embedding_size, vector_len)))
        self.model.add(MaxPooling1D(entity_embedding_size))
        # NOTE: deeper CNN didn't behave well, and was extremely slow
        # self.model.add(Dropout(0.2))
        # self.model.add(Conv1D(filters=arg, kernel_size=4, padding='same'))
        # self.model.add(MaxPooling1D(entity_embedding_size//2))
        self.model.add(Dense(139))
        self.model.compile(loss='cosine_proximity', optimizer=SGD())

        # To save model
        self.fname=fname
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train a shallow CNN model with cosine_proximity loss function 

        :return: loss
        """
        print("Training =========================================>")
        # print(x_train.shape, y_train.shape)
        y_train = np.expand_dims(y_train, axis=1)
        history = self.model.fit(x_train, y_train,
                                 validation_split=0.2,
                                 epochs=50, batch_size=2,
                                 callbacks=[EarlyStopping()],
                                 verbose=0)
        # plot.plot_model(history, "Training loss")
        print("*Training loss: ", history.history['loss'][-1])
        print("*Cross-val loss: ", history.history['val_loss'][-1])

        if isinstance(x_val, np.ndarray) and isinstance(y_val, np.ndarray):
            print("Validating =========================================>")
            y_val = np.expand_dims(y_val, axis=1)
            eval_loss = self.model.evaluate(x_val, y_val, verbose=0)
            print("*Eval loss: ", eval_loss)

        if self.fname:
            self.model.save("../tmp_output/"+self.fname)
        return history.history['loss'][-1], history.history['val_loss'][-1], eval_loss

    @staticmethod
    def eval(models, ref_vec, x, data, mask, filename, prediction=False):
        """
        Evaluate models
        
        :param models: a single model or a list of model to be used for evaluation
        :param ref_vec: vector space of reference words
        :param x: input numpy array
        :param data: DataFrame with input entity name.
                     DataFrame has columns: ['text_id', 'entity_id', 'input', 'input_vecs', 'labels', 'labels_vecs',
                                         'pred', 'pred_obt', 'method']
        :param mask: the entity indexes that are predicted with CNN
        :param filename: output filename
        :param prediction: whether the evaluation is on internal test data set (has labels) 
               or real test data set (no label)
        :return: if prediciton == False: return CNN accuracy (leaving perfect_match accuracy out)
        """
        print("Evaluating ============================>" )
        if isinstance(models, list):
            pred_result = pd.DataFrame(index=range(len(x)))

            # Get prediction result from each single model
            for i, m in enumerate(models):
                y_pred = m.model.predict(x[mask])
                y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[2]))

                # Find matched reference word by looking up the entity with the smallest cosine distance
                idx_pred = np.argmax(cosine_similarity(y_pred, np.nan_to_num(ref_vec)), axis=1)
                pred_result[i] = ref.name[idx_pred].reset_index(drop=True)
            data['score'] = None  # For debugging

            # Get voting result for each entity
            for i, m in enumerate(mask):
                counter = Counter(pred_result.iloc[i].tolist())
                data.pred[m] = max(counter, key=lambda k: counter[k])
                data.pred_obt[m] = ref.id[ref.name==data.pred[m]].item()
                data.score[m] = counter[data.pred[m]]  # Get voting information
        else:
            y_pred = models.model.predict(x)
            y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[2]))

            # Find matched reference word by looking up the entity with the smallest cosine distance
            idx_pred = np.argmax(cosine_similarity(y_pred, np.nan_to_num(ref_vec)), axis=1)

            data.pred[mask] = ref.name[idx_pred[mask]]
            data.pred_obt[mask] = ref.id[idx_pred[mask]]

        data.method[mask] = 'CNN'

        result = data.drop(columns=['input_vecs', 'labels_vecs'])

        if prediction:
            result.to_csv("../tmp_output/"+filename, sep='\t')
            result = result.drop(columns=['labels', 'method', 'score'])
            result.columns=['text_id', 'entity_id', 'word', 'matched_name', 'matched_id']
            result.to_csv('OntoBiotope_result.tsv', sep='\t', index=False)
        else:
            result = result.sort_values('input')
            result.to_csv("../tmp_output/"+filename, sep='\t')

            # Calculate accuracy
            accuracy_cnn = sum(result[result.method=='CNN'].labels==result[result.method=='CNN'].pred) \
                           / len(result[result.method=='CNN'])
            accuracy_total = sum(result.labels==result.pred) / len(result)
            print("CNN accuracy:", accuracy_cnn, " Total accuracy:", accuracy_total)
            return accuracy_cnn

    '''
    # Concept of cosine proximity loss function
    def cos_distance(y_true, y_pred):
        def l2_normalize(x, axis):
            norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
            return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())
        y_true = l2_normalize(y_true, axis=-1)
        y_pred = l2_normalize(y_pred, axis=-1)
        return -K.mean(y_true * y_pred, axis=-1)
    '''


def generate_data(x, y, tablename):
    """
    Merge word vector numpy array with DataFrame with richer information
    
    :param x: entity vector np.ndarray
    :param y: label vector np.ndarray
    :param tablename: the information table to incorporate the np arrays
    :return: updated tatble
    """
    names_and_labels = pd.read_csv(os.path.join(os.getcwd(), '../input_data/vsm/%s' % tablename), sep='\t')
    assert (len(names_and_labels.dict_name) == len(y))
    result = pd.DataFrame([names_and_labels.text_id.values, names_and_labels.entity_id.values,
                           names_and_labels.name.values, x, names_and_labels.dict_name.values, y]).T
    result.columns = ['text_id', 'entity_id', 'input', 'input_vecs', 'labels', 'labels_vecs']
    result['pred'] = None
    result['pred_obt'] = None
    result['method'] = None
    return result


def single_model():
    """
    Make predictions with a single CNN model with pre-defined training and validation data sets.
    Search for best param combinations
    """
    X = np.load(os.path.join(os.getcwd(), "../input_data/vsm/train.tsv_names_vectors.npy"),
                allow_pickle=True)
    Y = np.load(os.path.join(os.getcwd(), "../input_data/vsm/train.tsv_labels_vectors_norm.npy"),
                allow_pickle=True)
    X_val = np.load(os.path.join(os.getcwd(), "../input_data/vsm/dev.tsv_names_vectors.npy"),
                    allow_pickle=True)
    Y_val = np.load(os.path.join(os.getcwd(), "../input_data/vsm/dev.tsv_labels_vectors_norm.npy"),
                    allow_pickle=True)
    ref_vec = np.load(os.path.join(os.getcwd(), "../input_data/vsm/OBT_VSM_norm.npy"),
                      allow_pickle=True)
    train_data = generate_data(X, Y, 'train.tsv_names_and_labels.tsv')
    val_data = generate_data(X_val, Y_val, 'dev.tsv_names_and_labels.tsv')

    train_match_result, train_match_mask = perfect_match(train_data)
    val_match_result, val_match_mask = perfect_match(val_data)

    # params search
    # searching result dataframe header
    params = pd.DataFrame(columns=['value', 'Training_loss', 'Cross_val_loss', 'Validation_loss',
                                   'Training_accuracy', 'Validation_accuracy'])
    for i in range(1000, 10001, 1000): # result: 4700 is the best filter number
        print("----", i, "----")
        model = Model(i)
        loss = model.train(X[train_match_mask], Y[train_match_mask], X_val[val_match_mask], Y_val[val_match_mask])
        acc_train = Model.eval(model, ref_vec, X, train_match_result, train_match_mask, 'pred_result_train.tsv')
        acc_val = Model.eval(model, ref_vec, X_val, val_match_result, val_match_mask, 'pred_result_dev.tsv')
        params.loc[i] = [i]+list(loss)+[acc_train, acc_val]
    # params.to_csv('../tmp_output/Param_Search_Filter_50-5000_8x200_pac0.95.csv')
    # print(params.loc[params.Validation_accuracy.idxmax()])


def voting_model(prediction=False):
    """
    Make predictions with a combination of 5 CNN models, each model is trained with randomly split training and val set.

    :param prediction: whether using internal test dataset (with labels) or real test dataset (no labels)
    """
    X = np.load(os.path.join(os.getcwd(), "../input_data/vsm/5fold_train_names_vectors.npy"),
                allow_pickle=True)
    Y = np.load(os.path.join(os.getcwd(), "../input_data/vsm/5fold_train_labels_vectors_norm.npy"),
                allow_pickle=True)
    X_test = np.load(os.path.join(os.getcwd(), "../input_data/vsm/5fold_test_names_vectors.npy"),
                     allow_pickle=True)
    if prediction:
        Y_test = np.empty(X_test.shape, dtype=np.float32)
    else:
        Y_test = np.load(os.path.join(os.getcwd(), "../input_data/vsm/5fold_test_labels_vectors_norm.npy"),
                         allow_pickle=True)

    ref_vec = np.load(os.path.join(os.getcwd(), "../input_data/vsm/OBT_VSM_norm.npy"),
                      allow_pickle=True)

    train_and_val_data = generate_data(X, Y, '5fold_train.tsv')
    test_data = generate_data(X_test, Y_test, '5fold_test.tsv')

    # Perfect matching
    train_and_val_match_result, train_and_val_match_mask = perfect_match(train_and_val_data)
    test_match_result, test_match_mask = perfect_match(test_data)
    print(X.shape, X_test.shape)

    models = []
    tmp_shuffle = train_and_val_match_mask.tolist()
    for j in range(5):
        np.random.seed(j)
        np.random.shuffle(tmp_shuffle)
        train_match_mask = tmp_shuffle[:math.ceil(len(tmp_shuffle)*0.8)]
        val_match_mask = tmp_shuffle[math.ceil(len(tmp_shuffle)*0.8):]
        models.append(Model(5000, "Model_%s.h5" %j))
        loss = models[j].train(X[train_match_mask], Y[train_match_mask], X[val_match_mask], Y[val_match_mask])

    Model.eval(models, ref_vec, X, train_and_val_match_result, train_and_val_match_mask,
               '5fold_pred_result_train_and_val.tsv')
    Model.eval(models, ref_vec, X_test[test_data.index], test_match_result, test_match_mask,
               '5fold_pred_result_test.tsv', prediction=prediction)


if __name__=="__main__":
    # np.random.seed(42)
    ref = parse_biotope_dict()

    # Train one model
    # single_model()

    # Train a voting model of 5 single models
    voting_model(prediction=False)
