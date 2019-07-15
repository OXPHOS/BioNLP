import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity

import src.plot as plot
from src.parse import *


def perfect_match():
    pass


class Model:
    def __init__(self):
        print("Loading reference dictionary =================================>")
        self.ref = parse_biotope_dict()
        self.ref_vec = np.load(os.path.join(os.getcwd(), "../input_data/vsm/OBT_VSM_vectors.npy"), allow_pickle=True)

        print("Initiating =========================================>")
        self.model = Sequential()
        self.model.add(Conv1D(filters=500, kernel_size=2, padding='same', input_shape=(8, 200)))
        self.model.add(MaxPooling1D(8))
        self.model.add(Dense(200))
        self.model.compile(loss='cosine_proximity', optimizer=SGD())
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        print("Training =========================================>")
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
            eval_loss = self.model.evaluate(x_val, y_val)
            # print(x_val)
            # print(y_val)
            print("*Eval loss: ", eval_loss)
        return self.model

    def eval(self, x, y, lookup_table):
        print("Evaluating on: %s=====>" %lookup_table)
        names, labels = parse_entity_and_label_table(lookup_table)
        assert(len(labels)==len(y))

        y_pred = self.model.predict(x)
        y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[2]))
        idx_pred = np.argmax(cosine_similarity(y_pred, np.nan_to_num(self.ref_vec)), axis=1)
        output_form = pd.DataFrame([names.values, self.ref.name[idx_pred], labels.values]).T
        output_form.columns = ['input', 'pred', 'labels']
        output_form.sort_values('input', inplace=True)
        accuracy = (output_form[output_form.labels == output_form.pred].count())/len(output_form)

        output_form.to_csv('output_%s.tsv' %lookup_table, sep='\t')
        print(accuracy)
        return accuracy

    '''
    def cos_distance(y_true, y_pred):
        def l2_normalize(x, axis):
            norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
            return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())
        y_true = l2_normalize(y_true, axis=-1)
        y_pred = l2_normalize(y_pred, axis=-1)
        return -K.mean(y_true * y_pred, axis=-1)
    '''


def main():
    X = np.load(os.path.join(os.getcwd(), "../input_data/vsm/names_vectors.npy"), allow_pickle=True)
    Y = np.load(os.path.join(os.getcwd(), "../input_data/vsm/labels_vectors.npy"), allow_pickle=True)
    X_val = np.load(os.path.join(os.getcwd(), "../input_data/vsm/val_names_vectors.npy"), allow_pickle=True)
    Y_val = np.load(os.path.join(os.getcwd(), "../input_data/vsm/val_labels_vectors.npy"), allow_pickle=True)
    model = Model()
    model.train(X, Y, X_val, Y_val)
    model.eval(X, Y, 'entity_and_label_list_BioNLP-OST-2019_BB-norm_train.tsv')
    model.eval(X_val, Y_val, 'entity_and_label_list_BioNLP-OST-2019_BB-norm_dev.tsv')


if __name__=="__main__":
    main()
