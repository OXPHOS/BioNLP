import matplotlib.pyplot as plt
import seaborn as sns


def plot_model(model, title):
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.title(title)
    plt.savefig(title+'.png')


def plot_data(array):
    sns.set()
    fig, ax = plt.subplots(figsize=[10, 10])
    sns.heatmap(array)
