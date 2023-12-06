import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from argparse import ArgumentParser
import numpy as np 
from argparse import ArgumentParser

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
 
    return starts_from_zero / value_range

parser = ArgumentParser('feature visualization')
parser.add_argument('path', type=str, help='path to feature directory')

if __name__ == '__main__':
    args = parser.parse_args()
    file_list = os.listdir(args.path)
    data_list = ['ff', 'celeb', 'dfdc', 'vfhq', 'dff']

    data_label = []
    features_list = []
    labels_list = []

    for data in data_list:
        data_features = np.load(f'{args.path}/{data}_features.npy')
        data_labels = np.load(f'{args.path}/{data}_labels.npy')

        data_label.extend([data] * len(data_labels))
        features_list.append(data_features)
        labels_list.append(data_labels)
    
    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)

    datasets = {"ff": 0 , "celeb": 1 , "dfdc": 2, "vfhq": 3, "dff": 4, "real": 5}
    colors = plt.cm.rainbow(np.linspace(0, 1.0, 6))

    tsne = TSNE(n_components=2, random_state=1, perplexity=4)
    X_tsne = tsne.fit_transform(features)

    for data in data_list:
        for label in range(2):
            mask = [labels[i] == label and data_label[i] == data for i in range(len(labels))]
            tx = X_tsne[mask, 0]
            ty = X_tsne[mask, 1]

            color_idx = "real" if label == 0 else data
            label_idx = "real" if label == 0 and data == "ff" else "" if label == 0 else data
            plt.scatter(tx, ty, color=colors[datasets[color_idx]], label=label_idx, s=9, alpha=0.7)

    plt.legend()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    args.path = args.path[:-1] if args.path[-1] == '/' else args.path
    plt.savefig(f"{args.path}.png")
    plt.show()