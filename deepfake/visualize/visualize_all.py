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
parser.add_argument('--m', action='store_true', help='visualize by manipulation type in ff. visualize by datasets in default')

if __name__ == '__main__':
    args = parser.parse_args()
    file_list = os.listdir(args.path)
    data_list = ['ff'] if args.m else ['ff', 'celeb', 'dfdc', 'vfhq', 'dff']

    data_label = []
    features_list = []
    labels_list = []

    for data in data_list:
        data_features = np.load(f'{args.path}/{data}_features.npy')

        label_type = 'mtypes' if args.m else 'labels'
        data_labels = np.load(f'{args.path}/{data}_{label_type}.npy')

        data_label.extend([data] * len(data_labels))
        features_list.append(data_features)
        labels_list.append(data_labels)
    
    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)
    
    datasets = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'] if args.m else \
                {"ff": 0 , "celeb": 1 , "dfdc": 2, "vfhq": 3, "dff": 4, "real": 5}
    colors = plt.cm.rainbow(np.linspace(0, 1.0, len(datasets)))

    tsne = TSNE(n_components=2, random_state=1, perplexity=300, learning_rate=4000)
    X_tsne = tsne.fit_transform(features)

    for data in data_list:
        for label in range(5 if args.m else 2):
            mask = [labels[i] == label and data_label[i] == data for i in range(len(labels))]
            tx = X_tsne[mask, 0]
            ty = X_tsne[mask, 1]

            if args.m:
                color_val = colors[label]
                label_val = datasets[label]
            else:
                color_idx = "real" if label == 0 else data
                label_idx = "real" if label == 0 and data == "ff" else "" if label == 0 else data
                color_val = colors[datasets[color_idx]]
                label_val = label_idx

            plt.scatter(tx, ty, color=color_val, label=label_val, s=16 if args.m else 9, alpha=0.7)

    plt.legend()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    vis_type = 'mtype' if args.m else 'dataset'

    args.path = args.path[:-1] if args.path[-1] == '/' else args.path
    plt.savefig(f"{args.path}_{vis_type}.png", bbox_inches='tight')
    plt.show()