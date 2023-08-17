import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from argparse import ArgumentParser
import numpy as np 
from sklearn.neighbors import KernelDensity

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('trained_data', type=str, help='trained data')
parser.add_argument('-p', type=float, default=30.0, help='perpexity for t-sne')
parser.add_argument('-i', type=int, default=1000, help='iterations for t-sne')
parser.add_argument('-s', type=int, default=30, help='number of samples for each class')

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def balanced_sample(features, labels, class_count):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    min_class_count = class_count

    sampled_features = []
    sampled_labels = []

    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        random_indices = np.random.choice(class_indices, min_class_count, replace=False)
        sampled_features.extend(features[random_indices])
        sampled_labels.extend(labels[random_indices])

    sampled_features = np.array(sampled_features)
    sampled_labels = np.array(sampled_labels)

    # Shuffle the data to ensure randomness
    random_indices = np.random.permutation(len(sampled_features))
    sampled_features = sampled_features[random_indices]
    sampled_labels = sampled_labels[random_indices]

    return sampled_features, sampled_labels

if __name__ == '__main__':
    args = parser.parse_args()
    trained_data = args.trained_data
    # perplexity = args.p
    # iterations = args.i
    sample_num = 100

    file_list = os.listdir('features/')
    # file_list.remove('dfdc_dfdc_features.npy')
    # file_list.remove('dfdc_dfdc_labels.npy')
    data_list = list(set(item.split('_')[1] for item in file_list))
    data_list = sorted(data_list, key=lambda x: x != 'dfdc')

    data_label = []
    features_list = []
    labels_list = []

    for data in data_list:
        # if (data == 'dfdc'):
        #     data = 'dfdc_test'
            
        data_features = np.load(f'features/{trained_data}_{data}_features.npy')
        data_labels = np.load(f'features/{trained_data}_{data}_labels.npy')

        print(len(data_labels))

        sampled_features, sampled_labels = balanced_sample(data_features, data_labels, sample_num)
        

        data_label.extend([data] * len(sampled_labels))
        features_list.append(sampled_features)
        labels_list.append(sampled_labels)
    
    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)

    print(f"total number of obersvations: {len(features)}")

    perplexities = range(5, 20, 5)
    lr_list = range(3000, 6000, 1000)

    classes = ["o", "x"]
    datasets = {"dfdc": 0 , "ff": 1 , "celeb": 2}
    colors = plt.cm.plasma(np.linspace(0, 0.8, 3))

    fig, axes = plt.subplots(len(perplexities), len(lr_list), figsize=(12, 12))

    for p_idx, perplexity in enumerate(perplexities):
        for l_idx, lr in enumerate(lr_list):
            print(f"processing p{perplexity} i{lr}...")
            tsne = TSNE(n_components=2, random_state=1, perplexity=perplexity, n_iter=1000, learning_rate=lr)
            X_tsne = tsne.fit_transform(features)

            for data in data_list:
                for label in range(2):
                    mask = [labels[i] == label and data_label[i] == data for i in range(len(labels))]
                    tx = scale_to_01_range(X_tsne[mask, 0])
                    ty = scale_to_01_range(X_tsne[mask, 1])
                    axes[p_idx, l_idx].scatter(tx, ty, 
                                               color=colors[datasets[data]], label=f'{data}', marker=classes[label], alpha=0.7)

            if l_idx == 0:
                axes[p_idx, l_idx].set_ylabel(f"Perplexity: {perplexity}", rotation=90, labelpad=15)
            
            if p_idx == 0:    
                axes[p_idx, l_idx].set_title(f"Learning rate: {lr}")


    legend_labels = [[f'{data}_REAL', f'{data}_FAKE'] for data in data_list]
    annot = np.concatenate(legend_labels)
    fig.legend(annot, loc='lower right', ncol=len(data_list), bbox_to_anchor=(1, 0))

    plt.tight_layout()
    plt.savefig(f"visualize/tsne_{trained_data}_combined.png")
    plt.show()