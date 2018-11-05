
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import helpers.helpers as helpers

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

if __name__ == '__main__':
    dirNN = '/Users/towyku74/UniBas/sciCore/projects/PATREC/trained_models/dev/nz_20012011_reduction_FUSION_embedding_verylightgrouping_20_10_10_dropout_0.5_learningrate_0.05_batchnorm_True_batchsize_640/'
    filename_weights_main_diag = dirNN + 'weights_embedding_main_diag.npy'
    weights = np.load(filename_weights_main_diag);

    num_diags = 2600;
    num_categories = 26;
    cnt=0
    labels = np.zeros(num_diags)
    labels_maincat = [];
    for k in range(0, 26):
        for l in range(0, 100):
            labels[cnt] = k;
            labels_maincat.append(alphabet[k]);
            cnt+=1

    labels_finegrained = helpers.getDKverylightGrouping();
    filename_labels = dirNN + 'labels_cat.tsv';
    file_labels = open(filename_labels, 'w');
    file_labels.write('main_category' + '\t' + 'category' + '\n')
    for k in range(0, len(labels_maincat)):
        file_labels.write(labels_maincat[k] + '\t' + labels_finegrained[k] + '\n');
    file_labels.close();

    colors = plt.cm.rainbow(np.linspace(0, 1, num_categories));

    pca = PCA(n_components=2)
    weights_2d_pca = pca.fit_transform(weights);

    tsne = TSNE(n_components=2);
    weights_2d_tsne = tsne.fit_transform(weights);

    lda = LinearDiscriminantAnalysis(n_components=2)
    weights_2d_lda = lda.fit_transform(weights, labels);


    plt.figure();
    for k in range(0, 26):
        for l in range(0, 100):
            c = colors[k]
            plt.scatter(weights_2d_pca[k*100+l,0], weights_2d_pca[k*100+l,1], alpha=0.2, s=50, c=c);
    plt.title('pca')
    plt.draw()

    plt.figure();
    for k in range(0, 26):
        for l in range(0, 100):
            c = colors[k]
            plt.scatter(weights_2d_tsne[k*100+l,0], weights_2d_tsne[k*100+l,1], alpha=0.2, s=50, c=c);
    #plt.scatter(weights_main_diag_2d_tsne[:, 0], weights_main_diag_2d_tsne[:, 1], alpha=0.2, s=50)
    plt.title('t-sne')
    plt.draw()

    plt.figure();
    for k in range(0, 26):
        for l in range(0, 100):
            c = colors[k]
            plt.scatter(weights_2d_lda[k * 100 + l, 0], weights_2d_lda[k * 100 + l, 1], alpha=0.2, s=50, c=c);
    # plt.scatter(weights_main_diag_2d_tsne[:, 0], weights_main_diag_2d_tsne[:, 1], alpha=0.2, s=50)
    plt.title('lda')
    plt.draw()

    plt.show()