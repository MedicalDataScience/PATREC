
import os
import sys
import glob as glob
import numpy as np
import string

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

DIRPROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/';

from utils.DatasetOptions import DatasetOptions
import helpers.icd10_chapters as icd10_chapters

def getDiagCodesIndices(diag_group_names):
    main_groups = icd10_chapters.getMainGroups();
    indices_codes = [];
    for k, name in enumerate(main_groups):
        subgroups = icd10_chapters.getSubgroups(name);
        for l, sub in enumerate(subgroups):
            codes = icd10_chapters.getCodesSubgroup(name, sub);
            for code in codes:
                indices_codes.append(diag_group_names.index(code));
    return indices_codes;



if __name__ == '__main__':

    dir_model = sys.argv[1];
    threshold_epoch = 0;
    if len(sys.argv) > 2:
        threshold_epoch = int(sys.argv[2]);
    dict_data_train = {
        'dir_data':         DIRPROJECT + 'data/',
        'data_prefix':      'nz',
        'dataset':          '20122016',
        'encoding':         'embedding',
        'newfeatures':      None,
        'featurereduction': {'method': 'FUSION'},
        'grouping':         'verylightgrouping'
    }
    dataset_options_train = DatasetOptions(dict_data_train);

    diag_group_names = dataset_options_train.getDiagGroupNames();
    indices_diag_codes = getDiagCodesIndices(diag_group_names);
    main_groups = icd10_chapters.getMainGroups();

    num_colors = len(main_groups);
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors));

    num_diags = len(indices_diag_codes);


    filenames_encodings = glob.glob(dir_model + 'basic_encodings_*');
    var_encodings = [];
    for l,f in enumerate(sorted(filenames_encodings)):
        print(f)
        epoch = int(f.split('/')[-1].split('.')[0].split('_')[-1]);
        print('epoch: ' + str(epoch))
        basic_encodings = np.load(f);
        basic_encodings = basic_encodings[indices_diag_codes,:]

        encoding_dims = basic_encodings.shape[1];
        var_encodings.append(np.var(basic_encodings, axis=0));
        var_encoding_dims = np.array(var_encodings);
        mean_var_encoding_dims = np.mean(var_encoding_dims, axis=1);

        # print('var: ' + str(np.var(basic_encodings, axis=0)));
        print('mean var: ' + str(mean_var_encoding_dims))

        colors_var = plt.cm.rainbow(np.linspace(0, 1, encoding_dims+1));
        filename_plot = dir_model + 'plot_var_encoding_dims.png';
        plt.figure(figsize=(20,15))
        for k in range(0, encoding_dims):
            plt.plot(var_encoding_dims[:,k], c=colors_var[k], label= 'dim_' + str(k))
        plt.plot(mean_var_encoding_dims, linewidth=3, c=colors_var[-1], label='mean var')
        plt.grid(True);
        plt.title('variance encoding dims after epoch ' + str(epoch))
        plt.legend(loc='upper left')
        plt.draw()
        plt.savefig(filename_plot, format='png');
        plt.close();


        if epoch >= threshold_epoch:
            tsne = TSNE(n_components=2);
            pca = PCA(n_components=2)

            weights_2d_tsne = tsne.fit_transform(basic_encodings);
            weights_2d_pca = pca.fit_transform(basic_encodings);

            filename_plot = dir_model + 'plot_pca_encoding_2d_epoch_' + str(epoch) + '.png';
            plt.figure(figsize=(17, 17));
            for k in range(0, num_colors):
                c = colors[k]
                num_points_group = 0;
                for j,sub in enumerate(icd10_chapters.getSubgroups(main_groups[k])):
                    num_points_group += len(icd10_chapters.getCodesSubgroup(main_groups[k], sub));
                plt.scatter(weights_2d_pca[k * num_points_group:(k * num_points_group + num_points_group), 0],
                            weights_2d_pca[k * num_points_group:(k * num_points_group + num_points_group), 1],
                            label=main_groups[k], alpha=0.5, s=100,
                            c=c); #, marker='$' + string.ascii_uppercase[k] + '$'
            plt.legend()
            plt.title('pca: epoch ' + str(epoch))
            plt.draw();
            plt.savefig(filename_plot, format='png');
            plt.close();

            filename_plot = dir_model + 'plot_tsne_encoding_2d_epoch_' + str(epoch) + '.png';
            plt.figure(figsize=(17,17));
            for k in range(0, num_colors):
                c = colors[k]
                num_points_group = 0;
                for j,sub in enumerate(icd10_chapters.getSubgroups(main_groups[k])):
                    num_points_group += len(icd10_chapters.getCodesSubgroup(main_groups[k], sub));
                plt.scatter(weights_2d_tsne[k * num_points_group:(k * num_points_group + num_points_group), 0],
                            weights_2d_tsne[k * num_points_group:(k * num_points_group + num_points_group), 1],
                            label=main_groups[k], alpha=0.5, s=100,
                            c=c); # , marker='$' + string.ascii_uppercase[k] + '$'
            plt.legend()
            plt.title('t-sne: epoch ' + str(epoch))
            plt.draw()
            plt.savefig(filename_plot, format='png');
            plt.close();




