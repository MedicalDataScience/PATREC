import sys
import os

import pandas as pd


class DataPreparer:

    def __init__(self, dir_data, dataset, filename_options_in, filename_options_out, grouping, chunksize=10000):
        self.dir_data = dir_data;
        self.dataset = dataset
        self.filename_options_in = filename_options_in;
        self.filename_options_out = filename_options_out;
        self.grouping = grouping;
        self.chunksize = chunksize
        return;


    def __binaryToString(self, chunk):
        chunk = chunk.drop('Fall', axis=1);
        cols = chunk.columns
        bt = chunk.apply(lambda x: x > 0)
        bt = bt.apply(lambda x: list(cols[x.values]), axis=1)
        bt = bt.apply(' '.join)
        return bt;


    def __fuseSplittedColumnsString(self, df, subgroups):
        print('df_base.shape: ' + str(df.shape))
        for sub in subgroups:
            strFilenameInSubGroup = self.dataset + '_' + sub + '_' + self.grouping;
            filename_data_subgroup = self.dir_data + 'data_' + strFilenameInSubGroup + '.csv';
            subgroup_df = pd.Series([]);
            subgroup_data_reader = pd.read_csv(filename_data_subgroup, chunksize=self.chunksize);
            for k, chunk in enumerate(subgroup_data_reader):
                subgroup_df = subgroup_df.append(self.__binaryToString(chunk));

            df[sub] = subgroup_df.values;
        return df;


    def __fuseSplittedColumnsCategories(self, df, subgroups):
        print('df.shape: ' + str(df.shape))
        for sub in subgroups:
            strFilenameInSubGroup = self.dataset + '_' + sub + '_' + self.grouping;
            filename_data_subgroup = self.dir_data + 'data_' + strFilenameInSubGroup + '.csv';
            print(filename_data_subgroup)
            subgroup_df = pd.read_csv(filename_data_subgroup);
            subgroup_df = subgroup_df.drop('Fall', axis=1);
            group_names = subgroup_df.columns;
            for groupname in group_names:
                subgroup_df = subgroup_df.rename(columns={groupname: sub + '_' + groupname});
            print(list(subgroup_df.columns))
            df = pd.concat([df,subgroup_df], axis=1)
            print('df.shape: ' + str(df.shape))
        return df;


    def __getFilenameOptionStr(self):
        strFilenameIn = self.dataset + '_REST_' + self.filename_options_in;
        strFilenameOut = self.dataset + '_' + self.filename_options_out;
        return [strFilenameIn, strFilenameOut]


    def __fuseSubgroupsCategories(self, df, subgroups):
        df_fused = self.__fuseSplittedColumnsCategories(df, subgroups);
        return df_fused;


    def __fuseSubgroupsString(self, df, subgroups):
        df_fused = self.__fuseSplittedColumnsString(df, subgroups);
        return df_fused;


    def fuseSubgroups(self, subgroups, encoding, featureset = None, options_featureset=None):

        [filename_in_str, filename_out_str] = self.__getFilenameOptionStr()
        filename_data_out = self.dir_data + 'data_' + filename_out_str + '.csv';
        filename_data_in = self.dir_data + 'data_' + filename_in_str + '.csv';
        df_base = pd.read_csv(filename_data_in);

        if encoding == 'embedding':
            df_finish = self.__fuseSubgroupsString(df_base, subgroups);
        elif encoding == 'categorical' or encoding == 'binary':
            if featureset == 'reduction':
                if options_featureset['reduction_method'] == 'NOADMIN':
                    df_finish = self.__fuseSubgroupsCategories(df_base, subgroups);
                elif options_featureset['reduction_method'] == 'ONLYADMIN':
                    df_finish = df_base;
                else:
                    print('unknown input file...reduction method is weird...')
                    sys.exit();
            else:
                df_finish = self.__fuseSubgroupsCategories(df_base, subgroups);
        else:
            print('encoding scheme is not known...maybe not yet implemented..')
            sys.exit();
        df_finish.to_csv(filename_data_out, mode='w', index=False, line_terminator='\n');
        return;