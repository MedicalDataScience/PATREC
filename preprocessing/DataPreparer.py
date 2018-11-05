import sys
import os


import pandas as pd

import helpers.helpers as helpers

class DataPreparer:

    def __init__(self, options_dataset):
        self.options = options_dataset;
        return;


    def __binaryToString(self, chunk):
        cols = chunk.columns
        bt = chunk.apply(lambda x: x > 0)
        bt = bt.apply(lambda x: list(cols[x.values]), axis=1)
        bt = bt.apply(' '.join)
        return bt;


    def __fuseSplittedColumnsString(self, df):
        subgroups = self.options.getSubgroups();
        dir_data = self.options.getDirData();
        dataset = self.options.getDatasetName();
        grouping = self.options.getGroupingName();
        chunksize = self.options.getChunkSize();
        data_prefix = self.options.getDataPrefix();

        print('df_base.shape: ' + str(df.shape))
        for sub in subgroups:
            strFilenameInSubGroup = dataset + '_' + sub + '_' + grouping;
            filename_data_subgroup = dir_data + 'data_' + data_prefix + '_' + strFilenameInSubGroup + '.csv';
            print(filename_data_subgroup)
            subgroup_df = pd.DataFrame([]);
            subgroup_data_reader = pd.read_csv(filename_data_subgroup, chunksize=chunksize);
            for k, chunk in enumerate(subgroup_data_reader):
                event_ids = chunk[self.options.getEventColumnName()];
                chunk = chunk.drop(chunk.columns[chunk.columns.str.contains('unnamed', case=False)], axis=1)
                chunk = chunk.drop(self.options.getEventColumnName(), axis=1);
                #TODO: find better solution for this
                if data_prefix == 'nz':
                    chunk = chunk.drop('DIAG_COUNT', axis=1);
                str_values = self.__binaryToString(chunk);
                df_str = pd.DataFrame(data={self.options.getEventColumnName(): event_ids, sub: str_values});
                subgroup_df = subgroup_df.append(df_str);
                # subgroup_df = pd.merge(subgroup_df, df_str, on=self.options.getEventColumnName())
            print('subgroup_df: ' + str(subgroup_df.shape))
            # df[sub] = subgroup_df[sub].values;
            df = pd.merge(df, subgroup_df, on=self.options.getEventColumnName())
        print('df.shape: ' + str(df.shape))
        return df;


    def __fuseSplittedColumnsCategories(self, df):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        grouping = self.options.getGroupingName();
        subgroups = self.options.getSubgroups();
        event_column_name = self.options.getEventColumnName();
        print('df.shape: ' + str(df.shape))

        for sub in subgroups:
            strFilenameInSubGroup = dataset + '_' + sub + '_' + grouping;
            filename_data_subgroup = dir_data + 'data_' + data_prefix + '_' + strFilenameInSubGroup + '.csv';
            print(filename_data_subgroup)
            subgroup_df = pd.read_csv(filename_data_subgroup);
            # subgroup_df = subgroup_df.drop(event_column_name, axis=1);
            group_names = list(subgroup_df.columns);
            print('num group names: ' + str(len(group_names)))
            group_names.pop(group_names.index(event_column_name))
            #check if an unnamed column exist, if so -> remove
            subgroup_df = subgroup_df.drop(subgroup_df.columns[subgroup_df.columns.str.contains('unnamed', case=False)], axis=1)
            for groupname in group_names:
                subgroup_df = subgroup_df.rename(columns={groupname: sub + '_' + groupname});
            print(list(subgroup_df.columns))
            df = pd.merge(df, subgroup_df, on=self.options.getEventColumnName())
            print('df.shape: ' + str(df.shape))
        return df;


    def _removeAllFeaturesButDiag(self):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        grouping = self.options.getGroupingName();
        subgroups = self.options.getSubgroups();
        event_column_name = self.options.getEventColumnName();

        sub = 'diag';
        diag_group_names = helpers.getDKverylightGrouping();
        strFilenameInSubGroup = dataset + '_' + sub + '_' + grouping;
        filename_data_subgroup = dir_data + 'data_' + data_prefix + '_' + strFilenameInSubGroup + '.csv';
        print(filename_data_subgroup)
        subgroup_df = pd.read_csv(filename_data_subgroup, usecols=diag_group_names);
        # group_names = list(subgroup_df.columns);
        # subgroup_df = subgroup_df.drop(subgroup_df.columns[subgroup_df.columns.str.contains('unnamed', case=False)], axis=1)
        # for groupname in group_names:
        #     subgroup_df = subgroup_df.rename(columns={groupname: sub + '_' + groupname});
        try:
            subgroup_df = subgroup_df.drop(event_column_name, axis=1);
        except KeyError:
            pass;
        try:
            subgroup_df = subgroup_df.drop('DIAG_COUNT', axis=1);
        except KeyError:
            pass;
        print(list(subgroup_df.columns))
        return subgroup_df;


    def __getFilenameOptionStr(self):
        dataset = self.options.getDatasetName();
        encoding = self.options.getEncodingScheme();
        grouping = self.options.getGroupingName();
        feauture_set_str = self.options.getFeatureSetStr();
        name_dem_features = self.options.getFilenameOptionDemographicFeatures();

        filename_options_in = feauture_set_str + '_' + encoding;
        filename_options_out = feauture_set_str + '_' + encoding + '_' + grouping;

        strFilenameIn = dataset + '_' + name_dem_features + '_' + filename_options_in;
        strFilenameOut = dataset + '_' + filename_options_out;
        return [strFilenameIn, strFilenameOut]


    def __fuseSubgroupsCategories(self, df):
        df_fused = self.__fuseSplittedColumnsCategories(df);
        return df_fused;


    def __fuseSubgroupsString(self, df):
        df_fused = self.__fuseSplittedColumnsString(df);
        return df_fused;

    def fuseSubgroups(self):
        encoding = self.options.getEncodingScheme();
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        featurereduction = self.options.getFeatureReductionSettings();
        chunksize = self.options.getChunkSize();

        [filename_in_str, filename_out_str] = self.__getFilenameOptionStr()
        filename_data_out = dir_data + 'data_' + data_prefix + '_' + filename_out_str + '.csv';
        filename_data_in = dir_data + 'data_' + data_prefix + '_' + filename_in_str + '.csv';
        if not featurereduction['method'] == 'ONLYDIAG':
            df_base = pd.read_csv(filename_data_in);

        if encoding == 'embedding':
            df_finish = self.__fuseSubgroupsString(df_base);
        elif encoding == 'categorical' or encoding == 'binary':
            if featurereduction is not None:
                if featurereduction['method'] == 'ONLYADMIN':
                    df_finish = df_base;
                elif featurereduction['method'] == 'ONLYDIAG':
                    df_finish = self._removeAllFeaturesButDiag();
                else:
                    df_finish = self.__fuseSubgroupsCategories(df_base);
            else:
                df_finish = self.__fuseSubgroupsCategories(df_base);
        else:
            print('encoding scheme is not known...maybe not yet implemented..')
            sys.exit();

        print('df_finish.shape: ' + str(df_finish.shape))

        list_df = [df_finish[i:i + chunksize] for i in range(0, df_finish.shape[0], chunksize)]
        list_df[0].to_csv(filename_data_out, index=False, line_terminator='\n')
        for l in list_df[1:]:
            l.to_csv(filename_data_out, index=False, line_terminator='\n', header=False, mode='a')
        return;