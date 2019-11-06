import os
import sys

import pandas as pd

from shutil import copy2

from helpers.helpers import getCHOPgrouping
from helpers.helpers import getDKgrouping
from helpers.helpers import getDKlightGrouping
from helpers.helpers import getDKverylightGrouping
from helpers.helpers import getOEgrouping


class DataGrouper:

    def __init__(self, options_dataset):
        self.options = options_dataset;
        return;


    def __getGroupNames(self, group):
        if group == 'CHOP':
            group_names = getCHOPgrouping();
            group_names.insert(0, 'Fall');
        elif group == 'DK':
            if self.options.getGroupingName() == 'grouping':
                group_names = getDKgrouping();
            elif self.options.getGroupingName() == 'lightgrouping':
                group_names = getDKlightGrouping();
            elif self.options.getGroupingName() == 'verylightgrouping':
                group_names = getDKverylightGrouping();
            else:
                print('grouping scheme ist not known...exit')
                sys.exit();
            group_names.insert(0, 'Fall');
        elif group == 'OE':
            group_names = getOEgrouping();
            group_names.insert(0, 'Fall');
        else:
            print('group name is not known...exit')
            sys.exit()
        return group_names;


    def __getFilenameStrOutSubgroup(self, strGroup):
        dataset = self.options.getDatasetName();
        grouping = self.options.getGroupingName();
        strFilenameIn = dataset + '_' + strGroup + '_clean';
        strFilenameOut = dataset + '_' + strGroup + '_' + grouping;
        return [strFilenameIn, strFilenameOut];


    def __copyRestGroup(self):
        strGroup = 'REST';
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        [strFilenameIn, strFilenameOut] = self.__getFilenameStrOutSubgroup(strGroup);
        filename_data_in = os.path.join(dir_data, 'data_' + data_prefix + '_' + strFilenameIn + '.csv');
        filename_data_out = os.path.join(dir_data, 'data_' + data_prefix + '_' + strFilenameOut + '.csv');
        copy2(filename_data_in, filename_data_out);


    def _getDKendIndex(self):
        if self.options.getGroupingName() == 'grouping':
            return 4;
        elif self.options.getGroupingName() == 'lightgrouping':
            return 5;
        elif self.options.getGroupingName() == 'verylightgrouping':
            return 6;
        else:
            print('grouping scheme ist not known...exit')
            sys.exit();


    def __groupFeautureSubgroupNormal(self, strGroup):
        print('grouping: ' + str(strGroup))
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        chunksize = self.options.getChunkSize();

        [strFilenameIn, strFilenameOut] = self.__getFilenameStrOutSubgroup(strGroup);
        filename_data_in = os.path.join(dir_data, data_prefix + '_' + strFilenameIn + '.csv');
        filename_data_out = os.path.join(dir_data, data_prefix + '_' + strFilenameOut + '.csv');
        group_names = self.__getGroupNames(strGroup)
        print(group_names)
        print(len(group_names))

        df_headers = pd.read_csv(filename_data_in, header=None, nrows=1)
        num_headers_data = len(df_headers.iloc[0]);
        print('num headers data: ' + str(num_headers_data))
        headers_data = list(df_headers.iloc[0]);
        # print('headers data: ' + str(headers_data))
        headers_data_indices = []
        for k, h in enumerate(headers_data):
            headers_data_indices.append(headers_data.index(h));

        dk_end_index = self._getDKendIndex();
        data_reader = pd.read_csv(filename_data_in, usecols=headers_data_indices, chunksize=chunksize);

        for k, chunk in enumerate(data_reader):
            print('chunk: ' + str(k))
            chunk_new = pd.DataFrame(index=chunk.index, columns=group_names);
            chunk_new = chunk_new.fillna(0);
            chunk = chunk.fillna(0);

            for h_new in group_names:
                if h_new in headers_data:
                    chunk_new[h_new] = chunk[h_new];
                else:
                    chunk[h_new] = 0.0;
                    if strGroup == 'DK':
                        for h_old in headers_data:
                            if h_old[3:dk_end_index] == h_new:  # depends on grouping to apply: CAREFUL
                                chunk[h_new] = (chunk[h_new].astype(int) | chunk[h_old].astype(int)).astype(int);
                        chunk_new[h_new] = chunk[h_new];
                    if strGroup == 'CHOP':
                        for h_old in headers_data:
                            if h_old[5:7] == h_new:  # depends on grouping to apply: CAREFUL
                                chunk[h_new] = (chunk[h_new].astype(int) | chunk[h_old].astype(int)).astype(int);
                        chunk_new[h_new] = chunk[h_new];
                    if strGroup == 'OE':
                        for h_old in headers_data:
                            if h_old[3:] == h_new:  # depends on grouping to apply: CAREFUL
                                chunk[h_new] = (chunk[h_new].astype(int) | chunk[h_old].astype(int)).astype(int);
                        chunk_new[h_new] = chunk[h_new];

            if k == 0:
                chunk_new.to_csv(filename_data_out, mode='w', index=False, line_terminator='\n', header=group_names);
            else:
                chunk_new.to_csv(filename_data_out, mode='a', index=False, line_terminator='\n', header=False);


    def groupFeatures(self):
        grouping = self.options.getGroupingName();
        # assert(grouping == 'grouping', 'the only implemented scheme is normal grouping...')
        subgroups = self.options.getSubgroups();
        for subgroup in subgroups:
            self.__groupFeautureSubgroupNormal(subgroup);
        self.__copyRestGroup();

