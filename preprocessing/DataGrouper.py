
import sys

import pandas as pd

from shutil import copy2

from helpers.helpers import getCHOPgrouping
from helpers.helpers import getDKgrouping
from helpers.helpers import getOEgrouping


class DataGrouper:

    def __init__(self, dir_data, dataset, grouping, subgroup_names=None, chunksize=10000):
        self.dir_data = dir_data
        self.dataset = dataset
        self.filename_options_in = 'clean';
        self.grouping = grouping;
        self.filename_options_out = grouping;
        self.subgroup_names = subgroup_names;
        self.chunksize = chunksize;
        return;


    def __getGroupNames(self, group):
        if group == 'CHOP':
            group_names = getCHOPgrouping();
            group_names.insert(0, 'Fall');
        elif group == 'DK':
            group_names = getDKgrouping();
            group_names.insert(0, 'Fall');
        elif group == 'OE':
            group_names = getOEgrouping();
            group_names.insert(0, 'Fall');
        else:
            print('group name is not known...exit')
            sys.exit()
        return group_names;


    def __getFilenameStrOutSubgroup(self, strGroup):
        if self.filename_options_in is not None:
            strFilenameIn = self.dataset + '_' + strGroup + '_' + self.filename_options_in;
        else:
            strFilenameIn = self.dataset + '_' + strGroup;

        if self.filename_options_out is not None:
            strFilenameOut = self.dataset + '_' + strGroup + '_' + self.filename_options_out;
        else:
            strFilenameOut = strFilenameIn;
        return [strFilenameIn, strFilenameOut];


    def __copyRestGroup(self):
        strGroup = 'REST';
        [strFilenameIn, strFilenameOut] = self.__getFilenameStrOutSubgroup(strGroup);
        filename_data_in = self.dir_data + 'data_' + strFilenameIn + '.csv';
        filename_data_out = self.dir_data + 'data_' + strFilenameOut + '.csv';
        copy2(filename_data_in, filename_data_out);


    def __groupFeautureSubgroupNormal(self, strGroup):
        print('grouping: ' + str(strGroup))
        [strFilenameIn, strFilenameOut] = self.__getFilenameStrOutSubgroup(strGroup);
        filename_data_in = self.dir_data + 'data_' + strFilenameIn + '.csv';
        filename_data_out = self.dir_data + 'data_' + strFilenameOut + '.csv';
        group_names = self.__getGroupNames(strGroup)
        print(group_names)

        df_headers = pd.read_csv(filename_data_in, header=None, nrows=1)
        num_headers_data = len(df_headers.iloc[0]);
        print('num headers data: ' + str(num_headers_data))
        headers_data = list(df_headers.iloc[0]);
        # print('headers data: ' + str(headers_data))
        headers_data_indices = []
        for k, h in enumerate(headers_data):
            headers_data_indices.append(headers_data.index(h));

        data_reader = pd.read_csv(filename_data_in, usecols=headers_data_indices, chunksize=self.chunksize);

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
                            if h_old[3:4] == h_new:  # depends on grouping to apply: CAREFUL
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
                chunk_new.to_csv(filename_data_out, mode='a', index=False, line_terminaror='\n', header=False);


    def groupFeatures(self):
        assert(self.grouping == 'grouping', 'the only implemented scheme is normal grouping...')
        for subgroup in self.subgroup_names:
            self.__groupFeautureSubgroupNormal(subgroup);
        self.__copyRestGroup();

