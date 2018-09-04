import sys
import pandas as pd

from helpers.constants import IGNORE_HEADERS


class ColumnSplitter:

    def __init__(self, filename_src, chunksize):
        self.chunksize = chunksize
        self.filename_data_src = filename_src;
        self.headers_all_columns = self.__getHeadersAll();
        return;

    def __getColumnsCHOP(self):
        headers_chop = [];
        for h in self.headers_all_columns:
            if h.startswith('CHOP_'):
                headers_chop.append(h);
        print('num headers CHOP: ' + str(len(headers_chop)))
        return headers_chop;

    def __getColumnsDK(self):
        headers_dk = [];
        for h in self.headers_all_columns:
            if h.startswith('DK_'):
                headers_dk.append(h);
        print('num headers DK: ' + str(len(headers_dk)))
        return headers_dk;

    def __getColumnsOE(self):
        headers_oe = [];
        for h in self.headers_all_columns:
            if h.startswith('OE_'):
                headers_oe.append(h)
        print('num headers OE: ' + str(len(headers_oe)))
        return headers_oe;

    def __getColumnsRest(self):
        headers_rest = [];
        headers_oe = self.__getColumnsOE();
        headers_chop = self.__getColumnsCHOP();
        headers_dk = self.__getColumnsDK();

        for h in self.headers_all_columns:
            if h not in headers_oe and h not in headers_dk and h not in headers_chop:
                headers_rest.append(h);
        print('num headers REST: ' + str(len(headers_rest)))
        return headers_rest;

    def __getSubgroupColumns(self, strGroup):

        if strGroup == 'CHOP':
            headers_subgroup = self.__getColumnsCHOP();
        elif strGroup == 'DK':
            headers_subgroup = self.__getColumnsDK();
        elif strGroup == 'OE':
            headers_subgroup = self.__getColumnsOE();
        elif strGroup == 'REST':
            headers_subgroup = self.__getColumnsRest();
        else:
            print('group is not known...exit')
            sys.exit()
        headers_subgroup_cleaned = self.__removeIgnoreHeaders(headers_subgroup);
        return headers_subgroup_cleaned;

    def __removeIgnoreHeaders(self, headers):
        headers_filtered = [];
        for header in headers:
            if not header in IGNORE_HEADERS:
                headers_filtered.append(header);
            else:
                print('column name ' + str(header) + ' is part of the IGNORE_HEADERS....');
        return headers_filtered;

    def __getHeadersAll(self):
        df_headers = pd.read_csv(self.filename_data_src, header=None, nrows=1)
        num_headers_data = len(df_headers.iloc[0]);
        print('num headers data: ' + str(num_headers_data))

        headers_data_in_values = list(df_headers.iloc[0]);
        return headers_data_in_values;


    def splitColumns(self, subgroup_name, filename_data_out):
        headers_subgroup_columns = self.__getSubgroupColumns(subgroup_name)
        if subgroup_name != 'REST':
            headers_subgroup_columns.insert(0, 'Fall');
        headers_subgroup_indices = [];
        for k, h in enumerate(headers_subgroup_columns):
            headers_subgroup_indices.append(self.headers_all_columns.index(h));

        print('headers_data_in_values.index(Fall): ' + str(self.headers_all_columns.index('Fall')))
        print('len(headers_subgroup_indices): ' + str(len(headers_subgroup_indices)))

        readmission_data_reader = pd.read_csv(self.filename_data_src, usecols=headers_subgroup_indices, chunksize=self.chunksize);
        for k, chunk in enumerate(readmission_data_reader):
            print('chunk: ' + str(k))
            print('chunk.shape: ' + str(chunk.shape))
            if k == 0:
                chunk.to_csv(filename_data_out, mode='w', index=False, line_terminator='\n',header=headers_subgroup_columns);
            else:
                chunk.to_csv(filename_data_out, mode='a', index=False, line_terminaror='\n', header=False);