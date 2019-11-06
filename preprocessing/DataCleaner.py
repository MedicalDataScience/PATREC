import os
import sys

import pandas as pd
from collections import Counter

from helpers.helpers import convertDate


class DataCleaner:

    def __init__(self, options_dataset, filename_options_in, filename_options_out):
        self.options = options_dataset;
        self.filename_options_in = filename_options_in;
        self.filename_options_out = filename_options_out;


    def __filterColumns(self, df, column_name, column_value):
        print('before: ' + str(df.shape))
        df_filtered = df.loc[df[column_name] == column_value];
        print('after: ' + str(df_filtered.shape))
        return df_filtered

    def __findAndRemoveNonIntegerVerweildauer(self, df):
        cnt_noint = 0;
        for index, row in df.iterrows():
            try:
                int_value = int(row['Verweildauer']);
            except ValueError:
                cnt_noint += 1;
                df.drop(index, inplace=True);
        print('cnt no int verweildauer: ' + str(cnt_noint));
        return df;


    def __countRowsWiederkehrer(self, df):
        df_wiederkehrer = df.loc[df['Wiederkehrer'] == 1];
        print('num rows wiederkehrer==1: ' + str(df_wiederkehrer.shape[0]));

    def __simplifyWiederkehr(self, df):
        # no row indeces are valid anymore! --> sort list by patient id
        # df = df.sort_values(by=['Patient']);

        df_wiederkehrer = df.loc[df['Wiederkehrer'] == 1];
        df_normal = df.loc[df['Wiederkehrer'] == 0];

        wiederkehrer_patient_ids = df_wiederkehrer['Patient'];
        print(wiederkehrer_patient_ids.shape)
        counts = Counter(wiederkehrer_patient_ids)

        cnt_singleappearance = 0;
        cnt_threeappearances = 0;
        cnt_other = 0
        three_appearances = [];
        for id in list(set(wiederkehrer_patient_ids)):
            if counts[id] == 3:
                three_appearances.append(id);
                cnt_threeappearances += 1;
            if counts[id] == 1:
                cnt_singleappearance += 1;

            if not counts[id] == 3 and not counts[id] == 1:
                # print('id: ' + str(id) + ': ' + str(counts[id]))
                cnt_other += 1;

        print('3 occurrences: ' + str(cnt_threeappearances))
        print('1 occurrence: ' + str(cnt_singleappearance))
        print('other number of occurrences: ' + str(cnt_other))

        cnt_wiederkehrer = 0;
        indices_3times = [];
        for index, row in df_wiederkehrer.iterrows():
            if int(row['Patient']) in three_appearances:
                cnt_wiederkehrer += 1;
                indices_3times.append(index);

        print('max(indices_3_times): ' + str(max(indices_3times)))
        df_wiederkehrer_threetimes = df.loc[indices_3times]
        df_simplified = df_normal.append(df_wiederkehrer_threetimes);
        df_simplified.reset_index(drop=True);
        print('simplified: ' + str(df_simplified.shape));
        print('simpilified wiederkehrer: ' + str(df_simplified.loc[df_simplified['Wiederkehrer'] == 1].shape))
        print('simpilified normal: ' + str(df_simplified.loc[df_simplified['Wiederkehrer'] == 0].shape))
        print('')

        return df_simplified;

    def __cleanWiederkehrer(self, df):
        print('convert date...')
        df['Aufnahmedatum'] = df['Aufnahmedatum'].apply(convertDate);
        df['Entlassdatum'] = df['Entlassdatum'].apply(convertDate);
        print('DONE!')

        df = df.sort_values(by=['Patient', 'Aufnahmedatum', 'Entlassdatum']);
        df_wiederkehrer = df.loc[df['Wiederkehrer'] == 1];
        df_normal = df.loc[df['Wiederkehrer'] == 0];

        print('wiederkehrer: ' + str(df_wiederkehrer.shape))
        print('normal: ' + str(df_normal.shape))

        patient_ids_wiederkehrer = df_wiederkehrer['Patient'].unique();
        print('num different wiederkehrer patients: ' + str(len(patient_ids_wiederkehrer)))
        print(df_wiederkehrer.shape)
        for k in range(0, len(patient_ids_wiederkehrer)):
            p_id = patient_ids_wiederkehrer[k]

            cases_df = df_wiederkehrer.loc[df_wiederkehrer['Patient'] == p_id];
            num_cases_df = cases_df.shape[0];
            # print('num cases: ' + str(num_cases_df))
            if num_cases_df == 3:
                indices = cases_df.index.tolist();

                index_current = indices[0];
                index_next = indices[1];
                index_nextnext = indices[2];

                current = cases_df.loc[indices[0]];
                next = cases_df.loc[indices[1]];
                nextnext = cases_df.loc[indices[2]];

                starttime_current = current['Aufnahmedatum'];
                endtime_current = current['Entlassdatum'];
                patientid_current = current['Patient'];

                starttime_next = next['Aufnahmedatum'];
                endtime_next = next['Entlassdatum'];
                patient_id_next = next['Patient'];

                starttime_nextnext = nextnext['Aufnahmedatum'];
                endtime_nextnext = nextnext['Entlassdatum'];
                patient_id_nextnext = nextnext['Patient'];

                if patientid_current == patient_id_next == patient_id_nextnext:
                    if starttime_nextnext > starttime_current and endtime_nextnext == endtime_next and starttime_next == starttime_current and endtime_next > endtime_current:
                        # cnt_wiederkehrer_patients += 1;
                        if starttime_nextnext > starttime_current and endtime_nextnext == endtime_next:
                            # set wiederkehrerflag to 0
                            df_wiederkehrer.loc[index_nextnext, 'Wiederkehrer'] = 0;

                        if starttime_next == starttime_current and endtime_next > endtime_current:
                            # remove next: summarizing case
                            df_wiederkehrer.drop(index_next, inplace=True);
                    else:
                        print('there seems to be a problem with the current patient regarding the timings...')
                        print(df_wiederkehrer.shape)
                        df_wiederkehrer = df_wiederkehrer.loc[df_wiederkehrer['Patient'] != patientid_current]
                        print(df_wiederkehrer.shape)
                else:
                    print(str(k) + ': patient ids do not match...');
                    print('there should only be cases of a single patient in the patient\'s list...exit!');
                    sys.exit();
            else:
                print('there should be 3 cases per wiederkehrer patient...exit')
                sys.exit()

        df_clean = df_normal.append(df_wiederkehrer);
        print('clean: ' + str(df_clean.shape))
        print('clean wiederkehrer: ' + str(df_clean.loc[df_clean['Wiederkehrer'] == 1].shape))
        print('clean normal: ' + str(df_clean.loc[df_clean['Wiederkehrer'] == 0].shape))
        return df_clean;


    def cleanData(self):
        dir_data = self.options.getDirData();
        data_prefix = self.options.getDataPrefix();
        dataset = self.options.getDatasetName();
        chunksize = self.options.getChunkSize();
        subgroups = self.options.getSubgroups();
        if self.filename_options_in is not None:
            strFilenameIn = dataset + '_REST' + '_' + self.filename_options_in;
        else:
            strFilenameIn = dataset + '_REST'
        if self.filename_options_out is not None:
            strFilenameOut = strFilenameIn + '_' + self.filename_options_out;
        else:
            strFilenameOut = strFilenameIn + '_' + self.filename_options_out;

        filename_data_in = os.path.join(dir_data, data_prefix + '_' + strFilenameIn + '.csv');
        filename_data_out_clean = os.path.join(dir_data, data_prefix + '_' + strFilenameOut + '.csv');
        print(filename_data_in)

        data_df = pd.read_csv(filename_data_in);
        print('data: ' + str(data_df.shape))
        print('max(index): ' + str(max(data_df.index.tolist())))
        data_df = self.__filterColumns(data_df, 'EntlassartVerstorben', 0);
        data_df = data_df.drop('EntlassartVerstorben', axis=1);
        print('data: ' + str(data_df.shape))
        print('max(index): ' + str(max(data_df.index.tolist())))
        data_df = self.__findAndRemoveNonIntegerVerweildauer(data_df);

        self.__countRowsWiederkehrer(data_df);
        # TODO: find a better way to handle wiederkehr label: if a patient returns multiple times --> check date
        simplified_df = self.__simplifyWiederkehr(data_df);
        print('data: ' + str(simplified_df.shape))
        print('max(index): ' + str(max(simplified_df.index.tolist())))

        clean_df = self.__cleanWiederkehrer(simplified_df)
        print('data: ' + str(clean_df.shape))

        clean_df = clean_df.sort('Fall');
        clean_df.to_csv(filename_data_out_clean, mode='w', index=False, line_terminator='\n');
        cases_clean = clean_df['Fall'].values;

        for g in subgroups:
            print('subgroup: ' + str(g))
            strFilenameIn = dataset + '_' + g;
            strFilenameOut = dataset + '_' + g + '_' + self.filename_options_out;
            filename_data_subgroup_in = dir_data + data_prefix + '_' + strFilenameIn + '.csv';
            filename_data_subgroup_out = dir_data + data_prefix + '_' + strFilenameOut + '.csv';
            subgroup_reader = pd.read_csv(filename_data_subgroup_in, chunksize=chunksize);
            for k, chunk in enumerate(subgroup_reader):
                print('chunk: ' + str(k))
                indices_keep = [];
                for index, row in chunk.iterrows():
                    case_nr = int(row['Fall']);
                    if case_nr in cases_clean:
                        indices_keep.append(index);

                chunk_clean = chunk.loc[indices_keep];
                print('chunk: ' + str(chunk.shape))
                print('chunk_clean: ' + str(chunk_clean.shape))

                if k == 0:
                    chunk_clean.to_csv(filename_data_subgroup_out, mode='w', index=False, line_terminator='\n');
                else:
                    chunk_clean.to_csv(filename_data_subgroup_out, mode='a', index=False, line_terminaror='\n',header=False);
        return;
