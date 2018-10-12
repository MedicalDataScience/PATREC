import sys
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing.dummy as mp
from itertools import repeat

import helpers.helpers as helpers
import helpers.constantsNZ as constants
from helpers.icd_conversion_helper import ICDConversionHelper

from preprocessing.FeatureEncoder import FeatureEncoder
from preprocessing.DataPreparer import DataPreparer

class PreprocessorNZ():

    def __init__(self, options_dataset):
        self.options = options_dataset;
        self.conversion_helper = ICDConversionHelper();
        return;


    def _build_diagnosis_code_set(self, header_mode):
        ret_list = list()
        ret_list.append(constants.DIAGNOSIS_COUNT)

        if header_mode == constants.HEADER_MODE_COMPRESS:
            dk_grouping = helpers.getDKgrouping();
            for dk in dk_grouping:
                ret_list.append(constants.DIAGNOSIS_PREFIX + str(dk))
        # # Always append main diagnosis columns
        # for code in range(ord("A"), ord("Z") + 1):
        #     # ret_list.append(constants.DIAGNOSIS_PREFIX + chr(code) + constants.DIAGNOSIS_MAIN)
        #
        #     # Only add other diagnosis columns if in compressed mode
        #
        #         for i in range(0, 10):
        #             ret_list.append(constants.DIAGNOSIS_PREFIX + chr(code) + str(i))

        # If in original header mode, add one column for each ICD10v8 code
        if header_mode == constants.HEADER_MODE_ORIGINAL:
            codes = self.conversion_helper.get_icd10_8_codes()
            for code in codes:
                ret_list.append(constants.DIAGNOSIS_PREFIX + code)
        return ret_list


    def _clean_data(self, data_frame):
        #convert early readmission flag into 0 and 1
        data_frame[constants.EARLY_READMISSION_FLAG] = data_frame[constants.EARLY_READMISSION_FLAG].apply(constants.EARLY_READMISSION_VALUES.index);
        # Drop columns that should be ignored
        for drop_column in constants.DROP_COLUMNS:
            data_frame = data_frame.drop(drop_column, axis=1)
        # Drop rows where data is missing
        data_frame = data_frame.dropna()
        # Drop rows where the event ended in a patient's death
        data_frame = data_frame[data_frame['end_type'] != 'DD']
        return data_frame

    def _remove_duplicate_clin_cd_codes(self, data_frame):
        clin_systems = data_frame['CLIN_SYS'].unique()
        print('remove_duplicate_clin_cd_codes : Detected {} code systems'.format(clin_systems))

        if len(clin_systems) > 1:
            print('remove_duplicate_clin_cd_codes : Removing old codes...')

            # Here, we want to eliminate rows where the CLIN_SYS value is not equal to the SUB_SYS value
            data_frame = data_frame[data_frame.CLIN_SYS == data_frame.SUB_SYS]

        return data_frame


    def _convert_diag_code(self, diag_row):
        clin_cd = diag_row["CLIN_CD"]
        diag_typ = diag_row["DIAG_TYP"]
        clin_sys = diag_row["CLIN_SYS"]

        # TODO: Implement handling for procedures. For now, skip them
        if diag_typ == constants.DIAG_TYPE_OPERATION:
            return None;

        # If the diagnosis is in ICD9 or ICD10_6, convert it to ICD_10_8
        try:
            if clin_sys == constants.CLIN_SYS_ICD9:
                clin_cd = self.conversion_helper.convert_icd9_to_icd10(clin_cd, diag_typ)
            elif clin_sys == constants.CLIN_SYS_ICD10_6:
                clin_cd = self.conversion_helper.convert_icd10_6_to_icd10_8(clin_cd)
        except KeyError:
            print("KeyError encountered for old ICD code {}. Skipping...".format(clin_cd), file=sys.stderr)
            return None;
        return clin_cd;


    def _get_main_diag(self, diag_df, discharge_df):
        print('curr_diag_df: ' + str(diag_df.shape))
        diag_main_df = diag_df.loc[diag_df['DIAG_TYP'] == constants.DIAG_TYP_MAIN];
        print('curr_diag_main_df: ' + str(diag_main_df.shape))
        diag_main_df['main_diag'] = '';
        for k, row in diag_main_df.iterrows():
            main_diag = self._convert_diag_code(row)
            diag_main_df.at[k, 'main_diag'] = main_diag;

        diag_main_df = diag_main_df[['EVENT_ID', 'main_diag']]
        curr_df = pd.merge(discharge_df, diag_main_df, left_on='event_id', right_on='EVENT_ID')
        curr_df = curr_df.drop('EVENT_ID', axis=1);
        print('current discharge df: ' + str(curr_df.shape))
        return curr_df;


    def _process_diagnoses_loop(self, curr_disc_df, curr_diag_df, header_mode):
        disc_indices = curr_disc_df.index.values

        for disc_index in disc_indices:
            # Find rows in the diagnosis table corresponding to this event
            diag_rows = curr_diag_df[curr_diag_df.EVENT_ID == curr_disc_df.at[disc_index, 'event_id']]

            # Iterate over these diagnoses and update the discharge table accordingly
            for diag_index, diag_row in diag_rows.iterrows():
                clin_cd = self._convert_diag_code(diag_row);
                diag_typ = diag_row["DIAG_TYP"]

                if clin_cd is None:
                    continue;

                # If header mode is compressed
                if header_mode == constants.HEADER_MODE_COMPRESS:
                    # Get the first two characters of the clin_cd value
                    code_letter = clin_cd[0]
                    code_first_number = clin_cd[1]

                    # Increment the associated fields
                    if diag_typ == constants.DIAG_TYP_MAIN:
                        # curr_disc_df.at[disc_index, "{}{}{}".format(constants.DIAGNOSIS_PREFIX, code_letter, constants.DIAGNOSIS_MAIN)] += 1
                        continue;
                    else:
                        curr_disc_df.at[disc_index, "{}{}".format(constants.DIAGNOSIS_PREFIX, code_letter)] = 1
                else:
                    # Only use the first character to determine the main diagnosis type
                    code_letter = clin_cd[0]

                    # Increment the associated fields
                    if diag_typ == constants.DIAG_TYP_MAIN:
                        # curr_disc_df.at[disc_index, "{}{}{}".format(constants.DIAGNOSIS_PREFIX, code_letter, constants.DIAGNOSIS_MAIN)] += 1
                        continue;
                    else:
                        curr_disc_df.at[disc_index, "{}{}".format(constants.DIAGNOSIS_PREFIX, clin_cd)] = 1

                curr_disc_df.at[disc_index, constants.DIAGNOSIS_COUNT] += 1

        return curr_disc_df


    def _extract_and_drop_labels(self, data_frame):
        # Extract labels
        labels = data_frame[constants.EARLY_READMISSION_FLAG]
        data_frame.drop(constants.EARLY_READMISSION_FLAG, axis=1, inplace=True)
        return data_frame, labels


    def _drop_id_data(self, data_frame):
        data_frame = data_frame.drop('unique_patient_identifier', axis=1)
        data_frame = data_frame.drop('event_id', axis=1)
        return data_frame


    def _get_cols_max_values(self, matrix, headers):
        ret_list = []

        for i in range(0, matrix.shape[1]):
            # If the column is a categorical column, return the number of categories minus 1
            # These will be used later to normalize the data
            if headers[i] in constants.CATEGORICAL_DATA:
                ret_list.append(len(constants.CATEGORICAL_DATA[headers[i]]) - 1)
            else:
                ret_list.append(matrix[:, i].max(axis=0))

        return np.asarray(ret_list)


    def _write_headers(self, str_list, filename):
        file_out = open(filename, 'w')
        for hdr in str_list:
            file_out.write("%s\n" % hdr)
        file_out.close()


    def processDiagnosisFile(self):
        dataset = self.options.getDatasetName();

        print("main : Processing year {}".format(dataset))
        curr_discharge_file = constants.DISCHARGE_FILE_TEMPLATE.format(dataset)
        curr_disc_df = pd.read_table(curr_discharge_file, sep='|', dtype=constants.EXPLICIT_DATA_TYPES)
        curr_disc_df = curr_disc_df[['event_id']]
        print('curr discharge file: ' + str(curr_disc_df.shape))

        # Load the diagnosis file for the current year
        curr_diagnosis_file = constants.DIAGNOSIS_FILE_TEMPLATE.format(dataset)
        curr_diag_df = pd.read_table(curr_diagnosis_file, sep='|', usecols=[2, 3, 4, 5, 6, 7], dtype={"CLIN_CD": str})
        # Clean up the diagnosis data
        curr_diag_df = self._remove_duplicate_clin_cd_codes(curr_diag_df)

        header_mode = constants.HEADER_MODE_COMPRESS
        # Get list of columns to add
        additional_columns = self._build_diagnosis_code_set(header_mode)
        # Add the additional columns to the table, default the value to zero
        for col in additional_columns:
            curr_disc_df[col] = 0;

        print(list(curr_disc_df.columns))

        # Split discharge data frame into parts for multiprocessing
        curr_disc_df_split = np.array_split(curr_disc_df, constants.NUM_FRACTIONS)

        # Do multithreaded for loop
        with mp.Pool(constants.NUM_PROCS) as pool:
            print("main : Multiprocessing with {} processors".format(constants.NUM_PROCS), file=sys.stderr)

            result = pool.starmap_async(self._process_diagnoses_loop, zip(curr_disc_df_split, repeat(curr_diag_df),
                                                                          repeat(header_mode)), chunksize=1)

            # Iterate over all results and determine progress
            # TODO: Find a better way to measure progress without checking a private member
            num_chunks = result._number_left
            pbar = tqdm(total=num_chunks)

            mp_done = False
            curr_num_done = 0

            while mp_done is not True:
                time.sleep(1)

                num_done = num_chunks - result._number_left

                # Update progress
                curr_diff = num_done - curr_num_done

                if curr_diff != 0:
                    pbar.update(curr_diff)
                    curr_num_done = num_done

                if num_done == num_chunks:
                    mp_done = True

            # All done, join the output
            print("main : Multiprocessing done, joining output...")
            curr_disc_df = pd.concat(result.get())

            pbar.close()

        print("main : Done processing diagnoses for year {}".format(dataset))
        print('curr_diag_df: ' + str(curr_disc_df.shape));
        curr_diag_df = curr_disc_df;

        dir_data = self.options.getDirData();
        dataset = self.options.getDatasetName();
        grouping = self.options.getGroupingName();
        data_prefix = self.options.getDataPrefix();
        filename_out = dir_data + 'data_' + data_prefix + '_' + dataset + '_diag_' + grouping + '.csv';
        curr_diag_df.to_csv(filename_out);





    def processDischargeFile(self):
        # Either process all years, or only specified years (if provided)
        dataset = self.options.getDatasetName();

        # Process years one at a time
        # for year in years:
        # Process the discharge file for the current year
        print("main : Processing year {}".format(dataset))
        curr_discharge_file = constants.DISCHARGE_FILE_TEMPLATE.format(dataset)
        curr_disc_df = pd.read_table(curr_discharge_file, sep='|', dtype=constants.EXPLICIT_DATA_TYPES)
        print(list(curr_disc_df.columns))
        # Clean up the discharge data frame before processing it, so that we don't do unnecessary processing
        print("main : Cleaning data for year {}".format(dataset))
        print('main : current discharge file: ' + str(curr_disc_df.shape))
        try:
            curr_disc_df = self._clean_data(curr_disc_df)
        except ValueError:
            # We could not numerize the data, so skip this file
            print("main : Could not numerize date for year {}, skipping...".format(dataset))
            sys.exit()

        curr_diagnosis_file = constants.DIAGNOSIS_FILE_TEMPLATE.format(dataset)
        curr_diag_df = pd.read_table(curr_diagnosis_file, sep='|', usecols=[2, 3, 4, 5, 6, 7], dtype={"CLIN_CD": str})
        # Clean up the diagnosis data
        curr_diag_df = self._remove_duplicate_clin_cd_codes(curr_diag_df)
        curr_disc_df = self._get_main_diag(curr_diag_df, curr_disc_df);

        print("main : Done processing diagnoses for year {}".format(dataset))
        dir_data = self.options.getDirData();
        dataset = self.options.getDatasetName();
        data_prefix = self.options.getDataPrefix();
        filename_out = dir_data + 'data_' + data_prefix + '_' + dataset + '_discharge.csv';
        curr_disc_df.to_csv(filename_out);



    def encodeFeatures(self):
        encoder = FeatureEncoder(self.options);
        encoder.encodeFeaturesNZ();

    def fuse(self):
        preparer = DataPreparer(self.options);
        preparer.fuseSubgroupsNZ();