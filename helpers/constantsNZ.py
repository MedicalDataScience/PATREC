
import helpers.helpers as helpers

YEARS_TO_PROCESS = ["1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999",
                    "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",
                    "2012", "2013", "2014", "2015", "2016", "2017"]

DISCHARGE_FILE_TEMPLATE = '/Users/towyku74/UniBas/sciCore/projects/PATREC/nz/data_src/pus10175_discharge_readmit_{}.txt'
DIAGNOSIS_FILE_TEMPLATE = '/Users/towyku74/UniBas/sciCore/projects/PATREC/nz/data_src/pus10175_diagnosis_{}.txt'
#DISCHARGE_FILE_TEMPLATE = '/Users/thomas/UniBas/sciCore/projects/PATREC/nz/data_src/pus10175_discharge_readmit_{}.txt'
#DIAGNOSIS_FILE_TEMPLATE = '/Users/thomas/UniBas/sciCore/projects/PATREC/nz/data_src/pus10175_diagnosis_{}.txt'
# DISCHARGE_FILE_TEMPLATE = '/scicore/home/vogtju/towyku74/projects/PATREC/nz/data_src/pus10175_discharge_readmit_{}.txt'
# DIAGNOSIS_FILE_TEMPLATE = '/scicore/home/vogtju/towyku74/projects/PATREC/nz/data_src/pus10175_diagnosis_{}.txt'

OUTPUT_DIR = '/scicore/home/vogtju/towyku74/projects/PATREC/nz/data_prepared/'
# For debugging
#OUTPUT_DIR = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) + '/NZ/data/'
#DISCHARGE_FILE_TEMPLATE = OUTPUT_DIR + 'raw/pus10175_discharge_readmit_{}.txt'
#DIAGNOSIS_FILE_TEMPLATE = OUTPUT_DIR + 'raw/pus10175_diagnosis_{}.txt'

DIAGNOSIS_PREFIX = "DIAG_"
DIAGNOSIS_COUNT = "DIAG_COUNT"
DIAGNOSIS_MAIN = "_MAIN"
DIAGNOSIS_OTHER = "_OTHER"

DIAG_TYP_MAIN = "A"
DIAG_TYP_OTHER = "B"
DIAG_TYPE_EXTERNAL = "E"
DIAG_TYPE_OPERATION = "O"

CLIN_SYS_ICD9 = 1
CLIN_SYS_ICD10_6 = 13
CLIN_SYS_ICD10_8 = 14

# Categorical data
CATEGORICAL_DATA = dict()
CATEGORICAL_DATA['gender'] = ['M', 'F', 'U', 'I']
CATEGORICAL_DATA['adm_src'] = ['R', 'T']
CATEGORICAL_DATA['adm_type'] = ['AA', 'AC', 'AP', 'RL', 'WN', 'ZA', 'ZC', 'ZP', 'ZW', 'WU']
CATEGORICAL_DATA['event_type'] = ['BT', 'CM', 'CO', 'CS', 'DM', 'DP', 'DT', 'GP', 'ID', 'IM', 'IP', 'MC', 'NP', 'OP']
CATEGORICAL_DATA['end_type'] = ['DA', 'DC', 'DD', 'DF', 'DI', 'DL', 'DN', 'DO', 'DP', 'DR', 'DS', 'DT', 'DW', 'EA',
                                'ED', 'EI', 'ER', 'ES', 'ET', 'RO']
CATEGORICAL_DATA['facility_type'] = ['1', '2', '3', '4', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                     '19', '20', '21', '22', '23', '24', '25', '99']
CATEGORICAL_DATA['agency_type'] = ['1', '2', '9', '10', '11', '12', '13', '14', '8']
CATEGORICAL_DATA['private_flag'] = ['N', 'Y']
CATEGORICAL_DATA['purchaser'] = ['6', '17', '19', '20', '33', '34', '35', '55', '98', 'A0', '1', '2', '3', '4',
                                 '5', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '18', 'A1', 'A2',
                                 'A3', 'A4', 'A5', 'A6', 'A7']
CATEGORICAL_DATA['Short_Stay_ED_Flag'] = ['N', 'Y']
#CATEGORICAL_DATA['early_readmission_flag'] = ['N', 'Y']
CATEGORICAL_DATA['transfer_event_flag'] = ['N', 'Y']
CATEGORICAL_DATA['main_diag'] = helpers.getDKverylightGrouping();

EXPLICIT_DATA_TYPES = {'gender': str, 'adm_src': str, 'adm_type': str, 'event_type': str, 'end_type': str,
                       'facility_type': str, 'agency_type': str, 'private_flag': str, 'purchaser': str,
                       'Short_Stay_ED_Flag': str, 'early_readmission_flag': str, 'transfer_event_flag': str}

DROP_COLUMNS = ['discharge_year', 'evstdate', 'evsttime', 'evendate',
                'eventime', 'evntlvd', 'facility', 'agency']

EARLY_READMISSION_FLAG = 'early_readmission_flag'
EARLY_READMISSION_VALUES = ['N', 'Y'];

EVENT_FLAG = 'event_id';
HAUPTDIAGNOSE = 'main_diag';
NEBENDIAGNOSE = 'diag'
NUMERICAL = ['los', 'age_dsch'];

COLUMNS_TO_REMOVE_FOR_CLASSIFIER = ['unique_patient_identifier', 'event_id']

FUSION_FEATURES = ['event_id', 'age_dsch', 'gender', 'los', 'main_diag']

NAME_DEMOGRAPHIC_FEATURE = 'discharge';


NEW_FEATURES = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age',
                    'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK',
                    'ratio_drg_los_alos'];

# For multithreading
NUM_PROCS = 6
NUM_FRACTIONS = 100

# Header mode
HEADER_MODE_ORIGINAL = 0
HEADER_MODE_COMPRESS = 1
