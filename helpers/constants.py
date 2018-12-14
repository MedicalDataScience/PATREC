import sys
import string

import helpers.helpers as helpers

filename_data_20122015 = '/scicore/home/vogtju/GROUP/PATREC_USB/PR15_2012-15___Anonym18Konsent.csv';
filename_data_20162017 = '/scicore/home/vogtju/GROUP/PATREC_USB/PR15_2016-17___Anonym18Konsent.csv';



IGNORE_HEADERS = ['EintrittsartNotfall', 'Fallart', 'DKSepsis_', 'DKSepsis_0', 'OEIntensiv_', 'OEIntensiv_0', 'HauptdiagnoseText'];

CATEGORICAL = dict();
CATEGORICAL['Aufnahmeart'] = ['V: Voranmeldung', 'AE:Aus sonst.Sp', 'SO: Spender-OP', 'A: Einweisung',
                              'G: Entbindung', 'AT:ATS Transpla', 'NO: Notfall', 'WS: Wissenschaf', 'AV:Aus Vertr.Sp',
                              'NE Neue.Neueint', 'WE:Wiedereintr.', 'N1:Nierenstein1', 'N2:Nierenstein2',
                              'S: Selbsteinw.'];
CATEGORICAL['Entlassart'] = ['iniDri', 'exPat', 'gSpit', 'vSpit', 'sSpit','Plan', 'inPat', 'iniBeh'];
CATEGORICAL['Eintrittsart'] = ['Ver', 'Not', 'Ang', 'Geb', 'Int', 'unb'];
CATEGORICAL['EntlassBereich'] = ['SaO', 'Med', 'Gyn', 'Oth', 'N.A.'];
CATEGORICAL['Versicherungsklasse'] = ['A', 'S', 'P', 'H'];
CATEGORICAL['Entlassmonat'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
CATEGORICAL['Aufnahmemonat'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
CATEGORICAL['Aufnahmetag'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
CATEGORICAL['Entlasstag'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
CATEGORICAL['Entlassjahr'] = ['2011', '2012', '2013', '2014', '2015', '2016', '2017'];
CATEGORICAL['Aufnahmejahr'] = ['2011', '2012', '2013', '2014', '2015', '2016', '2017'];
CATEGORICAL['Liegestatus'] = ['kurz', 'norm', 'lang', 'vap', 'opti', 'unb'];
CATEGORICAL['Geschlecht'] = ['weiblich', 'maennlich'];
CATEGORICAL['Forschungskonsent'] = ['ein', 'unb'];
CATEGORICAL['Hauptdiagnose'] = helpers.getDKverylightGrouping();
CATEGORICAL['AufnehmOE'] = helpers.getOEgrouping();
CATEGORICAL['EntlassOE'] = helpers.getOEgrouping();
CATEGORICAL['DRGCode'] = helpers.getDRGgrouping()

SUBGROUPS = ['OE', 'DK', 'CHOP']

NUM_DAYS_READMISSION = 18;
EARLY_READMISSION_FLAG = 'Wiederkehrer';
EVENT_FLAG = 'Fall';
HAUPTDIAGNOSE = 'Hauptdiagnose';
NEBENDIAGNOSE = 'DK'

NEW_FEATURES = ['previous_visits', 'ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK', 'ratio_numCHOP_age',
                    'ratio_los_numOE', 'ratio_numOE_age', 'mult_los_numCHOP', 'mult_equalOE_numDK',
                    'ratio_drg_los_alos'];

NEW_FEATURES_FUSION = ['ratio_los_age', 'ratio_numDK_age', 'ratio_los_numDK' ];



ADMIN_FEATURES_NAMES = ['Fall', 'Aufnahmeart', 'Aufnahmedatum', 'Entlassdatum', 'Aufnahmejahr', 'Entlassjahr',
                        'Aufnahmemonat', 'Entlassmonat', 'Aufnahmetag', 'Entlasstag', 'Wiederkehrer', 'Eintrittsalter',
                        'Eintrittsart', 'Entlassart', 'EntlassBereich', 'Versicherungsklasse', 'Patient',
                        'Forschungskonsent', 'Geschlecht', 'Verweildauer', 'Liegestatus', 'Langlieger']

LIEGESTATUS_FEATURES = ['Liegestatus', 'Langlieger']

FUSION_FEATURES = ['Fall', 'Geschlecht', 'Eintrittsalter', 'Hauptdiagnose', 'Verweildauer'] + NEW_FEATURES_FUSION;

DISEASES = ['cardiovascular', 'oncology', 'chronic_lung']

MDC = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', '9'];

COLUMNS_TO_REMOVE_FOR_CLASSIFIER = ['Fall', 'Aufnahmedatum', 'Entlassdatum', 'Patient']


NAME_DEMOGRAPHIC_FEATURE = 'REST';

################################################
#
# OLD VERSION OF CONSTANT STORING
#
################################################


# YEARS = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
# YEARS_INT = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
# MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
# DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
# GESCHLECHT = ['weiblich', 'maennlich'];
# FORSCHUNGSKONSENTE = ['ein', 'unb'];
# VERSICHERUNGSKLASSEN = ['A', 'S', 'P', 'H'];
# ENTLASSBEREICHE = ['SaO', 'Med', 'Gyn', 'Oth', 'N.A.'];
# ENTLASSART = ['iniDri', 'exPat', 'gSpit', 'vSpit', 'sSpit','Plan', 'inPat', 'iniBeh'];
# AUFNAHMEART = ['V: Voranmeldung', 'AE:Aus sonst.Sp', 'SO: Spender-OP', 'A: Einweisung', 'G: Entbindung', 'AT:ATS Transpla',
#                'NO: Notfall', 'WS: Wissenschaf', 'AV:Aus Vertr.Sp', 'NE Neue.Neueint', 'WE:Wiedereintr.',
#                'N1:Nierenstein1', 'N2:Nierenstein2', 'S: Selbsteinw.']
# LIEGESTATUS = ['kurz', 'norm', 'lang', 'vap', 'opti', 'unb'];
# EINTRITTSART = ['Ver', 'Not', 'Ang', 'Geb', 'Int', 'unb'];
# VERWEILDAUER = ['sehrkurz', 'kurz', 'mittel', 'mittellang', 'lang', 'sehrlang'];
# EINTRITTSALTER = ['jung', 'mittel', 'mittelalt', 'alt'];


# IGNORE_HEADERS = ['Fall', 'EintrittsartNotfall', 'Fallart', 'DKSepsis_', 'DKSepsis_0', 'OEIntensiv_', 'OEIntensiv_0', 'HauptdiagnoseText', 'DRGCode'];

# TO_CATEGORICAL = ['Aufnahmeart', 'Entlassart', 'EntlassBereich', 'Versicherungsklasse', 'Forschungskonsent', 'Entlassmonat', 'Aufnahmemonat', 'Aufnahmetag', 'Entlasstag', 'Verweildauer', 'Eintrittsalter', 'Entlassjahr', 'Aufnahmejahr', 'Liegestatus']
#
# CATEGORICAL = ['Aufnahmeart', 'Entlassart', 'Eintrittsart', 'EntlassBereich', 'Versicherungsklasse', 'Forschungskonsent', 'Entlassmonat', 'Aufnahmemonat', 'Aufnahmetag', 'Entlasstag', 'Entlassjahr', 'Aufnahmejahr', 'Liegestatus']
# BINARY = ['Geschlecht', 'Langlieger', 'DKSepsis_1', 'OEIntensiv_1'];
# SPARSE = ['Hauptdiagnose', 'AufnehmOE', 'EntlassOE', 'DRGCode', 'CHOP', 'DK', 'OE'];
# TO_REMOVE = ['EntlassartVerstorben']
# PREPROCESSING = ['Patient', 'Aufnahmedatum', 'Entlassdatum']

# CATEGORICAL_ONEHOT = ['Aufnahmeart', 'Entlassart', 'Eintrittsart', 'EntlassBereich', 'Versicherungsklasse', 'Entlassmonat', 'Aufnahmemonat', 'Aufnahmetag', 'Entlasstag', 'Entlassjahr', 'Aufnahmejahr', 'Liegestatus', 'Hauptdiagnose', 'AufnehmOE', 'EntlassOE', 'DRGCode'];
# CATEGORICAL_BINARY = ['Geschlecht', 'Forschungskonsent'];

# NUMERICAL = ['Verweildauer', 'Eintrittsalter'];

# def getMonthValues():
#     return MONTHS.copy();
#
# def getYearValues():
#     return YEARS.copy();
#
# def getDayValues():
#     return DAYS.copy();
#
# def getGeschlechtValues():
#     return GESCHLECHT.copy()
#
# def getForschungskonsentValues():
#     return FORSCHUNGSKONSENTE.copy()
#
# def getVersicherungsklasseValues():
#     return VERSICHERUNGSKLASSEN.copy();
#
# def getEntlassBereichValues():
#     return ENTLASSBEREICHE.copy();
#
# def getEntlassartValues():
#     return ENTLASSART.copy();
#
# def getAufnahmeartValues():
#     return AUFNAHMEART.copy();
#
# def getLiegestatusValues():
#     return LIEGESTATUS.copy();
#
# def getEintrittsartValues():
#     return EINTRITTSART.copy();
#
#
# def getFeaturesToCategorize():
#     features = CATEGORICAL_BINARY + CATEGORICAL_ONEHOT;
#     return features.copy();
#
#
# def getCountFeaturesToBinarize():
#     features = NUMERICAL;
#     return features.copy();
#
#
# def getNumericalFeatures():
#     features = NUMERICAL;
#     return features.copy();
#
#
# def getAdminFeaturesNames():
#     return ADMIN_FEATURES_NAMES.copy();
#
#
# def getVerweildauerCategories():
#     return VERWEILDAUER.copy();
#
#
# def getAlterCategories():
#     return EINTRITTSALTER.copy();
#
#
# def getLOSState(valStr):
#     val = int(valStr)
#     if val < 3:
#         ind=0;
#     elif val >= 3 and val < 10:
#         ind = 1;
#     elif val >= 10 and val < 30:
#         ind = 2;
#     elif val >= 30 and val < 70:
#         ind = 3;
#     elif val >= 70 and val < 150:
#         ind = 4;
#     elif val >= 150:
#         ind = 5;
#     else:
#         print('LOS: this should not happen...exit')
#         sys.exit();
#     return VERWEILDAUER[ind];
#
#
# def getFeatureCategories(name):
#     if name == 'Liegestatus':
#         return getLiegestatusValues();
#     elif name == 'EntlassBereich':
#         return getEntlassBereichValues();
#     elif name == 'Versicherungsklasse':
#         return getVersicherungsklasseValues();
#     elif name == 'Eintrittsart':
#         return getEintrittsartValues();
#     elif name == 'Entlassart':
#         return getEntlassartValues();
#     elif name == 'Forschungskonsent':
#         return getForschungskonsentValues();
#     elif name == 'Geschlecht':
#         return getGeschlechtValues();
#     elif name.endswith('tag'):
#         return getDayValues();
#     elif name.endswith('monat'):
#         return getMonthValues();
#     elif name.endswith('jahr'):
#         return getYearValues();
#     elif name == 'Aufnahmeart':
#         return getAufnahmeartValues();
#     elif name.endswith('OE') or name.startswith('OE'):
#         return getOEgrouping();
#     elif name == 'Hauptdiagnose':
#         return getDKlightGrouping();
#     elif name == 'DRGCode':
#         return getDRGgrouping();
#     elif name.startswith('DK'):
#         return getDKgrouping();
#     elif name.startswith('CHOP'):
#         return getCHOPgrouping();
#     elif name == 'Verweildauer':
#         return getVerweildauerCategories();
#     elif name == 'Eintrittsalter':
#         return getAlterCategories();
#     else:
#         print('feature is not known...exit: ' + str(name))
#         sys.exit();
#
#
# def getAgeState(valStr):
#     val = int(valStr)
#     if val < 30:
#         ind = 0;
#     elif val >= 30 and val < 60:
#         ind = 1;
#     elif val >= 60 and val < 90:
#         ind = 2;
#     elif val >= 90:
#         ind = 3;
#     else:
#         print('AGE: this should not happen...exit')
#         sys.exit();
#     return EINTRITTSALTER[ind];