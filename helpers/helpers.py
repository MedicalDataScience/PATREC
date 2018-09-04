import sys
import os
import string
import datetime

dirHelpers = os.path.dirname(os.path.abspath(__file__));
if not dirHelpers in sys.path:
    sys.path.append(dirHelpers)

from helpers.constants import IGNORE_HEADERS
from helpers.constants import OE_str;
from helpers.constants import ADMIN_FEATURES_NAMES
from helpers.constants import MONTHS
from helpers.constants import DAYS
from helpers.constants import YEARS
from helpers.constants import GESCHLECHT
from helpers.constants import FORSCHUNGSKONSENTE
from helpers.constants import VERSICHERUNGSKLASSEN
from helpers.constants import ENTLASSBEREICHE
from helpers.constants import ENTLASSART
from helpers.constants import AUFNAHMEART
from helpers.constants import LIEGESTATUS
from helpers.constants import EINTRITTSART
from helpers.constants import CATEGORICAL_ONEHOT;
from helpers.constants import CATEGORICAL_BINARY
from helpers.constants import NUMERICAL
from helpers.constants import VERWEILDAUER
from helpers.constants import EINTRITTSALTER


def compareListOfStrings(list1, list2):
    cnt_1notin2 = 0;
    for el in list1:
        if not el in list2:
            cnt_1notin2 += 1;
            #print('element from list 1 not in list 2: ' + str(el))

    cnt_2notin1 = 0;
    for el in list2:
        if not el in list1:
            cnt_2notin1 += 1;
            #print('element from list 2 not in list 1: ' + str(el))

    print('number of elements from list 1 not in list 2: ' + str(cnt_1notin2))
    print('number of elements from list 2 not in list 1: ' + str(cnt_2notin1))

    commonheaders = [];
    for el1 in list1:
        if el1 in list2:
            commonheaders.append(el1);

    print('num common elements: ' + str(len(commonheaders)))
    return commonheaders;


def makeColumnsBinary(data, headers):

    for k,header in enumerate(headers):
        if header.startswith('CHOP_') or header.startswith('DK_'):
            col = data[:,k];
            b = col >= 1;
            col_bin = b.astype(int);
            data[:,k] = col_bin;
            # print(np.max(col_bin))

    return data;


def getFilenameStrDataset(strDataset, preprocessing, filterKey=None, filterValue=None, extraStr=None):

    strFilename = ''

    if filterKey is not None:
        strFilename = preprocessing + '_' + strDataset + '_' + str(filterKey) + '_' + str(filterValue);
    else:
        strFilename = preprocessing + '_' + strDataset;

    if extraStr is not None:
        strFilename = strFilename + '_' + extraStr;

    return strFilename;


def addFilenameOptions(suffixFilename, clf_name, clf_options, extra_options):
    strFilename = suffixFilename;
    if clf_options is not None:
        strFilename = clf_options + '_' + strFilename;

    strFilename = clf_name + '_' + strFilename;

    if extra_options is not None:
        strFilename = strFilename + '_' + extra_options;
    return strFilename;


def getFilenameStrTraining(strFilenameData, clf_name, clf_options=None, extra_options=None):
    strTraining = strFilenameData;
    strTraining = addFilenameOptions(strTraining, clf_name, clf_options, extra_options);
    return strTraining;


def getFilenameStrResults(strFilenameTraining, strFilenameTesting, clf_name, clf_options=None, extra_options=None):
    strResults = 'training_' + strFilenameTraining + '_testing_' + strFilenameTesting;
    strResults = addFilenameOptions(strResults, clf_name, clf_options, extra_options);
    return strResults;


def removeIgnoreHeaders(headers):
    headers_filtered = [];
    for header in headers:
        if not header in IGNORE_HEADERS:
            headers_filtered.append(header);
        else:
            print('column name ' + str(header) + ' is part of the IGNORE_HEADERS....');
    return headers_filtered;


def getCHOPgrouping():
    group_headers = [];
    for k in range(0, 100):  # number of chop groups
        group_headers.append(str(k).zfill(2));
    return group_headers.copy() ;

def getDKgrouping():
    group_headers = [];
    for k in range(0, 26):  # number of characters in the alphabet --> number of dk groups
        group_headers.append(string.ascii_uppercase[k]);
    return group_headers.copy();

def getDKlightGrouping():
    group_headers = [];
    for k in range(0, 26):  # number of characters in the alphabet --> number of dk groups
        for l in range(0, 10):
            group_headers.append(string.ascii_uppercase[k] + str(l));
    return group_headers.copy();

def getOEgrouping():
    return OE_str.copy();

def getDRGgrouping():
    group_headers = [];
    for k in range(0, 26):  # number of characters in the alphabet --> number of dk groups
        group_headers.append(string.ascii_uppercase[k]);
    group_headers.append('9');
    return group_headers.copy();

def getMonthValues():
    return MONTHS.copy();

def getYearValues():
    return YEARS.copy();

def getDayValues():
    return DAYS.copy();

def getGeschlechtValues():
    return GESCHLECHT.copy()

def getForschungskonsentValues():
    return FORSCHUNGSKONSENTE.copy()

def getVersicherungsklasseValues():
    return VERSICHERUNGSKLASSEN.copy();

def getEntlassBereichValues():
    return ENTLASSBEREICHE.copy();

def getEntlassartValues():
    return ENTLASSART.copy();

def getAufnahmeartValues():
    return AUFNAHMEART.copy();

def getLiegestatusValues():
    return LIEGESTATUS.copy();

def getEintrittsartValues():
    return EINTRITTSART.copy();


def getFeaturesToCategorize():
    features = CATEGORICAL_BINARY + CATEGORICAL_ONEHOT;
    return features.copy();


def getCountFeaturesToBinarize():
    features = NUMERICAL;
    return features.copy();


def getAdminFeaturesNames():
    return ADMIN_FEATURES_NAMES.copy();


def getVerweildauerCategories():
    return VERWEILDAUER.copy();


def getAlterCategories():
    return EINTRITTSALTER.copy();


def getLOSState(valStr):
    val = int(valStr)
    if val < 3:
        ind=0;
    elif val >= 3 and val < 10:
        ind = 1;
    elif val >= 10 and val < 30:
        ind = 2;
    elif val >= 30 and val < 70:
        ind = 3;
    elif val >= 70 and val < 150:
        ind = 4;
    elif val >= 150:
        ind = 5;
    else:
        print('LOS: this should not happen...exit')
        sys.exit();
    return VERWEILDAUER[ind];


def getFeatureCategories(name):

    if name == 'Liegestatus':
        return getLiegestatusValues();
    elif name == 'EntlassBereich':
        return getEntlassBereichValues();
    elif name == 'Versicherungsklasse':
        return getVersicherungsklasseValues();
    elif name == 'Eintrittsart':
        return getEintrittsartValues();
    elif name == 'Entlassart':
        return getEntlassartValues();
    elif name == 'Forschungskonsent':
        return getForschungskonsentValues();
    elif name == 'Geschlecht':
        return getForschungskonsentValues();
    elif name.endswith('tag'):
        return getDayValues();
    elif name.endswith('monat'):
        return getMonthValues();
    elif name.endswith('jahr'):
        return getYearValues();
    elif name == 'Aufnahmeart':
        return getAufnahmeartValues();
    else:
        print('feature is not known...exit: ' + str(name))
        sys.exit();


def getAgeState(valStr):
    val = int(valStr)
    if val < 30:
        ind = 0;
    elif val >= 30 and val < 60:
        ind = 1;
    elif val >= 60 and val < 90:
        ind = 2;
    elif val >= 90:
        ind = 3;
    else:
        print('AGE: this should not happen...exit')
        sys.exit();
    return EINTRITTSALTER[ind];


def convertDate(datestring):
    d = datetime.datetime.strptime(datestring, '%d.%m.%Y');
    tstamp = (d.replace(tzinfo=None) - datetime.datetime(1970, 1, 1)).total_seconds()
    return tstamp

