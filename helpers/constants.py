import sys

NUM_DAYS_READMISSION = 18;

INDEXCOLORS = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"];


PLOTCOLORS = ['b', 'r', 'g', 'y', 'm', 'c', 'k'];

YEARS = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
YEARS_INT = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
GESCHLECHT = ['weiblich', 'maennlich'];
FORSCHUNGSKONSENTE = ['ein', 'unb'];
VERSICHERUNGSKLASSEN = ['A', 'S', 'P', 'H'];
ENTLASSBEREICHE = ['SaO', 'Med', 'Gyn', 'Oth', 'N.A.'];
ENTLASSART = ['iniDri', 'exPat', 'gSpit', 'vSpit', 'sSpit','Plan', 'inPat', 'iniBeh'];
AUFNAHMEART = ['V: Voranmeldung', 'AE:Aus sonst.Sp', 'SO: Spender-OP', 'A: Einweisung', 'G: Entbindung', 'AT:ATS Transpla',
               'NO: Notfall', 'WS: Wissenschaf', 'AV:Aus Vertr.Sp', 'NE Neue.Neueint', 'WE:Wiedereintr.',
               'N1:Nierenstein1', 'N2:Nierenstein2', 'S: Selbsteinw.']
LIEGESTATUS = ['kurz', 'norm', 'lang', 'vap', 'opti', 'unb'];
EINTRITTSART = ['Ver', 'Not', 'Ang', 'Geb', 'Int', 'unb'];
VERWEILDAUER = ['sehrkurz', 'kurz', 'mittel', 'mittellang', 'lang', 'sehrlang'];
EINTRITTSALTER = ['jung', 'mittel', 'mittelalt', 'alt'];

# IGNORE_HEADERS = ['Fall', 'EintrittsartNotfall', 'Fallart', 'DKSepsis_', 'DKSepsis_0', 'OEIntensiv_', 'OEIntensiv_0', 'HauptdiagnoseText', 'DRGCode'];
IGNORE_HEADERS = ['EintrittsartNotfall', 'Fallart', 'DKSepsis_', 'DKSepsis_0', 'OEIntensiv_', 'OEIntensiv_0', 'HauptdiagnoseText'];
TO_CATEGORICAL = ['Aufnahmeart', 'Entlassart', 'EntlassBereich', 'Versicherungsklasse', 'Forschungskonsent', 'Entlassmonat', 'Aufnahmemonat', 'Aufnahmetag', 'Entlasstag', 'Verweildauer', 'Eintrittsalter', 'Entlassjahr', 'Aufnahmejahr', 'Liegestatus']

CATEGORICAL = ['Aufnahmeart', 'Entlassart', 'Eintrittsart', 'EntlassBereich', 'Versicherungsklasse', 'Forschungskonsent', 'Entlassmonat', 'Aufnahmemonat', 'Aufnahmetag', 'Entlasstag', 'Entlassjahr', 'Aufnahmejahr', 'Liegestatus']
BINARY = ['Geschlecht', 'Langlieger', 'DKSepsis_1', 'OEIntensiv_1'];
SPARSE = ['Hauptdiagnose', 'AufnehmOE', 'EntlassOE', 'DRGCode', 'CHOP', 'DK', 'OE'];
TO_REMOVE = ['EntlassartVerstorben']
PREPROCESSING = ['Patient', 'Aufnahmedatum', 'Entlassdatum']

NUMERICAL = ['Verweildauer', 'Eintrittsalter'];
CATEGORICAL_ONEHOT = ['Aufnahmeart', 'Entlassart', 'Eintrittsart', 'EntlassBereich', 'Versicherungsklasse', 'Entlassmonat', 'Aufnahmemonat', 'Aufnahmetag', 'Entlasstag', 'Entlassjahr', 'Aufnahmejahr', 'Liegestatus', 'Hauptdiagnose', 'AufnehmOE', 'EntlassOE', 'DRGCode'];
CATEGORICAL_BINARY = ['Geschlecht', 'Forschungskonsent'];
ADMIN_FEATURES_NAMES = ['Fall', 'Aufnahmeart', 'Aufnahmedatum', 'Entlassdatum', 'Aufnahmejahr', 'Entlassjahr', 'Aufnahmemonat', 'Entlassmonat', 'Aufnahmetag', 'Entlasstag', 'Wiederkehrer', 'Eintrittsalter', 'Eintrittsart', 'Entlassart', 'EntlassBereich', 'Versicherungsklasse', 'Patient', 'Forschungskonsent', 'Geschlecht', 'Verweildauer', 'Liegestatus', 'Langlieger']


OE_20122015 = [152,153,320,323,324,325,326,327,328,330,331,332,336,337,342,348,352,353,354,356,357,358,361,362,363,365,368,371,373,376,380,392,393,751,752,753,777];
OE_str = [str(item) for item in OE_20122015];
MDC = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', '9'];