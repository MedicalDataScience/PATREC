import sys
import os
import string
import datetime

OE_20122015 = [152,153,320,323,324,325,326,327,328,330,331,332,336,337,342,348,352,353,354,356,357,358,361,362,363,365,368,371,373,376,380,392,393,751,752,753,777];
OE_str = [str(item) for item in OE_20122015];

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

def getDRGgrouping():
    group_headers = [];
    for k in range(0, 26):  # number of characters in the alphabet --> number of dk groups
        group_headers.append(string.ascii_uppercase[k]);
    group_headers.append('9');
    return group_headers.copy();

def getOEgrouping():
    return OE_str.copy();


def convertDate(datestring):
    d = datetime.datetime.strptime(datestring, '%d.%m.%Y');
    tstamp = (d.replace(tzinfo=None) - datetime.datetime(1970, 1, 1)).total_seconds()
    return tstamp

