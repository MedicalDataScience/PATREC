
import helpers.helpers as helpers


diag_group_names = helpers.getDKverylightGrouping();

def getListOfCodesFromStartAndEndCode(start,end):
    ind_start = diag_group_names.index(start);
    ind_end = diag_group_names.index(end);
    subgroup_list = diag_group_names[ind_start:ind_end+1];
    return subgroup_list;

icd10_chapters = dict();
icd10_chapters['A00-B99'] = dict();
icd10_chapters['A00-B99']['A00-A09'] = getListOfCodesFromStartAndEndCode('A00','A09');
icd10_chapters['A00-B99']['A15-A19'] = getListOfCodesFromStartAndEndCode('A15','A19');
icd10_chapters['A00-B99']['A20-A28'] = getListOfCodesFromStartAndEndCode('A20','A28');
icd10_chapters['A00-B99']['A30-A49'] = getListOfCodesFromStartAndEndCode('A30','A49');
icd10_chapters['A00-B99']['A50-A64'] = getListOfCodesFromStartAndEndCode('A50','A64');
icd10_chapters['A00-B99']['A65-A69'] = getListOfCodesFromStartAndEndCode('A65','A69');
icd10_chapters['A00-B99']['A70-A74'] = getListOfCodesFromStartAndEndCode('A70','A74');
icd10_chapters['A00-B99']['A75-A79'] = getListOfCodesFromStartAndEndCode('A75','A79');
icd10_chapters['A00-B99']['A80-A89'] = getListOfCodesFromStartAndEndCode('A80','A89');
icd10_chapters['A00-B99']['A92-A99'] = getListOfCodesFromStartAndEndCode('A92','A99');
icd10_chapters['A00-B99']['B00-B09'] = getListOfCodesFromStartAndEndCode('B00','B09');
icd10_chapters['A00-B99']['B15-B19'] = getListOfCodesFromStartAndEndCode('B15','B19');
icd10_chapters['A00-B99']['B20-B24'] = getListOfCodesFromStartAndEndCode('B20','B24');
icd10_chapters['A00-B99']['B25-B34'] = getListOfCodesFromStartAndEndCode('B25','B34');
icd10_chapters['A00-B99']['B35-B49'] = getListOfCodesFromStartAndEndCode('B35','B49');
icd10_chapters['A00-B99']['B50-B64'] = getListOfCodesFromStartAndEndCode('B50','B64');
icd10_chapters['A00-B99']['B65-B83'] = getListOfCodesFromStartAndEndCode('B65','B83');
icd10_chapters['A00-B99']['B85-B89'] = getListOfCodesFromStartAndEndCode('B85','B89');
icd10_chapters['A00-B99']['B90-B94'] = getListOfCodesFromStartAndEndCode('B90','B94');
icd10_chapters['A00-B99']['B95-B98'] = getListOfCodesFromStartAndEndCode('B95','B98');
icd10_chapters['A00-B99']['B99'] = getListOfCodesFromStartAndEndCode('B99','B99');

icd10_chapters['C00-D48'] = dict();
icd10_chapters['C00-D48']['C00-C97'] = getListOfCodesFromStartAndEndCode('C00','C97');
icd10_chapters['C00-D48']['D00-D09'] = getListOfCodesFromStartAndEndCode('D00','D09');
icd10_chapters['C00-D48']['D10-D36'] = getListOfCodesFromStartAndEndCode('D10','D36');
icd10_chapters['C00-D48']['D37-D48'] = getListOfCodesFromStartAndEndCode('D37','D48');

icd10_chapters['D50-D90'] = dict();
icd10_chapters['D50-D90']['D50-D53'] = getListOfCodesFromStartAndEndCode('D50','D53');
icd10_chapters['D50-D90']['D55-D59'] = getListOfCodesFromStartAndEndCode('D55','D59');
icd10_chapters['D50-D90']['D60-D64'] = getListOfCodesFromStartAndEndCode('D60','D64');
icd10_chapters['D50-D90']['D65-D69'] = getListOfCodesFromStartAndEndCode('D65','D69');
icd10_chapters['D50-D90']['D70-D77'] = getListOfCodesFromStartAndEndCode('D70','D77');
icd10_chapters['D50-D90']['D80-D90'] = getListOfCodesFromStartAndEndCode('D80','D90');

icd10_chapters['E00-E90'] = dict();
icd10_chapters['E00-E90']['E00-E07'] = getListOfCodesFromStartAndEndCode('E00','E07');
icd10_chapters['E00-E90']['E10-E14'] = getListOfCodesFromStartAndEndCode('E10','E14');
icd10_chapters['E00-E90']['E15-E16'] = getListOfCodesFromStartAndEndCode('E15','E16');
icd10_chapters['E00-E90']['E20-E35'] = getListOfCodesFromStartAndEndCode('E20','E35');
icd10_chapters['E00-E90']['E40-E46'] = getListOfCodesFromStartAndEndCode('E40','E46');
icd10_chapters['E00-E90']['E50-E64'] = getListOfCodesFromStartAndEndCode('E50','E64');
icd10_chapters['E00-E90']['E65-E68'] = getListOfCodesFromStartAndEndCode('E65','E68');
icd10_chapters['E00-E90']['E70-E90'] = getListOfCodesFromStartAndEndCode('E70','E90');

icd10_chapters['F00-F99'] = dict();
icd10_chapters['F00-F99']['F00-F09'] = getListOfCodesFromStartAndEndCode('F00','F09');
icd10_chapters['F00-F99']['F10-F19'] = getListOfCodesFromStartAndEndCode('F10','F19');
icd10_chapters['F00-F99']['F20-F29'] = getListOfCodesFromStartAndEndCode('F20','F29');
icd10_chapters['F00-F99']['F30-F39'] = getListOfCodesFromStartAndEndCode('F30','F39');
icd10_chapters['F00-F99']['F40-F48'] = getListOfCodesFromStartAndEndCode('F40','F48');
icd10_chapters['F00-F99']['F50-F59'] = getListOfCodesFromStartAndEndCode('F50','F59');
icd10_chapters['F00-F99']['F60-F69'] = getListOfCodesFromStartAndEndCode('F60','F69');
icd10_chapters['F00-F99']['F70-F79'] = getListOfCodesFromStartAndEndCode('F70','F79');
icd10_chapters['F00-F99']['F80-F89'] = getListOfCodesFromStartAndEndCode('F80','F89');
icd10_chapters['F00-F99']['F90-F98'] = getListOfCodesFromStartAndEndCode('F90','F98');
icd10_chapters['F00-F99']['F99'] = getListOfCodesFromStartAndEndCode('F99','F99');

icd10_chapters['G00-G99'] = dict();
icd10_chapters['G00-G99']['G00-G09'] = getListOfCodesFromStartAndEndCode('G00','G09');
icd10_chapters['G00-G99']['G10-G14'] = getListOfCodesFromStartAndEndCode('G10','G14');
icd10_chapters['G00-G99']['G20-G26'] = getListOfCodesFromStartAndEndCode('G20','G26');
icd10_chapters['G00-G99']['G30-G32'] = getListOfCodesFromStartAndEndCode('G30','G32');
icd10_chapters['G00-G99']['G35-G37'] = getListOfCodesFromStartAndEndCode('G35','G37');
icd10_chapters['G00-G99']['G40-G47'] = getListOfCodesFromStartAndEndCode('G40','G47');
icd10_chapters['G00-G99']['G50-G59'] = getListOfCodesFromStartAndEndCode('G50','G59');
icd10_chapters['G00-G99']['G60-G64'] = getListOfCodesFromStartAndEndCode('G60','G64');
icd10_chapters['G00-G99']['G70-G73'] = getListOfCodesFromStartAndEndCode('G70','G73');
icd10_chapters['G00-G99']['G80-G83'] = getListOfCodesFromStartAndEndCode('G80','G83');
icd10_chapters['G00-G99']['G90-G99'] = getListOfCodesFromStartAndEndCode('G90','G99');

icd10_chapters['H00-H59'] = dict();
icd10_chapters['H00-H59']['H00-H06'] = getListOfCodesFromStartAndEndCode('H00','H06');
icd10_chapters['H00-H59']['H10-H13'] = getListOfCodesFromStartAndEndCode('H10','H13');
icd10_chapters['H00-H59']['H15-H22'] = getListOfCodesFromStartAndEndCode('H15','H22');
icd10_chapters['H00-H59']['H25-H28'] = getListOfCodesFromStartAndEndCode('H25','H28');
icd10_chapters['H00-H59']['H30-H36'] = getListOfCodesFromStartAndEndCode('H30','H36');
icd10_chapters['H00-H59']['H40-H42'] = getListOfCodesFromStartAndEndCode('H40','H42');
icd10_chapters['H00-H59']['H43-H45'] = getListOfCodesFromStartAndEndCode('H43','H45');
icd10_chapters['H00-H59']['H46-H48'] = getListOfCodesFromStartAndEndCode('H46','H48');
icd10_chapters['H00-H59']['H49-H52'] = getListOfCodesFromStartAndEndCode('H49','H52');
icd10_chapters['H00-H59']['H53-H54'] = getListOfCodesFromStartAndEndCode('H53','H54');
icd10_chapters['H00-H59']['H55-H59'] = getListOfCodesFromStartAndEndCode('H55','H59');

icd10_chapters['H60-H95'] = dict();
icd10_chapters['H60-H95']['H60-H62'] = getListOfCodesFromStartAndEndCode('H60','H62');
icd10_chapters['H60-H95']['H65-H75'] = getListOfCodesFromStartAndEndCode('H65','H75');
icd10_chapters['H60-H95']['H80-H83'] = getListOfCodesFromStartAndEndCode('H80','H83');
icd10_chapters['H60-H95']['H90-H95'] = getListOfCodesFromStartAndEndCode('H90','H95');

icd10_chapters['I00-I99'] = dict();
icd10_chapters['I00-I99']['I00-I02'] = getListOfCodesFromStartAndEndCode('I00','I02');
icd10_chapters['I00-I99']['I05-I09'] = getListOfCodesFromStartAndEndCode('I05','I09');
icd10_chapters['I00-I99']['I10-I15'] = getListOfCodesFromStartAndEndCode('I10','I15');
icd10_chapters['I00-I99']['I20-I25'] = getListOfCodesFromStartAndEndCode('I20','I25');
icd10_chapters['I00-I99']['I26-I28'] = getListOfCodesFromStartAndEndCode('I26','I28');
icd10_chapters['I00-I99']['I30-I52'] = getListOfCodesFromStartAndEndCode('I30','I52');
icd10_chapters['I00-I99']['I60-I69'] = getListOfCodesFromStartAndEndCode('I60','I69');
icd10_chapters['I00-I99']['I70-I79'] = getListOfCodesFromStartAndEndCode('I70','I79');
icd10_chapters['I00-I99']['I80-I89'] = getListOfCodesFromStartAndEndCode('I80','I89');
icd10_chapters['I00-I99']['I95-I99'] = getListOfCodesFromStartAndEndCode('I95','I99');

icd10_chapters['J00-J99'] = dict();
icd10_chapters['J00-J99']['J00-J06'] = getListOfCodesFromStartAndEndCode('J00','J06');
icd10_chapters['J00-J99']['J09-J18'] = getListOfCodesFromStartAndEndCode('J09','J18');
icd10_chapters['J00-J99']['J20-J22'] = getListOfCodesFromStartAndEndCode('J20','J22');
icd10_chapters['J00-J99']['J30-J39'] = getListOfCodesFromStartAndEndCode('J30','J39');
icd10_chapters['J00-J99']['J40-J47'] = getListOfCodesFromStartAndEndCode('J40','J47');
icd10_chapters['J00-J99']['J60-J70'] = getListOfCodesFromStartAndEndCode('J60','J70');
icd10_chapters['J00-J99']['J80-J84'] = getListOfCodesFromStartAndEndCode('J80','J84');
icd10_chapters['J00-J99']['J85-J86'] = getListOfCodesFromStartAndEndCode('J85','J86');
icd10_chapters['J00-J99']['J90-J94'] = getListOfCodesFromStartAndEndCode('J90','J94');
icd10_chapters['J00-J99']['J95-J99'] = getListOfCodesFromStartAndEndCode('J95','J99');

icd10_chapters['K00-K93'] = dict();
icd10_chapters['K00-K93']['K00-K14'] = getListOfCodesFromStartAndEndCode('K00','K14');
icd10_chapters['K00-K93']['K20-K31'] = getListOfCodesFromStartAndEndCode('K20','K31');
icd10_chapters['K00-K93']['K35-K38'] = getListOfCodesFromStartAndEndCode('K35','K38');
icd10_chapters['K00-K93']['K40-K46'] = getListOfCodesFromStartAndEndCode('K40','K46');
icd10_chapters['K00-K93']['K50-K52'] = getListOfCodesFromStartAndEndCode('K50','K52');
icd10_chapters['K00-K93']['K55-K64'] = getListOfCodesFromStartAndEndCode('K55','K64');
icd10_chapters['K00-K93']['K55-K67'] = getListOfCodesFromStartAndEndCode('K65','K67');
icd10_chapters['K00-K93']['K70-K77'] = getListOfCodesFromStartAndEndCode('K70','K77');
icd10_chapters['K00-K93']['K80-K87'] = getListOfCodesFromStartAndEndCode('K80','K87');
icd10_chapters['K00-K93']['K90-K93'] = getListOfCodesFromStartAndEndCode('K90','K93');

icd10_chapters['L00-L99'] = dict();
icd10_chapters['L00-L99']['L00-L08'] = getListOfCodesFromStartAndEndCode('L00','L08');
icd10_chapters['L00-L99']['L10-L14'] = getListOfCodesFromStartAndEndCode('L10','L14');
icd10_chapters['L00-L99']['L20-L30'] = getListOfCodesFromStartAndEndCode('L20','L30');
icd10_chapters['L00-L99']['L40-L45'] = getListOfCodesFromStartAndEndCode('L40','L45');
icd10_chapters['L00-L99']['L50-L54'] = getListOfCodesFromStartAndEndCode('L50','L54');
icd10_chapters['L00-L99']['L55-L59'] = getListOfCodesFromStartAndEndCode('L55','L59');
icd10_chapters['L00-L99']['L60-L75'] = getListOfCodesFromStartAndEndCode('L60','L75');
icd10_chapters['L00-L99']['L80-L99'] = getListOfCodesFromStartAndEndCode('L80','L99');

icd10_chapters['M00-M99'] = dict();
icd10_chapters['M00-M99']['M00-M25'] = getListOfCodesFromStartAndEndCode('M00','M25');
icd10_chapters['M00-M99']['M30-M36'] = getListOfCodesFromStartAndEndCode('M30','M36');
icd10_chapters['M00-M99']['M40-M54'] = getListOfCodesFromStartAndEndCode('M40','M54');
icd10_chapters['M00-M99']['M60-M79'] = getListOfCodesFromStartAndEndCode('M60','M79');
icd10_chapters['M00-M99']['M80-M94'] = getListOfCodesFromStartAndEndCode('M80','M94');
icd10_chapters['M00-M99']['M95-M99'] = getListOfCodesFromStartAndEndCode('M95','M99');

icd10_chapters['N00-N99'] = dict();
icd10_chapters['N00-N99']['N00-N08'] = getListOfCodesFromStartAndEndCode('N00','N08');
icd10_chapters['N00-N99']['N10-N16'] = getListOfCodesFromStartAndEndCode('N10','N16');
icd10_chapters['N00-N99']['N17-N19'] = getListOfCodesFromStartAndEndCode('N17','N19');
icd10_chapters['N00-N99']['N20-N23'] = getListOfCodesFromStartAndEndCode('N20','N23');
icd10_chapters['N00-N99']['N25-N29'] = getListOfCodesFromStartAndEndCode('N25','N29');
icd10_chapters['N00-N99']['N30-N39'] = getListOfCodesFromStartAndEndCode('N30','N39');
icd10_chapters['N00-N99']['N40-N51'] = getListOfCodesFromStartAndEndCode('N40','N51');
icd10_chapters['N00-N99']['N60-N64'] = getListOfCodesFromStartAndEndCode('N60','N64');
icd10_chapters['N00-N99']['N70-N77'] = getListOfCodesFromStartAndEndCode('N70','N77');
icd10_chapters['N00-N99']['N80-N98'] = getListOfCodesFromStartAndEndCode('N80','N98');
icd10_chapters['N00-N99']['N99'] = getListOfCodesFromStartAndEndCode('N99','N99');

icd10_chapters['O00-O99'] = dict();
icd10_chapters['O00-O99']['O00-O08'] = getListOfCodesFromStartAndEndCode('O00','O08');
icd10_chapters['O00-O99']['O09-O09'] = getListOfCodesFromStartAndEndCode('O09','O09');
icd10_chapters['O00-O99']['O10-O16'] = getListOfCodesFromStartAndEndCode('O10','O16');
icd10_chapters['O00-O99']['O20-O29'] = getListOfCodesFromStartAndEndCode('O20','O29');
icd10_chapters['O00-O99']['O30-O48'] = getListOfCodesFromStartAndEndCode('O30','O48');
icd10_chapters['O00-O99']['O60-O75'] = getListOfCodesFromStartAndEndCode('O60','O75');
icd10_chapters['O00-O99']['O80-O82'] = getListOfCodesFromStartAndEndCode('O80','O82');
icd10_chapters['O00-O99']['O85-O92'] = getListOfCodesFromStartAndEndCode('O85','O92');
icd10_chapters['O00-O99']['O94-O99'] = getListOfCodesFromStartAndEndCode('O94','O99');

icd10_chapters['P00-P96'] = dict();
icd10_chapters['P00-P96']['P00-P04'] = getListOfCodesFromStartAndEndCode('P00','P04');
icd10_chapters['P00-P96']['P05-P08'] = getListOfCodesFromStartAndEndCode('P05','P08');
icd10_chapters['P00-P96']['P10-P15'] = getListOfCodesFromStartAndEndCode('P10','P15');
icd10_chapters['P00-P96']['P20-P29'] = getListOfCodesFromStartAndEndCode('P20','P29');
icd10_chapters['P00-P96']['P35-P39'] = getListOfCodesFromStartAndEndCode('P35','P39');
icd10_chapters['P00-P96']['P50-P61'] = getListOfCodesFromStartAndEndCode('P50','P61');
icd10_chapters['P00-P96']['P70-P74'] = getListOfCodesFromStartAndEndCode('P70','P74');
icd10_chapters['P00-P96']['P75-P78'] = getListOfCodesFromStartAndEndCode('P75','P78');
icd10_chapters['P00-P96']['P80-P83'] = getListOfCodesFromStartAndEndCode('P80','P83');
icd10_chapters['P00-P96']['P90-P96'] = getListOfCodesFromStartAndEndCode('P90','P96');

icd10_chapters['Q00-Q99'] = dict();
icd10_chapters['Q00-Q99']['Q00-Q07'] = getListOfCodesFromStartAndEndCode('Q00','Q07');
icd10_chapters['Q00-Q99']['Q10-Q18'] = getListOfCodesFromStartAndEndCode('Q10','Q18');
icd10_chapters['Q00-Q99']['Q20-Q28'] = getListOfCodesFromStartAndEndCode('Q20','Q28');
icd10_chapters['Q00-Q99']['Q30-Q34'] = getListOfCodesFromStartAndEndCode('Q30','Q34');
icd10_chapters['Q00-Q99']['Q35-Q37'] = getListOfCodesFromStartAndEndCode('Q35','Q37');
icd10_chapters['Q00-Q99']['Q38-Q45'] = getListOfCodesFromStartAndEndCode('Q38','Q45');
icd10_chapters['Q00-Q99']['Q50-Q56'] = getListOfCodesFromStartAndEndCode('Q50','Q56');
icd10_chapters['Q00-Q99']['Q60-Q64'] = getListOfCodesFromStartAndEndCode('Q60','Q64');
icd10_chapters['Q00-Q99']['Q65-Q79'] = getListOfCodesFromStartAndEndCode('Q65','Q79');
icd10_chapters['Q00-Q99']['Q80-Q89'] = getListOfCodesFromStartAndEndCode('Q80','Q89');
icd10_chapters['Q00-Q99']['Q90-Q99'] = getListOfCodesFromStartAndEndCode('Q90','Q99');

icd10_chapters['R00-R99'] = dict();
icd10_chapters['R00-R99']['R00-R09'] = getListOfCodesFromStartAndEndCode('R00','R09');
icd10_chapters['R00-R99']['R10-R19'] = getListOfCodesFromStartAndEndCode('R10','R19');
icd10_chapters['R00-R99']['R20-R23'] = getListOfCodesFromStartAndEndCode('R20','R23');
icd10_chapters['R00-R99']['R25-R29'] = getListOfCodesFromStartAndEndCode('R25','R29');
icd10_chapters['R00-R99']['R30-R39'] = getListOfCodesFromStartAndEndCode('R30','R39');
icd10_chapters['R00-R99']['R40-R46'] = getListOfCodesFromStartAndEndCode('R40','R46');
icd10_chapters['R00-R99']['R47-R49'] = getListOfCodesFromStartAndEndCode('R47','R49');
icd10_chapters['R00-R99']['R50-R69'] = getListOfCodesFromStartAndEndCode('R50','R69');
icd10_chapters['R00-R99']['R70-R79'] = getListOfCodesFromStartAndEndCode('R70','R79');
icd10_chapters['R00-R99']['R80-R82'] = getListOfCodesFromStartAndEndCode('R80','R82');
icd10_chapters['R00-R99']['R83-R89'] = getListOfCodesFromStartAndEndCode('R83','R89');
icd10_chapters['R00-R99']['R90-R94'] = getListOfCodesFromStartAndEndCode('R90','R94');
icd10_chapters['R00-R99']['R95-R99'] = getListOfCodesFromStartAndEndCode('R95','R99');

icd10_chapters['S00-T98'] = dict();
icd10_chapters['S00-T98']['S00-S09'] = getListOfCodesFromStartAndEndCode('S00','S09');
icd10_chapters['S00-T98']['S10-S19'] = getListOfCodesFromStartAndEndCode('S10','S19');
icd10_chapters['S00-T98']['S20-S29'] = getListOfCodesFromStartAndEndCode('S20','S29');
icd10_chapters['S00-T98']['S30-S39'] = getListOfCodesFromStartAndEndCode('S30','S39');
icd10_chapters['S00-T98']['S40-S49'] = getListOfCodesFromStartAndEndCode('S40','S49');
icd10_chapters['S00-T98']['S50-S59'] = getListOfCodesFromStartAndEndCode('S50','S59');
icd10_chapters['S00-T98']['S60-S69'] = getListOfCodesFromStartAndEndCode('S60','S69');
icd10_chapters['S00-T98']['S70-S79'] = getListOfCodesFromStartAndEndCode('S70','S79');
icd10_chapters['S00-T98']['S80-S89'] = getListOfCodesFromStartAndEndCode('S80','S89');
icd10_chapters['S00-T98']['S90-S99'] = getListOfCodesFromStartAndEndCode('S90','S99');
icd10_chapters['S00-T98']['T00-T07'] = getListOfCodesFromStartAndEndCode('T00','T07');
icd10_chapters['S00-T98']['T08-T14'] = getListOfCodesFromStartAndEndCode('T08','T14');
icd10_chapters['S00-T98']['T15-T19'] = getListOfCodesFromStartAndEndCode('T15','T19');
icd10_chapters['S00-T98']['T20-T32'] = getListOfCodesFromStartAndEndCode('T20','T32');
icd10_chapters['S00-T98']['T33-T35'] = getListOfCodesFromStartAndEndCode('T33','T35');
icd10_chapters['S00-T98']['T36-T50'] = getListOfCodesFromStartAndEndCode('T36','T50');
icd10_chapters['S00-T98']['T51-T65'] = getListOfCodesFromStartAndEndCode('T51','T65');
icd10_chapters['S00-T98']['T66-T78'] = getListOfCodesFromStartAndEndCode('T66','T78');
icd10_chapters['S00-T98']['T79'] = getListOfCodesFromStartAndEndCode('T79','T79');
icd10_chapters['S00-T98']['T80-T88'] = getListOfCodesFromStartAndEndCode('T80','T88');
icd10_chapters['S00-T98']['T89'] = getListOfCodesFromStartAndEndCode('T89','T89');
icd10_chapters['S00-T98']['T90-T98'] = getListOfCodesFromStartAndEndCode('T90','T98');

icd10_chapters['V01-Y84'] = dict();
#icd10_chapters['V01-Y84']['V01-X59'] = ['V99','W49','W64','W87','W91','W92','W93','W94',
 #                                      'X19','X29','X49','X59'];
icd10_chapters['V01-Y84']['V01-X59'] = getListOfCodesFromStartAndEndCode('V01','X59');
icd10_chapters['V01-Y84']['X60-X84'] = getListOfCodesFromStartAndEndCode('X60','X84');
icd10_chapters['V01-Y84']['X85-Y09'] = getListOfCodesFromStartAndEndCode('X85','Y09');
icd10_chapters['V01-Y84']['Y10-Y34'] = getListOfCodesFromStartAndEndCode('Y10','Y34');
icd10_chapters['V01-Y84']['Y35-Y36'] = getListOfCodesFromStartAndEndCode('Y35','Y36');
icd10_chapters['V01-Y84']['Y40-Y84'] = getListOfCodesFromStartAndEndCode('Y40','Y84');

icd10_chapters['Z00-Z99'] = dict();
icd10_chapters['Z00-Z99']['Z00-Z13'] = getListOfCodesFromStartAndEndCode('Z00','Z13');
icd10_chapters['Z00-Z99']['Z20-Z29'] = getListOfCodesFromStartAndEndCode('Z20','Z29');
icd10_chapters['Z00-Z99']['Z30-Z39'] = getListOfCodesFromStartAndEndCode('Z30','Z39');
icd10_chapters['Z00-Z99']['Z40-Z54'] = getListOfCodesFromStartAndEndCode('Z40','Z54');
icd10_chapters['Z00-Z99']['Z55-Z65'] = getListOfCodesFromStartAndEndCode('Z55','Z65');
icd10_chapters['Z00-Z99']['Z70-Z76'] = getListOfCodesFromStartAndEndCode('Z70','Z76');
icd10_chapters['Z00-Z99']['Z80-Z99'] = getListOfCodesFromStartAndEndCode('Z80','Z99');


def getNumbersSubgroups(group_name):
    num_subgroups = len(icd10_chapters[group_name].keys());
    return num_subgroups;

def getSubgroups(group_name):
    name_subgroups = icd10_chapters[group_name].keys();
    return name_subgroups;

def getCodesSubgroup(group_name, subgroup_name):
    codes = icd10_chapters[group_name][subgroup_name];
    return codes;

def getMainGroups():
    return list(icd10_chapters.keys());


def getCodesMainGroup(maingroup):
    codes = [];
    for key in icd10_chapters[maingroup].keys():
        codes = codes + list(icd10_chapters[maingroup][key]);
    return codes;





if __name__ == '__main__':

    name = 'S00-T98';
    getSubgroups(name);
    getNumbersSubgroups(name);

