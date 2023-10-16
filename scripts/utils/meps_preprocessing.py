"""Preprocessing of the MEPS dataset.

Copied from https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/

Essentially we wanna make the MEPS prediction task more difficult, since
currently all models seem to achieve very similar results.
"""

import pandas as pd

custom_mappings = {
    'label_maps': [{1.0: '>= 10 Visits', 0.0: '< 10 Visits'}],
    'protected_attribute_maps': [{0.0: 'White', 1.0: 'Non-White', 2.0: 'Hispanic'}]
}


def _original_race_preproc(row):
    if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
        return 'White'
    return 'Non-White'


def _custom_race_preproc(row):
    if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
        return 0
        # return 'White'
    elif (row['HISPANX'] == 1):
        return 2
        # return 'Hispanic'
    else:
        return 1
        # return 'Non-White'


def custom_meps_preproc(df, panel_num: int):
    """
    1. Create a new column, RACE
    2. Restrict to Panel `panel_num`
    3. RENAME all columns that are PANEL/ROUND SPECIFIC
    4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
    5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
    """
    assert panel_num in {19, 20, 21}
    panel_year = {
        19: 15,     # 2015
        20: 15,     # 2015
        21: 16,     # 2016
    }[panel_num]

    df['RACEV2X'] = df.apply(lambda row: _custom_race_preproc(row), axis=1)
    df = df.rename(columns = {'RACEV2X' : 'RACE'})

    df = df[df['PANEL'] == panel_num]

    # RENAME COLUMNS
    df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              f'POVCAT{panel_year}' : 'POVCAT', f'INSCOV{panel_year}' : 'INSCOV'})

    df = df[df['REGION'] >= 0] # remove values -1
    df = df[df['AGE'] >= 0] # remove values -1

    df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

    df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

    df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

    def utilization(row):
        return row[f'OBTOTV{panel_year}'] + row[f'OPTOTV{panel_year}'] + row[f'ERTOT{panel_year}'] + row[f'IPNGTD{panel_year}'] + row[f'HHTOTD{panel_year}']

    df[f'TOTEXP{panel_year}'] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df[f'TOTEXP{panel_year}'] < 10.0
    df.loc[lessE,f'TOTEXP{panel_year}'] = 0.0
    moreE = df[f'TOTEXP{panel_year}'] >= 10.0
    df.loc[moreE,f'TOTEXP{panel_year}'] = 1.0

    df = df.rename(columns = {f'TOTEXP{panel_year}' : 'UTILIZATION'})
    return df
