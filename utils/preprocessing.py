import pandas as pd
import numpy as np
from datetime import datetime

# file path settings
base_path = '/Users/jiyounkim/mimic-iii-clinical-database-1.4'
icu_stay_path = str(base_path) + '/ICUSTAYS.csv'
admissions_path = str(base_path) + '/ADMISSIONS.csv'
chart_events_path = str(base_path) + '/CHARTEVENTS.csv'

# columns needed from icu stay csv
icu_keys = ['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS']
icu_dates = ['INTIME', 'OUTTIME']

# columns needed from admissions csv
admissions_keys = ["HADM_ID", "ADMITTIME", "DEATHTIME", "ADMISSION_LOCATION",
                   "INSURANCE", "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY",
                   "DIAGNOSIS","HAS_CHARTEVENTS_DATA"]
admissions_dates = ["ADMITTIME","DEATHTIME"]

# columns needed from chartevents csv
chart_events_keys = ['ICUSTAY_ID', 'ITEMID', 'VALUENUM', 'CHARTTIME']
chart_events_dates = ['CHARTTIME']

# read icu stay csv, and save into dataframe
icu_stays = pd.read_csv(icu_stay_path, usecols=icu_keys, parse_dates=icu_dates)
# print("ICU Stays shape: ", icu_stays.shape)

# only keep icu stays with 24 hrs <= length-of-stay <= 48 hrs
icu_stays = icu_stays.loc[(icu_stays['LOS'] >= 1) & (icu_stays['LOS'] <= 2)]
# print("ICU Stays shape: ", icu_stays.shape)

# read admissions csv, and save into dataframe
admissions = pd.read_csv(admissions_path, usecols=admissions_keys, parse_dates=admissions_dates)
# print("Admissions shape: ", admissions.shape)

# left join icu_stays, admissions data tables, based on HADM_ID
icu_adm = pd.merge(icu_stays, admissions, on="HADM_ID", how="left")
# print("ICU Adm shape: ", icu_adm.shape)

# only keep rows that have chart events data
icu_adm = icu_adm.loc[(icu_adm["HAS_CHARTEVENTS_DATA"] == 1)]
print(icu_adm.shape)
# show detailed dataframe info without abbreviation
pd.set_option('display.max_columns', None)

# initialize chartevents column with empty array for each row
icu_adm['CHARTEVENTS'] = np.empty((len(icu_adm), 0)).tolist()

# set label to 1 if 1) death time exists 2) death time is between in time and out time of icu stay, else label 0
icu_adm['LABEL'] = np.where(((pd.notna(icu_adm['DEATHTIME'])) & (icu_adm['INTIME'] <= icu_adm['DEATHTIME']) & (icu_adm['DEATHTIME'] <= icu_adm['OUTTIME'])), 1, 0)

# convert icu stay dataframe to dictionary format
icu_adm_dict = icu_adm.to_dict('records')

# read chart events csv in chunks, and append chart events to matching icu stay id (3hrs from in_time)
chunk_size = 10000000
with pd.read_csv(chart_events_path, usecols=chart_events_keys, parse_dates=chart_events_dates, chunksize=chunk_size) as reader:
    count = 0
    for chunk in reader:
        s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("[", s, "] START READING CHARTEVENTS CHUNK ", count)

        # convert null to None (valuenum)
        chunk = chunk.replace({np.nan: None})
        # create 'ITEMID', 'VALUENUM', 'CHARTTIME' of each row as single list, and add into new column 'CHARTEVENTS'
        chunk['CHARTEVENTS'] = chunk[chunk.columns[1:]].apply(lambda x: list(x), axis=1)
        # drop previous 'ITEMID', 'VALUENUM', 'CHARTTIME' columns
        chunk.drop(['ITEMID', 'CHARTTIME', 'VALUENUM'], axis=1, inplace=True)
        # group chunk dataframe by ICUSTAY_ID, and concat all chart events data into single list
        chunk = chunk.groupby('ICUSTAY_ID')['CHARTEVENTS'].apply(list).reset_index(name='CHARTEVENTS')
        # convert dataframe to dictionary format
        chunk_dict = chunk.to_dict('records')

        # for each ICUSTAY_ID in chart events chunk
        for i in chunk_dict:
            icu_id = i['ICUSTAY_ID']
            r = 0
            # loop through icu stays dataframe
            for j in icu_adm_dict:
                # if match exists between chunk ICUSTAY_ID and icu stays dataframe ICUSTAY_ID
                if j['ICUSTAY_ID'] == icu_id:
                    # if chart events of icu stay is equal to 100, break
                    if len(j['CHARTEVENTS']) == 100:
                        r += 1
                        break

                    # access in time of icu stay
                    in_time = j['INTIME']

                    # for each chunk CHARTEVENTS data
                    for k in i['CHARTEVENTS']:
                        # if chart event time is within 3 hours after in time
                        if (k[1] >= in_time) & ((k[1] - in_time)/np.timedelta64(1, 'h') <= 3):
                            # modify timestamp to (chart event time - in time) in minutes
                            k[1] = (k[1] - in_time)/np.timedelta64(1, 'm')
                            # append icu stays dataframe CHARTEVENTS list with chunk CHARTEVENTS data
                            j['CHARTEVENTS'].append([k[1], k[0], k[2]])
                        # if chart events of icu stay length is equal to 100, break
                        if len(j['CHARTEVENTS']) == 100:
                            r += 1
                            break

        s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("[", s, "] FINISHED READING CHARTEVENTS CHUNK ", count)
        count += 1


# save icu stays with chart events to csv file
icu_chart_events = pd.DataFrame.from_records(icu_adm_dict)
print(icu_chart_events.head())
icu_chart_events.to_csv("../data/icu_with_chart_events.csv", header=True, index=False)




