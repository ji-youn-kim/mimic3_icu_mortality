import pandas as pd
import numpy as np
from csv import reader
from datetime import datetime

# 파일 경로 설정
base_path = '/Users/jiyounkim/mimic-iii-clinical-database-1.4'
icu_stay_path = str(base_path) + '/ICUSTAYS.csv'
admissions_path = str(base_path) + '/ADMISSIONS.csv'
chart_events_path = str(base_path) + '/CHARTEVENTS.csv'
diagnoses_icd_path = str(base_path) + '/DIAGNOSES_ICD.csv'
d_icd_diagnoses_path = str(base_path) + '/D_ICD_DIAGNOSES.csv'

# icu stay csv 에서 필요한 column 지정
icu_keys = ['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS']
icu_dates = ['INTIME', 'OUTTIME']

# admissions csv 에서 필요한 column 지정
admissions_keys = ["HADM_ID", "ADMITTIME", "DEATHTIME", "ADMISSION_LOCATION",
                   "INSURANCE", "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY",
                   "DIAGNOSIS","HAS_CHARTEVENTS_DATA"]
admissions_dates = ["ADMITTIME","DEATHTIME"]

# chartevents csv 에서 필요한 column 지정
chart_events_keys = ['ICUSTAY_ID', 'ITEMID', 'VALUENUM', 'CHARTTIME']
chart_events_dates = ['CHARTTIME']

# diagnosis_icd csv 에서 필요한 column 지정
# diagnoses_icd_keys = ["HADM_ID","SEQ_NUM","ICD9_CODE"]

# d_icd_diagnoses csv 에서 필요한 column 지정
# d_icd_diagnoses_keys = ["ICD9_CODE","SHORT_TITLE"]

# .to_dict('records')
# icu stay csv 파일 읽어서 dataframe 으로 저장
icu_stays = pd.read_csv(icu_stay_path, usecols=icu_keys, parse_dates=icu_dates)
# print("ICU Stays shape: ", icu_stays.shape)

# length-of-stay 24 시간 이상 48 시간 이하인 icu stay 만 저장
icu_stays = icu_stays.loc[(icu_stays['LOS'] >= 1) & (icu_stays['LOS'] <= 2)]
# print("ICU Stays shape: ", icu_stays.shape)

# admissions csv 파일 읽어서 dataframe 으로 저장
admissions = pd.read_csv(admissions_path, usecols=admissions_keys, parse_dates=admissions_dates)
# print("Admissions shape: ", admissions.shape)

# diagnosis_icd csv 파일 읽어서 dataframe 으로 저장
# diagnoses_icd = pd.read_csv(diagnoses_icd_path, usecols=diagnoses_icd_keys)
# print(diagnoses_icd.shape)

# d_icd_diagnoses csv 파일 읽어서 dataframe 으로 저장
# d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_path, usecols=d_icd_diagnoses_keys)
# print(d_icd_diagnoses.shape)

# icu_stays, admissions 데이터테이블을 left join 으로 HADM_ID 기준 병합
icu_adm = pd.merge(icu_stays, admissions, on="HADM_ID", how="left")
# print("ICU Adm shape: ", icu_adm.shape)

icu_adm = icu_adm.loc[(icu_adm["HAS_CHARTEVENTS_DATA"] == 1)]
print(icu_adm.shape)
pd.set_option('display.max_columns', None)
# print(icu_adm.head())

# init chartevents with empty array
icu_adm['CHARTEVENTS'] = np.empty((len(icu_adm), 0)).tolist()

# set label to 1 if deathtime is not null, else label 0
icu_adm['LABEL'] = np.where(pd.isnull(icu_adm['DEATHTIME']), 0, 1)

# convert icu adm to dictionary
icu_adm_dict = icu_adm.to_dict('records')

print(len(icu_adm_dict))

s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("[", s, "] START READING CHARTEVENTS")
# open and read chart events csv
with open(chart_events_path, 'r') as read_chart:
    chart_reader = reader(read_chart)
    # skip chart events header and read from next line
    header = next(chart_reader)
    # print("chart events length: ", len(list(chart_reader)))
    i = 0
    for chart_row in chart_reader:
        # print(chart_row)
        if chart_row[3] and chart_row[4] and chart_row[9] and chart_row[5]:
            icustay_id = int(chart_row[3])
            itemid = int(chart_row[4])
            valuenum = float(chart_row[9])
            charttime = chart_row[5]
            # loop through icu_adm_dict
            # print(icustay_id)
            for icu_adm_row in icu_adm_dict:
                # print(icu_adm_row)
                # if icustay id matches with chart events, chartevents data is less than 100
                if icu_adm_row['ICUSTAY_ID'] == icustay_id and (len(icu_adm_row['CHARTEVENTS']) < 100):
                    # append the chart events data to icu_adm_row
                    icu_adm_row['CHARTEVENTS'].append([itemid, valuenum, charttime])
                    break
            if i == 10000:
                break
            i += 1

s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("[", s, "] FINISHED READING CHARTEVENTS")

print(len(icu_adm_dict))
# print(icu_adm_dict[:10])

temp_dict = []

for row in icu_adm_dict:
    if row['CHARTEVENTS']:
        temp_dict.append(row)
        print(row['ICUSTAY_ID'], " chart event length: ", len(row['CHARTEVENTS']))

print(temp_dict)

