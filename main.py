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

# convert icu adm to dictionary
# icu_adm_dict = icu_adm.to_dict('records')
#
# print(len(icu_adm_dict))

icu_adm['label'] = np.where(((pd.notna(icu_adm['DEATHTIME'])) & (icu_adm['INTIME'] <= icu_adm['DEATHTIME']) & (icu_adm['DEATHTIME'] <= icu_adm['OUTTIME'])), 1, 0)
# print(icu_adm.head(50))

icu_adm_dict = icu_adm.to_dict('records')
# print(icu_adm_dict[:10])

chunk_size = 5000000
with pd.read_csv(chart_events_path, usecols=chart_events_keys, parse_dates=chart_events_dates, chunksize=chunk_size) as reader:
    for chunk in reader:
        s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("[", s, "] START READING CHARTEVENTS CHUNK")
        chunk['ITEM_TIME_VALUE'] = chunk[chunk.columns[1:]].apply(lambda x: list(x), axis=1)
        # print(chunk.head())
        chunk.drop(['ITEMID', 'CHARTTIME', 'VALUENUM'], axis=1, inplace=True)
        # chunk['ICUSTAY_ID'] = pd.to_numeric(chunk['ICUSTAY_ID'])
        chunk = chunk.groupby('ICUSTAY_ID')['ITEM_TIME_VALUE'].apply(list).reset_index(name='ITEM_TIME_VALUE')
        # print(chunk.head())
        chunk_dict = chunk.to_dict('records')
        # print(chunk_dict[:2])
        for i in chunk_dict:
            icu_id = i['ICUSTAY_ID']
            for j in icu_adm_dict:
                if j['ICUSTAY_ID'] == icu_id:
                    if len(i['ITEM_TIME_VALUE']) > 100:
                        j['CHARTEVENTS'] = i['ITEM_TIME_VALUE'][:100]
                    else:
                        j['CHARTEVENTS'] = i['ITEM_TIME_VALUE']
                    # print(j)
                    # print(len(i['ITEM_TIME_VALUE']))
                    # print(len(j['CHARTEVENTS']))
        # icu_adm = pd.merge(icu_adm, chunk, on="ICUSTAY_ID", how="left")
        # print(icu_adm.head())
        s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("[", s, "] FINISHED READING CHARTEVENTS CHUNK")

print(icu_adm.shape)
print(icu_adm.head())


