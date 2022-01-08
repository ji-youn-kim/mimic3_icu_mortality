import pandas as pd
from ast import literal_eval

# read icu chart events csv, and convert into dataframe
icu_chart_events_path = "../data/icu_with_chart_events_v_not_nan.csv"
icu_chart_keys = ['ICUSTAY_ID', 'LOS', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', \
                  'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'CHARTEVENTS', 'LABEL']
icu_chart_events = pd.read_csv(icu_chart_events_path, usecols=icu_chart_keys)

# fill na with values to construct word2id
icu_chart_events['MARITAL_STATUS'].fillna('UNKNOWN (DEFAULT)', inplace=True)
icu_chart_events.fillna('unknown', inplace=True)
# convert label type from int to float
icu_chart_events['LABEL'] = icu_chart_events['LABEL'].astype(float)

# display df without abbreviation
pd.set_option('display.max_columns', None)
# print(icu_chart_events.head(200))

icu_chart_events['CHARTEVENTS'] = icu_chart_events['CHARTEVENTS'].apply(literal_eval)

print("before empty chart event rows removal: ", len(icu_chart_events))
count = 0
for index, row in icu_chart_events.iterrows():
    if len(row['CHARTEVENTS']) == 0:
        # print(index, row)
        count += 1
print("count: ", count)
# remove icu stay rows with no chart events
icu_chart_events.drop(icu_chart_events[icu_chart_events['CHARTEVENTS'].map(len) == 0].index, inplace=True)
print("after empty chart event rows removal:", len(icu_chart_events))

# set word2id for icu stay categorical features
admit_loc_dict = {'unknown': 0}
insurance_dict = {'unknown': 0}
lang_dict = {'unknown': 0}
religion_dict = {'unknown': 0}
marital_dict = {'UNKNOWN (DEFAULT)': 0}
ethnicity_dict = {'UNKNOWN/NOT SPECIFIED': 0}
diagnosis_dict = {'unknown': 0}
gender_dict = {'unknown': 0}
item_ids_dict = {'unknown': 0}
unique_item_list = []

# loop through df and create word2id for categorical features
for count, row in icu_chart_events.iterrows():
    admit_loc = row['ADMISSION_LOCATION']
    insurance = row['INSURANCE']
    lang = row['LANGUAGE']
    religion = row['RELIGION']
    marital = row['MARITAL_STATUS']
    ethnicity = row['ETHNICITY']
    diagnosis = row['DIAGNOSIS']
    gender = row['GENDER']
    chart_events = row['CHARTEVENTS']
    if admit_loc not in admit_loc_dict:
        admit_loc_dict[admit_loc] = len(admit_loc_dict)
    if insurance not in insurance_dict:
        insurance_dict[insurance] = len(insurance_dict)
    if lang not in lang_dict:
        lang_dict[lang] = len(lang_dict)
    if religion not in religion_dict:
        religion_dict[religion] = len(religion_dict)
    if marital not in marital_dict:
        marital_dict[marital] = len(marital_dict)
    if ethnicity not in ethnicity_dict:
        ethnicity_dict[ethnicity] = len(ethnicity_dict)
    if diagnosis not in diagnosis_dict:
        diagnosis_dict[diagnosis] = len(diagnosis_dict)
    if gender not in gender_dict:
        gender_dict[gender] = len(gender_dict)
    for event in chart_events:
        item_id = event[1]
        if item_id not in unique_item_list:
            unique_item_list.append(item_id)

unique_item_list.sort()
j = 1
for i in unique_item_list:
    item_ids_dict[i] = j
    j += 1

admit_dict_len = len(admit_loc_dict)
insurance_dict_len = len(insurance_dict)
lang_dict_len = len(lang_dict)
religion_dict_len = len(religion_dict)
marital_dict_len = len(marital_dict)
ethnicity_dict_len = len(ethnicity_dict)
diagnosis_dict_len = len(diagnosis_dict)
gender_dict_len = len(gender_dict)
item_ids_dict_len = len(item_ids_dict)

general_dict = {}
for key in item_ids_dict:
    if key == "unknown":
        general_dict[len(general_dict)] = 'unknown_item_id'
    else:
        general_dict[len(general_dict)] = key

general_dict[len(general_dict)] = 'value_num'
general_dict[len(general_dict)] = 'los'

for key in admit_loc_dict:
    if key == 'unknown':
        general_dict[len(general_dict)] = 'unknown_admit'
    else:
        general_dict[len(general_dict)] = key
for key in insurance_dict:
    if key == 'unknown':
        general_dict[len(general_dict)] = 'unknown_insurance'
    else:
        general_dict[len(general_dict)] = key
for key in lang_dict:
    if key == "unknown":
        general_dict[len(general_dict)] = 'unknown_lang'
    else:
        general_dict[len(general_dict)] = key
for key in religion_dict:
    if key == "unknown":
        general_dict[len(general_dict)] = 'unknown_religion'
    else:
        general_dict[len(general_dict)] = key
for key in marital_dict:
    if key == "UNKNOWN (DEFAULT)":
        general_dict[len(general_dict)] = 'unknown_marital'
    else:
        general_dict[len(general_dict)] = key
for key in ethnicity_dict:
    if key == "UNKNOWN/NOT SPECIFIED":
        general_dict[len(general_dict)] = 'unknown_ethnicity'
    else:
        general_dict[len(general_dict)] = key
for key in diagnosis_dict:
    if key == "unknown":
        general_dict[len(general_dict)] = 'unknown_diagnosis'
    else:
        general_dict[len(general_dict)] = key
for key in gender_dict:
    if key == "unknown":
        general_dict[len(general_dict)] = 'unknown_gender'
    else:
        general_dict[len(general_dict)] = key

# print("general_dict: ", general_dict)
# print("len(general_dict): ", len(general_dict))
