import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler

# configure path to save numpy files
x_train_npy_path = "../data/X_train_rnn_indv.npy"
x_test_npy_path = "../data/X_test_rnn_indv.npy"

# read icu chart events csv, and convert into dataframe
icu_chart_events_path = "../data/icu_with_chart_events_v_not_nan.csv"
icu_chart_keys = ['ICUSTAY_ID', 'LOS', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', \
                  'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'CHARTEVENTS', 'LABEL']
icu_chart_events = pd.read_csv(icu_chart_events_path, usecols=icu_chart_keys)

# fill na with values to construct word2id
icu_chart_events['MARITAL_STATUS'].fillna('UNKNOWN (DEFAULT)', inplace=True)
icu_chart_events.fillna('unknown', inplace=True)

# set scaler for scaling numeric features
standardScaler = StandardScaler()

# display df without abbreviation
pd.set_option('display.max_columns', None)

# print(icu_chart_events.head())

# set word2id for icu stay categorical features
admit_loc_dict = {'unknown': 0}
insurance_dict = {'unknown': 0}
lang_dict = {'unknown': 0}
religion_dict = {'UNOBTAINABLE': 0}
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
    chart_events = literal_eval(row['CHARTEVENTS'])
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

train_los, train_admit, train_insurance, train_lang, train_religion, train_marital, train_ethnicity, \
    train_diagnosis, train_gender, train_item_id, train_value_num = ([] for i in range(11))

test_los, test_admit, test_insurance, test_lang, test_religion, test_marital, test_ethnicity, \
    test_diagnosis, test_gender, test_item_id, test_value_num = ([] for i in range(11))

# create individual lists for features in icu stay row order
for count, row in icu_chart_events.iterrows():
    icu_stay_id = row['ICUSTAY_ID']
    los = row['LOS']
    admit_loc = row['ADMISSION_LOCATION']
    insurance = row['INSURANCE']
    lang = row['LANGUAGE']
    religion = row['RELIGION']
    marital = row['MARITAL_STATUS']
    ethnicity = row['ETHNICITY']
    diagnosis = row['DIAGNOSIS']
    gender = row['GENDER']
    chart_events = literal_eval(row['CHARTEVENTS'])
    lack_chart_events = 100 - len(chart_events)
    event_item, event_value = ([] for i in range(2))

    for event in chart_events:
        item_id = event[1]
        value_num = event[2]
        timestamp = event[0]
        event_item.append(item_id)
        event_value.append(value_num)

    # zero padding on chart events data if less than 100
    for i in range(0, lack_chart_events):
        event_item.append(0)
        event_value.append(0)

    # store feature lists to train data if icu stay id does not end with 8 or 9
    if (icu_stay_id % 10 != 8) & (icu_stay_id % 10 != 9):
        train_los.append(los)
        train_admit.append(admit_loc_dict[admit_loc])
        train_insurance.append(insurance_dict[insurance])
        train_lang.append(lang_dict[lang])
        train_religion.append(religion_dict[religion])
        train_marital.append(marital_dict[marital])
        train_ethnicity.append(ethnicity_dict[ethnicity])
        train_diagnosis.append(diagnosis_dict[diagnosis])
        train_gender.append(gender_dict[gender])
        train_item_id.append(event_item)
        train_value_num.append(event_value)
    # store feature lists to test data if icu stay id ends with 8 or 9
    else:
        test_los.append(los)
        test_admit.append(admit_loc_dict[admit_loc])
        test_insurance.append(insurance_dict[insurance])
        test_lang.append(lang_dict[lang])
        test_religion.append(religion_dict[religion])
        test_marital.append(marital_dict[marital])
        test_ethnicity.append(ethnicity_dict[ethnicity])
        test_diagnosis.append(diagnosis_dict[diagnosis])
        test_gender.append(gender_dict[gender])
        test_item_id.append(event_item)
        test_value_num.append(event_value)


# scale los, value_num data for train, test data individually
train_los = standardScaler.fit_transform(np.array(train_los).reshape(-1, 1))
train_value_num = standardScaler.fit_transform(train_value_num)
test_los = standardScaler.fit_transform(np.array(test_los).reshape(-1, 1))
test_value_num = standardScaler.fit_transform(test_value_num)

train_los = train_los.flatten().tolist()
train_value_num = train_value_num.tolist()
test_los = test_los.flatten().tolist()
test_value_num = test_value_num.tolist()

print(len(train_los), len(train_admit), len(train_insurance), len(train_lang), len(train_religion), len(train_marital), \
      len(train_ethnicity), len(train_diagnosis), len(train_gender), len(train_item_id), len(train_value_num))

print(len(test_los), len(test_admit), len(test_insurance), len(test_lang), len(test_religion), len(test_marital), \
      len(test_ethnicity), len(test_diagnosis), len(test_gender), len(test_item_id), len(test_value_num))

# append all test data to single list
x_train, x_test = ([] for i in range(2))
x_train.append(train_los)
x_train.append(train_admit)
x_train.append(train_insurance)
x_train.append(train_lang)
x_train.append(train_religion)
x_train.append(train_marital)
x_train.append(train_ethnicity)
x_train.append(train_diagnosis)
x_train.append(train_gender)
x_train.append(train_item_id)
x_train.append(train_value_num)

# append all train data to single list
x_test.append(test_los)
x_test.append(test_admit)
x_test.append(test_insurance)
x_test.append(test_lang)
x_test.append(test_religion)
x_test.append(test_marital)
x_test.append(test_ethnicity)
x_test.append(test_diagnosis)
x_test.append(test_gender)
x_test.append(test_item_id)
x_test.append(test_value_num)

x_train = np.array(x_train, dtype=object)
x_test = np.array(x_test, dtype=object)

print(len(x_train), len(x_test))
print(x_train[0][:20])
print(x_train[2][:20])
print(x_train[-2][:2])
print(x_train[-1][:2])

print(x_test[0][:20])
print(x_test[2][:20])
print(x_test[-2][:2])
print(x_test[-1][:2])

# save train, test numpy data to npy file
np.save(x_train_npy_path, x_train)
np.save(x_test_npy_path, x_test)
