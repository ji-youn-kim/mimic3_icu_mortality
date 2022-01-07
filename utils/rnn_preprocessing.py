import numpy as np
from sklearn.preprocessing import StandardScaler
from rnn_word2id import *

# configure path to save numpy files
x_train_npy_path = "../data/X_train_rnn_indv.npy"
x_test_npy_path = "../data/X_test_rnn_indv.npy"
y_train_npy_path = "../data/y_train_rem_nochev.npy"
y_test_npy_path = "../data/y_test_rem_nochev.npy"

# set scaler for scaling numeric features
standardScaler = StandardScaler()

# display df without abbreviation
pd.set_option('display.max_columns', None)

train_los, train_admit, train_insurance, train_lang, train_religion, train_marital, train_ethnicity, \
    train_diagnosis, train_gender, train_item_id, train_value_num, train_no_pad_len = ([] for i in range(12))

test_los, test_admit, test_insurance, test_lang, test_religion, test_marital, test_ethnicity, \
    test_diagnosis, test_gender, test_item_id, test_value_num, test_no_pad_len = ([] for i in range(12))

x_train, x_test, y_train, y_test = ([] for i in range(4))

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
    chart_events = row['CHARTEVENTS']
    label = row['LABEL']
    chart_event_len = len(chart_events)
    lack_chart_events = 100 - chart_event_len
    event_item, event_value = ([] for i in range(2))

    for event in chart_events:
        item_id = event[1]
        value_num = event[2]
        timestamp = event[0]
        event_item.append(item_ids_dict[item_id])
        event_value.append(value_num)

    # zero padding on chart events data if less than 100
    for i in range(0, lack_chart_events):
        event_item.append(item_ids_dict['unknown'])
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
        train_no_pad_len.append(chart_event_len)
        y_train.append([label])
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
        test_no_pad_len.append(chart_event_len)
        y_test.append([label])

# scale los, value_num data for train, test data individually
train_los = standardScaler.fit_transform(np.array(train_los).reshape(-1, 1))
train_value_num = standardScaler.fit_transform(train_value_num)
test_los = standardScaler.fit_transform(np.array(test_los).reshape(-1, 1))
test_value_num = standardScaler.fit_transform(test_value_num)

print("train_value_num")
print(train_value_num)
print("test_value_num")
print(test_value_num)

train_los = train_los.flatten().tolist()
train_value_num = train_value_num.tolist()
test_los = test_los.flatten().tolist()
test_value_num = test_value_num.tolist()

print(len(train_los), len(train_admit), len(train_insurance), len(train_lang), len(train_religion), len(train_marital), \
      len(train_ethnicity), len(train_diagnosis), len(train_gender), len(train_item_id), len(train_value_num))

print(len(test_los), len(test_admit), len(test_insurance), len(test_lang), len(test_religion), len(test_marital), \
      len(test_ethnicity), len(test_diagnosis), len(test_gender), len(test_item_id), len(test_value_num))

# append all test data to single list
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
x_train.append(train_no_pad_len)

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
x_test.append(test_no_pad_len)

x_train = np.array(x_train, dtype=object)
x_test = np.array(x_test, dtype=object)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(len(x_train), len(x_test), len(y_train), len(y_test))
print(len(train_no_pad_len))
print(len(test_no_pad_len))
print(x_train[0][:20])
print(x_train[2][:20])
print(x_train[-3][:2])
print(x_train[-2][:2])
print(x_train[-1][:2])
print(y_train[:10])

print(x_test[0][:20])
print(x_test[2][:20])
print(x_test[-3][:2])
print(x_test[-2][:2])
print(x_test[-1][:2])
print(y_test[:10])

# save train, test numpy data to npy file
np.save(x_train_npy_path, x_train)
np.save(x_test_npy_path, x_test)
np.save(y_train_npy_path, y_train)
np.save(y_test_npy_path, y_test)
