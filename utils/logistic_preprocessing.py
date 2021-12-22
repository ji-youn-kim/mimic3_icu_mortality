import pandas as pd
import numpy as np
from ast import literal_eval

# configure path to save numpy files
x_train_npy_path = "../data/X_train_logistic.npy"
y_train_npy_path = "../data/y_train.npy"
x_test_npy_path = "../data/X_test_logistic.npy"
y_test_npy_path = "../data/y_test.npy"

# read icu chart events csv, and convert into dataframe
icu_chart_events_path = "../data/icu_with_chart_events.csv"
icu_chart_keys = ['ICUSTAY_ID', 'LOS', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', \
                  'ETHNICITY', 'DIAGNOSIS', 'CHARTEVENTS', 'LABEL']
icu_chart_events = pd.read_csv(icu_chart_events_path, usecols=icu_chart_keys)

# one hot encode categorical data
icu_chart_events = pd.get_dummies(icu_chart_events, columns=['ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', \
                                                             'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS'], drop_first=True)

# display df without abbreviation
pd.set_option('display.max_columns', None)

item_total_list = []
value_total_list = []
timestamp_total_list = []
unique_item_list = []
# store all item_id / value_num / timestamp values per each dataframe row
for count, row in icu_chart_events.iterrows():
    chart_events = literal_eval(row['CHARTEVENTS'])
    item_id_list = []
    value_num_dict = {}
    value_num_count_dict = {}
    timestamp_dict = {}
    timestamp_count_dict = {}
    for event in chart_events:
        item_id = event[1]
        timestamp = event[0]
        value_num = event[2]
        cur_value_num = value_num if value_num else 0
        if item_id not in unique_item_list:
            unique_item_list.append(item_id)
        if item_id not in item_id_list:
            item_id_list.append(item_id)
            value_num_dict[item_id] = cur_value_num
            value_num_count_dict[item_id] = 1
            timestamp_dict[item_id] = timestamp
            timestamp_count_dict[item_id] = 1
        else:
            value_num_dict[item_id] = value_num_dict[item_id] + cur_value_num
            timestamp_dict[item_id] = timestamp_dict[item_id] + timestamp
            value_num_count_dict[item_id] += 1
            timestamp_count_dict[item_id] += 1

    # store average value_num if multiple values exist per item_id
    for key in value_num_dict:
        value_num_dict[key] = value_num_dict[key]/value_num_count_dict[key]
    # store average timestamp if multiple values exist per item_id
    for key in timestamp_dict:
        timestamp_dict[key] = timestamp_dict[key]/timestamp_count_dict[key]

    # store current item_id / value_num / timestamp lists
    item_total_list.append(item_id_list)
    value_total_list.append(value_num_dict)
    timestamp_total_list.append(timestamp_dict)

# sort list containing all item_ids
unique_item_list.sort()

unique_value_list = []
unique_time_list = []
# create value_num / timestamp columns per total item_ids
for i in unique_item_list:
    value_column = str(i) + '_VALNUM'
    time_column = str(i) + '_TIME'
    unique_value_list.append(value_column)
    unique_time_list.append(time_column)

unique_list = unique_item_list + unique_value_list + unique_time_list
# initialize item_id / value_num / timestamp column values to zero
data = np.zeros(shape=(len(icu_chart_events), len(unique_list)))
icu_chart_events_add = pd.DataFrame(data=data, columns=unique_list)
# add item_id / value_num / timestamp columns to icu_chart_events dataframe
icu_chart_events = pd.concat([icu_chart_events, icu_chart_events_add], axis=1)
# drop chart events column
icu_chart_events = icu_chart_events.drop(columns='CHARTEVENTS')

# for each row in dataframe, set item_id value = 1, value_num, timestamp
for index, row in icu_chart_events.iterrows():
    for item_id in item_total_list[index]:
        row[item_id] = 1
        row[str(item_id) + '_VALNUM'] = value_total_list[index][item_id]
        row[str(item_id) + '_TIME'] = timestamp_total_list[index][item_id]

# if icu stay id is 8, 9 -> test data
train_data = icu_chart_events[(icu_chart_events['ICUSTAY_ID'] % 10 != 8) & (icu_chart_events['ICUSTAY_ID'] % 10 != 9)]
test_data = icu_chart_events[(icu_chart_events['ICUSTAY_ID'] % 10 == 8) | (icu_chart_events['ICUSTAY_ID'] % 10 == 9)]

print("ICU CHART EVENTS SHAPE: ", icu_chart_events.shape)
print("TRAIN DATA SHAPE: ", train_data.shape)
print("TEST DATA SHAPE: ", test_data.shape)

# convert train, test data to numpy - x, y
x_train = train_data.drop(columns='LABEL').to_numpy()
y_train = train_data[['LABEL']].to_numpy()
x_test = test_data.drop(columns='LABEL').to_numpy()
y_test = test_data[['LABEL']].to_numpy()

print("X_TRAIN SHAPE: ", x_train.shape)
print("Y_TRAIN SHAPE: ", y_train.shape)
print("X_TEST SHAPE: ", x_test.shape)
print("Y_TEST SHAPE: ", y_test.shape)

# save train, test numpy data to npy file
np.save(x_train_npy_path, x_train)
np.save(y_train_npy_path, y_train)
np.save(x_test_npy_path, x_test)
np.save(y_test_npy_path, y_test)