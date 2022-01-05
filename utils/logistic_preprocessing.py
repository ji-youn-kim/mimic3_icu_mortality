import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import logistic_options
op = logistic_options.Options()

# configure path to save numpy files
x_train_npy_path = "../data/X_train_logistic.npy"
y_train_npy_path = "../data/y_train.npy"
x_test_npy_path = "../data/X_test_logistic.npy"
y_test_npy_path = "../data/y_test.npy"

# read icu chart events csv, and convert into dataframe
icu_chart_events_path = "../data/icu_with_chart_events.csv"
icu_chart_keys = ['ICUSTAY_ID', 'LOS', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', \
                  'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'CHARTEVENTS', 'LABEL']
icu_chart_events = pd.read_csv(icu_chart_events_path, usecols=icu_chart_keys)

# one hot encode categorical data
icu_chart_events = pd.get_dummies(icu_chart_events, columns=['ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', \
                                                             'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS', 'GENDER'], drop_first=True)

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
    timestamp_dict = {}
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
        timestamp_dict[item_id] = timestamp

    # store current item_id / value_num / timestamp lists
    item_total_list.append(item_id_list)
    value_total_list.append(value_num_dict)
    timestamp_total_list.append(timestamp_dict)

# sort list containing all item_ids
unique_item_list.sort()

# create item_id2id dictionary
unique_item_dict = {}
for idx, item in enumerate(unique_item_list):
    unique_item_dict[item] = idx

# shape: (icu_stay, len(unique_item_dict))
value_scaler_list = [[0.0 for i in range(len(unique_item_dict))] for idx, icu_item_list in enumerate(item_total_list)]
time_scaler_list = [[0.0 for i in range(len(unique_item_dict))] for idx, icu_item_list in enumerate(item_total_list)]

# add value_num to item_id index
for index, val_list in enumerate(value_scaler_list):
    val_dict = value_total_list[index]
    for idx, key in enumerate(val_dict):
        # item value_num for item_id in val_dict
        item_val = val_dict[key]
        value_scaler_list[index][unique_item_dict[key]] = item_val

# add timestamp to item_id index
for index, val_list in enumerate(time_scaler_list):
    time_dict = timestamp_total_list[index]
    for idx, key in enumerate(time_dict):
        # item timestamp for item_id in val_dict
        item_time = time_dict[key]
        time_scaler_list[index][unique_item_dict[key]] = item_time

# scale value_num, timestamp, los
if op.apply_scaler:
    if op.scaler_type == "MinMax":
        scaler = MinMaxScaler()
    if op.scaler_type == "Standard":
        scaler = StandardScaler()
    # train_los = standardScaler.fit_transform(np.array(train_los).reshape(-1, 1))
    value_scaler_list = scaler.fit_transform(value_scaler_list).tolist()
    time_scaler_list = scaler.fit_transform(time_scaler_list).tolist()
    df_los = icu_chart_events['LOS']
    los_scaled = scaler.fit_transform(df_los.to_numpy().reshape(-1, 1))
    los_scaled = pd.DataFrame(los_scaled, columns=['LOS'])
    icu_chart_events['LOS'] = los_scaled

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
# convert label type from int to float
icu_chart_events['LABEL'] = icu_chart_events['LABEL'].astype(float)

# for each row in dataframe, set item_id value = 1, value_num, timestamp
for index, row in icu_chart_events.iterrows():
    for item_id in item_total_list[index]:
        row[item_id] = 1
        row[str(item_id) + '_VALNUM'] = value_scaler_list[index][unique_item_dict[item_id]]
        row[str(item_id) + '_TIME'] = time_scaler_list[index][unique_item_dict[item_id]]

# if icu stay id is 8, 9 -> test data
train_data = icu_chart_events[(icu_chart_events['ICUSTAY_ID'] % 10 != 8) & (icu_chart_events['ICUSTAY_ID'] % 10 != 9)]
test_data = icu_chart_events[(icu_chart_events['ICUSTAY_ID'] % 10 == 8) | (icu_chart_events['ICUSTAY_ID'] % 10 == 9)]

print("ICU CHART EVENTS SHAPE: ", icu_chart_events.shape)
print("TRAIN DATA SHAPE: ", train_data.shape)
print("TEST DATA SHAPE: ", test_data.shape)

# convert train, test data to numpy - x, y
x_train = train_data.drop(columns=['LABEL', 'ICUSTAY_ID']).to_numpy()
y_train = train_data[['LABEL']].to_numpy()
x_test = test_data.drop(columns=['LABEL', 'ICUSTAY_ID']).to_numpy()
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
