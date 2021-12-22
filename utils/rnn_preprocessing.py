import pandas as pd
import numpy as np
from ast import literal_eval

# configure path to save numpy files
x_train_npy_path = "../data/X_train_rnn.npy"
x_test_npy_path = "../data/X_test_rnn.npy"

# read icu chart events csv, and convert into dataframe
icu_chart_events_path = "../data/icu_with_chart_events.csv"
icu_chart_keys = ['ICUSTAY_ID', 'LOS', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', \
                  'ETHNICITY', 'DIAGNOSIS', 'GENDER', 'CHARTEVENTS', 'LABEL']
icu_chart_events = pd.read_csv(icu_chart_events_path, usecols=icu_chart_keys)

# one hot encode categorical data
icu_chart_events = pd.get_dummies(icu_chart_events, columns=['ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', \
                                                             'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS', 'GENDER'], drop_first=True)

# display df without abbreviation
# pd.set_option('display.max_columns', None)

# get icu info without labels and chart_events
general_icu_info = icu_chart_events.drop(columns=['LABEL', 'CHARTEVENTS']).to_numpy()
# print("len(general_icu_info): ", len(general_icu_info))
# print("len(icu_chart_events): ", len(icu_chart_events))
print(icu_chart_events.head())

train_list = []
test_list = []
# store all item_id / value_num / timestamp values per each dataframe row
for index, row in icu_chart_events.iterrows():
    current_icu_info = general_icu_info[index]
    icu_stay_id = current_icu_info[0]
    chart_events = literal_eval(row['CHARTEVENTS'])
    # number of zero paddings needed for chart events
    lack_chart_events = 100 - len(chart_events)
    item_id_list = []
    for event in chart_events:
        value_num = event[2] if event[2] else 0
        item_id_list.append(event)
    # zero padding
    for i in range(0, lack_chart_events):
        item_id_list.append([0, 0, 0])
    # train data
    if (icu_stay_id % 10 != 8) & (icu_stay_id % 10 != 9):
        train_list.append([current_icu_info, item_id_list])
        # print("CUR TRAIN LIST: ", len(current_icu_info), len(item_id_list))
    # test data
    else:
        test_list.append([current_icu_info, item_id_list])
        # print("CUR TEST LIST: ", len(current_icu_info), len(item_id_list))

# convert train, test data to numpy - x, y
x_train = np.array(train_list, dtype=object)
x_test = np.array(test_list, dtype=object)

# save train, test numpy data to npy file
np.save(x_train_npy_path, x_train)
np.save(x_test_npy_path, x_test)