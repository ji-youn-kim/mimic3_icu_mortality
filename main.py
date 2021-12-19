import pandas as pd
import numpy as np
from csv import reader
from datetime import datetime

icu_chart_events_path = "./data/icu_chart_events.csv"
icu_chart_keys = ['ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS', 'ADMITTIME', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS', 'CHARTEVENTS', 'label']
icu_chart_dates = ['INTIME', 'OUTTIME', 'ADMITTIME']

icu_chart_events = pd.read_csv(icu_chart_events_path, usecols=icu_chart_keys, parse_dates=icu_chart_dates)

# if icu stay id is 8, 9 -> test data
train_data = icu_chart_events[(icu_chart_events['ICUSTAY_ID'] % 10 != 8) & (icu_chart_events['ICUSTAY_ID'] % 10 != 9)]
test_data = icu_chart_events[(icu_chart_events['ICUSTAY_ID'] % 10 == 8) | (icu_chart_events['ICUSTAY_ID'] % 10 == 9)]

print(icu_chart_events.shape)
print("Train data")
print(train_data.shape)
print(train_data.head())
print("Test data")
print(test_data.shape)
print(test_data.head())

