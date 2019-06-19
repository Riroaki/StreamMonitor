import pandas as pd
import logging
from model import Model

logging.basicConfig(level=logging.INFO)
m = Model()
data = pd.read_csv('../data/CIP_1.csv', sep='\t')
data.time = pd.to_datetime(data.time, format='%Y-%m-%d %H:%M:%S.%f')
size = data.size
train_data = data.loc[: size / 2]
test_data = data.loc[size / 2:]
m.load_history(train_data)
for row in test_data.itertuples():
    m.process(row)
