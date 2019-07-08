import pandas as pd
from _log import logger
from monitor import Monitor, DataManager

data = pd.read_csv('../data/CIP_1.csv', sep='\t')
data.time = pd.to_datetime(data.time, format='%Y-%m-%d %H:%M:%S.%f')
train_size = 20000
test_size = 20000
train_data = data.loc[: train_size]
test_data = data.loc[train_size: train_size + test_size]

# Load history data
manager = DataManager(limit=train_size)
manager.load(train_data, clear=True)

# Build monitor using manager
monitor = Monitor(manager)
count = [0, 0]
for row in test_data.itertuples():
    is_safe = monitor.process(row)
    count[is_safe] += 1
logger.debug(str(count) + 'trust rate:' + str(count[1] / sum(count)))
