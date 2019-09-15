import pandas as pd
import os
import numpy as np
import re
from collections import OrderedDict


def load_HDFS_data_timestamp_approach(input_path, time_delta_sec, timestamp_format, cached_workflow_path='data_df.csv', sep=',', encoding ='utf-8', cache_workflow=True):
    """
        Downloads cached workflow data from csv file

        Args:
            input_path: path to cached workflow csv file
            time_delta_sec: analyzed period of time in seconds
            timestamp_format: timestamp format in logs
            cached_workflow_path: path to cached workflow csv file
            cache_workflow: cache workflow or not

        Returns:
            x_data: array of lists of event id's np.array(['E21', 'E22', ...], [...],...)
    """

    print('====== Input data summary ======')

    struct_log = pd.read_csv(input_path, sep=sep,encoding=encoding,header=0)
    freq_val = str(time_delta_sec) + 'S'
    struct_log['Timestamp'] = pd.to_datetime(struct_log['Timestamp'], format=timestamp_format, errors='ignore')
    struct_log = struct_log.drop(['LineId', 'Pid'], axis=1)
    struct_log.set_index('Timestamp', inplace=True)
    struct_log = struct_log.groupby(pd.Grouper(freq=freq_val)).apply(lambda x:(x + ',').sum())
    struct_log = pd.DataFrame(struct_log['EventId'])

    # drop rows of NaT values in struct_log.index
    struct_log = struct_log[pd.notnull(struct_log.index)]

    data_dict = OrderedDict()
    for idx, row in struct_log.iterrows():
        group_id_list = str(idx)
        if not group_id_list in data_dict:
            data_dict[group_id_list] = None
        data_dict[group_id_list] = list(filter(None, str(row['EventId']).split(',')))
    data_df = pd.DataFrame(list(data_dict.items()), columns=['group_id', 'event_sequence'])

    data_df['number_of_events'] = data_df['event_sequence'].apply(lambda x: len(x))
    cols = ['group_id', 'number_of_events', 'event_sequence']
    data_df = data_df[cols]

    if cache_workflow:
        data_df.to_csv(cached_workflow_path, index=False)
    x_data = data_df['event_sequence'].values

    print('Total: {} instances'.format(x_data.shape[0]))

    return x_data


def load_HDFS_data_debug(input_path):
    """
        Downloads cached workflow data from *.csv file
        to speed up the analysis of logs and debug.

        Args:
            input_path: path to cached workflow csv file

        Returns:
            x_data: array of lists of event id's np.array(['E21', 'E22', ...], [...],...)
    """

    print('====== Input data summary ======')

    data_df = pd.read_csv(input_path, header=0)
    x_data = np.asarray(list(map(lambda x: x[2:-2].split('\', \''), data_df['event_sequence'].values)))

    return x_data