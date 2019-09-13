import numpy as np
import pandas as pd
from matplotlib import pyplot
from collections import Counter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.transforms as transforms


def make_key_id_dict(input_path, output_path, encoding='utf-8'):
    """
        Creates key-id dictionary from log file workflow
        and saves it to *.txt file

        Args:
            input_path: input path to file with log for which
    """
    df = pd.read_csv(input_path, encoding=encoding, sep='~', header=0)
    df["key_and_id"] = df["template"] + "~" +df["event_id"]
    np.savetxt(output_path, df["key_and_id"].unique(), header="key~id",
               comments='', fmt='%s')


def get_graph(workflow_path, prediction, threshold, tick_period_in_sec = 900, output_path='graph.png'):
    """
        Draws graph with x axis: timestamp
        y axis: SPE (squared prediction error) for corresponding timestamp,
        horizontal green line for threshold SPE value.
        if the calculated SPE value exceeds the threshold value,
        then an anomaly is diagnosed and is indicated by a vertical red line.

        Args:
            workflow_path: path to cached workflow csv file
            prediction: df with SPE value calculated and
            threshold: the boundary value of the error when exceeded
                        which an anomaly is diagnosed
            tick_period_in_sec: time interval in seconds through
                        which timestamp marks on the x axis will be marked
            output_path: output path to save result graph

    """
    data_df = pd.read_csv(workflow_path, header=0)
    data_df = pd.concat([data_df, prediction], axis=1)
    data_df['number_of_events'] = pd.to_numeric(data_df['number_of_events'])
    data_df['group_id'] = pd.to_datetime(data_df['group_id'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    fig, ax = pyplot.subplots()
    pyplot.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    pyplot.bar(data_df['group_id'].values, data_df['SPE'].values, width=0.004)
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=tick_period_in_sec))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    #draw red vertical line for every anomaly found
    for idx, row in data_df.iterrows():
        if row['prediction'] == 1:
            ax.axvline(x=row['group_id'], color='r', alpha=0.4)

    pyplot.xlabel('Timestamp')
    pyplot.ylabel('SPE')
    pyplot.title('Graph')
    pyplot.axhline(y=threshold, color='g', linestyle='-')
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, threshold, "{:.0f}".format(threshold), color="g", ha="right", transform=trans, va="center")
    pyplot.gcf().set_size_inches((20, 10), forward=False)
    #pyplot.show()
    fig.savefig(output_path, dpi=fig.dpi)


def get_anomalies_report(workflow_path, dict_path, prediction,output_path, time_delta_sec):
    """
        Makes *.txt report with list of detected anomalies.
        In format:
        time at which an anomaly is detected|
        number of events which took place in this time|
        quantitative contribution of each individual key

        Args:
            workflow_path: path to cached workflow csv file
            dict_path: path to file with key-id-weight information
            prediction: df with SPE value calculated and
            output_path: output path to save result graph
            time_delta_sec: time period in seconds within
                            which anomalies are searched

    """
    df_data = pd.read_csv(workflow_path, header=0)
    df_dict = pd.read_csv(dict_path, header=0, sep='~')
    max_weights = df_dict.sort_values('weight').drop_duplicates('weight', keep='last').tail(3)['weight']
    keys_within_max_weights = df_dict[df_dict['weight'].isin(max_weights)]
    prediction['prediction'] = list(map(lambda x: True if x else False, prediction['prediction']))
    filter_anomalies = pd.Series(prediction['prediction'])
    df_data = df_data[filter_anomalies]
    df_data['group_id'] = df_data['group_id'].apply(lambda x: 'From: ' + x
                                                              + ' to: ' + str(datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                                                                    + timedelta(seconds=time_delta_sec)))
    event_sequences_list = list(map(lambda x: x[2:-2].split('\', \''), df_data['event_sequence'].values))
    df_data = df_data.reset_index()
    df_data = df_data.drop(columns=['index'])
    for i in range(0, len(df_data)):
        c = Counter(event_sequences_list[i])
        # set_within_max_weights = set(c.elements()) & set(keys_within_max_weights['id'].values)
        # c = {x: c[x] for x in c if x in set_within_max_weights}
        df_data.at[i, 'event_sequence'] = 'Caused by: ' + str(sorted(c.items(), key=lambda x: x[1],reverse=True))
    df_data.to_csv(output_path, index=False)


def add_weights_to_key_dict(path_to_key_id_dict, key_weight_dict, output_path = 'key_dict.txt'):
    """
        Merges 2 dictionaries in format key:id and key:weight,
        saves result file.

        Args:
            path_to_key_id_dict: path to file which contains
                                dictionary in format key|id
            key_weight_dict: dictionary which contains key|weight
                            information
            output_path: output path to save result file

    """
    df_dict = pd.read_csv(path_to_key_id_dict, sep='~', header=0)
    df_dict['weight'] = df_dict[df_dict.columns[1]]
    df_dict = df_dict.replace({'weight': key_weight_dict})
    df_dict["weight"] = pd.to_numeric(df_dict["weight"])
    df_dict = df_dict.sort_values(by='weight', ascending=False)
    df_dict.to_csv(output_path, sep='~', index=False)


def get_keys_chart(path_to_key_id_weight_df, output_path = r'keys_chart.png'):
    """
        Builds chart illustrating each unique key weight.

        Args:
            path_to_key_id_weight_df: path to file containing information
                                    in format key|id|weight
            output_path: output path to save result chart

    """
    key_id_weight_df = pd.read_csv(path_to_key_id_weight_df, sep='~', header=0)
    key_id_weight_df['key'] = key_id_weight_df['id'] + ': ' + key_id_weight_df['key']
    key_id_weight_df = key_id_weight_df.sort_values('weight')
    x = key_id_weight_df['key'].values
    y = key_id_weight_df['weight'].values
    fig, ax = pyplot.subplots()
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    width = 1  # the width of the bars
    ind = 2 * np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y, width, color='b')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(x, minor=False)
    for i, v in enumerate(y):
        ax.text(v, ind[i] - width / 4, "{0:.2f}".format(round(v, 3)), color='black', fontsize=15)
    pyplot.gcf().set_size_inches(50, 100)
    pyplot.subplots_adjust(left=0.6)
    pyplot.title('keys chart')
    pyplot.xlabel('tf-idf')
    pyplot.ylabel('keys')
    pyplot.box(False)
    # pyplot.show()
    pyplot.savefig(output_path)