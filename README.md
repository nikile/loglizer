# Main information about [loglizer](https://github.com/logpai/loglizer) fork


**Loglizer is a machine learning-based log analysis toolkit for automated anomaly detection
In this fork, the main emphasis is on improving the performance of a PCA model by changing the data preprocessing - splitting the analyzed data window by different time intervals
and also adding functionality to build graphs and reports.
You may find additional information in the [original repository](https://github.com/logpai/loglizer)**.


## Framework

![Framework of Anomaly Detection](/docs/img/framework.png)

The log analysis framework for anomaly detection usually comprises the following components:

1. **Log collection:** Logs are generated at runtime and aggregated into a centralized place with a data streaming pipeline, such as Flume and Kafka.
2. **Log parsing:** The goal of log parsing is to convert unstructured log messages into a map of structured events, based on which sophisticated machine learning models can be applied. The details of log parsing can be found at [our logparser project](https://github.com/logpai/logparser).
3. **Feature extraction:** Structured logs can be sliced into short log sequences through interval window, sliding window, or session window. Then, feature extraction is performed to vectorize each log sequence, for example, using an event counting vector.
4. **Anomaly detection:** Anomaly detection models are trained to check whether a given feature vector is an anomaly or not.


## Models

Anomaly detection unsupervised model :

| Model | Paper reference |
| PCA | [**SOSP'09**] [Large-Scale System Problems Detection by Mining Console Logs](http://iiis.tsinghua.edu.cn/~weixu/files/sosp09.pdf), by Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan. [**Intel**] |


## Log data
[Log data](https://github.com/nikile/loglizer/tree/master/data/HDFS) used in the following fork.
[Or from original repository](https://github.com/logpai/loghub/tree/master/HDFS)

## Install
```bash
git clone https://github.com/logpai/loglizer.git
cd loglizer
pip install -r requirements.txt
```

## API usage

```python
# Load HDFS dataset. If you would like to try your own log, you need to rewrite the load function.
(x_train, _), (_, _) = dataloader.load_HDFS(...)

# Feature extraction and transformation
feature_extractor = preprocessing.FeatureExtractor()
feature_extractor.fit_transform(...)

# Model training
model = PCA()
model.fit(...)

# Feature transform after fitting
x_test = feature_extractor.transform(...)
# Model evaluation with labeled data
model.evaluate(...)

# Anomaly prediction
x_test = feature_extractor.transform(...)
model.predict(...) # predict anomalies on given data
```

For more details, please follow [the demo](./docs/demo.md) in the docs to get started.

## Analysis results for [sample data](https://github.com/nikile/loglizer/tree/master/data/HDFS)

The graph illustrates anomalies search based on a comparison of the SPE (squared prediction error) threshold calculated for all training dataset and SPE calculated for events that occurred in the corresponding time period.
If the SPE calculated at a given moment of time is greater than SPE threshold , then this is considered as an anomaly.
For this data example, events are analyzed per second (alternatively analyzed time periods can be specified in the [code](https://github.com/nikile/loglizer/blob/master/demo/PCA_HDFS_demo.py) by time_delta_sec parameter)

<p align="center"> <img src="https://github.com/nikile/loglizer/blob/master/analysis_results/HDFS/graph.png"></p>

All events have id,template and weight counted by tf–idf method. SPE is calculated from these weights.

<p align="center"> <img src="https://github.com/nikile/loglizer/blob/master/analysis_results/HDFS/keys_chart.png"></p>

All found anomalies are recorded in the [report](https://github.com/nikile/loglizer/blob/master/analysis_results/HDFS/anomalies_report.txt)





