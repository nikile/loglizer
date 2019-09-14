''' This is a demo file for the PCA model.
    API usage:
        dataloader.load_HDFS(): load HDFS dataset
        feature_extractor.fit_transform(): fit and transform features
        feature_extractor.transform(): feature transform after fitting
        model.fit(): fit the model
        model.predict(): predict anomalies on given data
        model.evaluate(): evaluate model accuracy with labeled data
'''
from loglizer.models import PCA
from loglizer import dataloader, preprocessing, graphs_and_reports

if __name__ == '__main__':

    train_path = r'..\data\HDFS\presentation_train.csv'
    test_path = r'..\data\HDFS\presentation_test.csv'
    # time period in seconds within which anomalies will be searched
    time_delta_sec = 100

    x_train = dataloader.load_HDFS_data_timestamp_approach(train_path, time_delta_sec=time_delta_sec,
                                                          timestamp_format='%Y-%m-%d %H:%M:%S,%f',
                                                          cached_workflow_path=r'..\cached\HDFS\train_workflow.csv')
    x_train = dataloader.load_HDFS_data_debug(r'..\cached\HDFS\train_workflow.csv')

    graphs_and_reports.make_key_id_dict(input_path="../data/HDFS/presentation_train.csv",
                                        output_path="../analysis_results/HDFS/key_dict.txt")

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.df_fit_transform(x_train)
    # x_train = feature_extractor.fit_transform(x_train)

    graphs_and_reports.add_weights_to_key_dict("../analysis_results/HDFS/key_dict.txt",
                                               feature_extractor.idf_vec.to_dict(),
                                               output_path='../analysis_results/HDFS/key_dict.txt')
    graphs_and_reports.get_keys_chart("../analysis_results/HDFS/key_dict.txt",
                                      output_path=r'../analysis_results/HDFS/keys_chart.png')

    ## Train an unsupervised model
    print('Train phase:')
    # Initialize PCA
    model = PCA()
    model.fit(x_train)

    print('Prediction phase:')
    prediction = model.predict(x_train.values)

    print(prediction)
    print(prediction['prediction'].values)

    x_train = dataloader.load_HDFS_data_timestamp_approach(test_path, time_delta_sec=time_delta_sec,
                                                      timestamp_format='%Y-%m-%d %H:%M:%S,%f',
                                                      cached_workflow_path=r'..\cached\HDFS\test_workflow.csv')
    x_train = dataloader.load_HDFS_data_debug(r'..\cached\HDFS\test_workflow.csv')
    x_train = feature_extractor.transform(x_train)
    prediction = model.predict(x_train)

    print(prediction)
    print(prediction['prediction'].values)

    graphs_and_reports.get_anomalies_report(r'..\cached\HDFS\test_workflow.csv',
                                            r'../analysis_results/HDFS/key_dict.txt',
                                            prediction, r'..\analysis_results\HDFS\anomalies_report.txt',
                                            time_delta_sec=time_delta_sec)
    graphs_and_reports.get_graph(r'..\cached\HDFS\test_workflow.csv', prediction, model.threshold,
                                 output_path=r'../analysis_results/HDFS/graph.png')

    # x_test = feature_extractor.transform(np.array([['E10', 'E10', 'E10', 'E10', 'E10', 'E10', 'E10']]))
    # y_test = model.predict(x_test)
    # print(y_test)

    print('Done')
    


