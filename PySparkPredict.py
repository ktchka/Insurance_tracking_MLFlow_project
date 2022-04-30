import argparse
import os

from pyspark.sql import SparkSession

import mlflow


os.environ['MLFLOW_S3_ENDPOINT_URL'] = '[]]'
os.environ['AWS_ACCESS_KEY_ID'] = '[]]'
os.environ['AWS_SECRET_ACCESS_KEY'] = '[]]'

def process(spark, data_path, result):
    """
    The main process of the task.

    :param spark: SparkSession
    :param data_path: path to the dataset
    :param result: path to save the result
    """

    logged_model = '[]]'
    model = mlflow.spark.load_model(logged_model)

    prediction = model.transform(data_path)
    new_path = os.makedirs(f'/home/ktchka/PycharmProjects/ML_Karpov/Insurance/{result}')
    prediction.to_csv(f'{new_path}/result')



def main(data, result):
    spark = _spark_session()
    process(spark, data, result)


def _spark_session():
    """
    Creating SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkPredict').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.parquet', help='Please set datasets path.')
    parser.add_argument('--result', type=str, default='result', help='Please set result path.')
    args = parser.parse_args()
    data = args.data
    result = args.result
    main(data, result)
