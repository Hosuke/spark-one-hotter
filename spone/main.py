# -*- coding: utf-8 -*-
# author: Huang Geyang
from pyspark.sql import SparkSession
from spone import one_hot

if __name__ == '__main__':
    # spark = SparkSession.builder\
    #     .appName('one-hotter') \
    #     .config('spark.sql.warehouse.dir', '/user/hive/warehouse') \
    #     .enableHiveSupport() \
    #     .getOrCreate()

    spark = SparkSession.builder \
        .appName('one-hotter') \
        .getOrCreate()

    print('One-hot encoding...')
    one_hot(spark)

    spark.stop()