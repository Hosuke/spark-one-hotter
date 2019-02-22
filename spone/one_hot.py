# -*- coding: utf-8 -*-
# author: Huang Geyang

import gc
import os

from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


from pyspark.sql.functions import udf

@udf
def process_missing_category(s):
    if s == "":
        return "NA"
    else:
        return s


def one_hot(spark):
    root_path = "hdfs://xxx/"
    data_path = "xxx.csv"
    input_path = os.path.join(root_path, data_path)

    data = spark.read.parquet(input_path)

    data.show()

    features_onehot = ["NAME_FAMILY_STATUS", "OCCUPATION_TYPE", "CODE_GENDER", "FLAG_OWN_REALTY"]

    features_used = [col_name for col_name in data.columns if col_name not in features_onehot] \
                    + [col_name + "_onehot" for col_name in features_onehot]

    for fea in features_onehot:
        data = data.withColumn(fea, process_missing_category(fea))

    # StringIndexer and OneHotEncoder
    stage_str = [StringIndexer(inputCol=col_name,
                               outputCol=col_name + "_str_encoded") for col_name in features_onehot]

    stage_onehot = [OneHotEncoder(inputCol=col_name + "_str_encoded",
                                  outputCol=col_name + "_onehot") for col_name in features_onehot]

    stage_assembler = [VectorAssembler(inputCols=features_used, outputCol="features")]

    ppl = Pipeline(stages=stage_str + stage_onehot)


    encoder = ppl.fit(data)
    try:
        output = encoder.transform(data)
    except Exception as e:
        print(e)

    output.show()


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
