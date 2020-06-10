from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql import SparkSession
import pandas as pd


spark = SparkSession.builder.appName('Test').getOrCreate()

@pandas_udf("id long, v long", PandasUDFType.GROUPED_MAP)
def subtract_mean(pdf):
    # pdf is a pandas.DataFrame
    print('test1')
    v = pdf.v
    return pdf.assign(v=v - v.mean())

if __name__ == '__main__':
    df = pd.DataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)], columns =("id", "v") )
    df = spark.createDataFrame(df)
    df1 = df.groupby("id").apply(subtract_mean)
    df.show()