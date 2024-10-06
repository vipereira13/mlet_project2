import pandas as pd
from src.utils.functions import lastbussinessday
from src.utils.functions import caminho

def preparetoupload():
    file_path = caminho()
    day = lastbussinessday()
    day = day.replace('-', '')
    df = pd.read_csv(file_path + f'/TradeInformationConsolidatedFile_{day}_1.csv', skiprows=1, sep=';')
    df = df.fillna(0)

    s3_url = 's3://bucket/folder/bucket.parquet.gzip'
    df.to_parquet(s3_url, compression='gzip')\

    return df


