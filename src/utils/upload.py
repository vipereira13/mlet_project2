import pandas as pd
from src.utils.functions import lastbussinessday
from src.utils.functions import caminho

def preparatoupload():
    file_path = caminho()
    day = lastbussinessday()
    day_nodash = day.replace('-', '')
    df = pd.read_csv(f"{file_path}/TradeInformationConsolidatedFile_{day_nodash}_1.csv", skiprows=1, sep=';')
    df = df.fillna(0)

    # Adjust the S3 URL to include the date partition
    s3_url = f's3://b3-project/raw-data/date={day}/TradeInformationConsolidatedFile_{day_nodash}_1.parquet.gzip'

    # Save the DataFrame to S3 in Parquet format with gzip compression
    df.to_parquet(s3_url, compression='gzip')

    return df

