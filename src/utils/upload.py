import pandas as pd
from src.utils.functions import lastbussinessday
from src.utils.functions import caminho
import pymysql
import boto3
import os

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAST6S64SLCYZ7BWXN'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

# Get RDS endpoint
client = boto3.client('rds', region_name='us-east-2')
endpoint = 'mlet-b3.c3qk06ewapu6.us-east-2.rds.amazonaws.com'

# Connect to the RDS instance
connection = pymysql.connect(
    host=endpoint,
    port=3306,
    user='admin',
    password='Fazcw5fo1!',
    db='mlet_b3'
)


def preparetoupload():
    file_path = caminho()
    day = lastbussinessday()
    day = day.replace('-', '')
    df = pd.read_csv(file_path + f'/TradeInformationConsolidatedFile_{day}_1.csv', skiprows=1, sep=';')
    df = df.fillna(0)


    df.to_sql(con=connection, name='TradeInformationConsolidated', if_exists='append')



    return df

connection.close()


