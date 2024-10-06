from datetime import datetime
from pandas.tseries.offsets import BDay
import os
def lastbussinessday():
        return (datetime.today() - BDay()).strftime('%Y-%m-%d')

def caminho():
        base_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = base_dir + '/storage'
        return file_path