
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, Query, Depends, Body  # Importa classes e funções necessárias do FastAPI
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from src.utils.functions import caminho

import os
from src.utils.functions import lastbussinessday

def download_arquivo():

    # Faz o scraping do site especificado

    try:

        file_path = caminho()

        day = lastbussinessday()

        url = f"https://arquivos.b3.com.br/tabelas/TradeInformationConsolidated/{day}?lang=pt"

        prefs = {'download.default_directory': file_path}
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option('prefs', prefs)

        # Inicia o webdriver do Chrome
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(10)

        day_2 = day.replace('-', '')
        if not os.path.exists(file_path + f'/TradeInformationConsolidatedFile_{day_2}_1.csv'):

           file = driver.find_element(By.LINK_TEXT, "Baixar arquivo completo").click()

        time.sleep(10)

    except:
        print("Erro ao salvar arquivo")




def teste():
    day = lastbussinessday()
    day = day.replace('-', '')
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = base_dir + '/storage'
    confirmacao = os.path.exists(file_path + f'/TradeInformationConsolidatedFile_{day}_1.csv')
    caminho = file_path + f'/TradeInformationConsolidatedFile_{day}_1.csv'
    return confirmacao, caminho