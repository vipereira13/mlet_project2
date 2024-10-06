from fastapi import FastAPI
from src.utils.scrapping import download_arquivo
from src.utils.upload import preparetoupload
from src.utils.scrapping import teste

app = FastAPI()
@app.get("/link/")
async def get_website_link():
    # Faz o scraping do site especificado

    download_arquivo()
    preparetoupload()
