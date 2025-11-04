# pip install playwright
#  

import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


async def main():
    url = "http://quotes.toscrape.com/scroll"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Puedes usar `headless=True` si no quieres abrir el navegador
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url)
        await page.wait_for_timeout(3000)  # Esperar 3 segundos para que cargue el contenido dinámico

        # Obtener el HTML de la página renderizada
        html = await page.content()
        await browser.close()

        # Procesar con BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        quotes = soup.select("div.quote")

        print("Citas encontradas:", len(quotes))
        for quote in quotes:
            text = quote.find("span", class_="text").get_text()
            author = quote.find("small", class_="author").get_text()
            print(f"{text} - {author}")


# Ejecutar el script
asyncio.run(main())
