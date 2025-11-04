import asyncio
from playwright.async_api import async_playwright


async def main():
    url = "http://quotes.toscrape.com/scroll"
    SCROLL_PAUSE_TIME = 2  # segundos

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headless=True para modo invisible
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url)

        quotes_set = set()
        last_height = await page.evaluate("document.body.scrollHeight")

        for i in range(3):  # hasta 3 scrolls
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await page.wait_for_timeout(SCROLL_PAUSE_TIME * 1000)

            # Extraer todas las frases actualmente visibles
            quote_elements = await page.query_selector_all(".quote")
            for quote in quote_elements:
                text_span = await quote.query_selector(".text")
                if text_span:
                    text = await text_span.inner_text()
                    quotes_set.add(text)

            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        await browser.close()

    print(f"Total de frases únicas cargadas: {len(quotes_set)}")
    for quote in quotes_set:
        print(quote)


# Ejecutar el script
asyncio.run(main())
