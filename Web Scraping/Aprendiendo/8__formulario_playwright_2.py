import asyncio
from playwright.async_api import async_playwright


async def run():
    async with async_playwright() as p:
        # Lanzar el navegador
        browser = await p.chromium.launch(headless=False)  # headless=False para ver lo que hace
        context = await browser.new_context()
        page = await context.new_page()

        # 1. Ir al sitio
        await page.goto("http://quotes.toscrape.com/scroll")
        await page.wait_for_timeout(2000)  # Esperar 2 segundos

        # 2. Hacer clic en "Login"
        await page.click("text=Login")

        # 3. Llenar usuario y contraseña
        await page.fill("#username", "platzi-admin")
        await page.fill("#password", "12345")
        await page.wait_for_timeout(2000)  # Esperar 2 segundos

        # 4. Clic en Submit
        await page.click("input[type='submit']")
        await page.wait_for_timeout(2000)  # Esperar 2 segundos

        # 5. Extraer los tags
        tags = await page.locator("div.tags-box a.tag").all_inner_texts()
        print("Top Ten Tags:")
        for tag in tags:
            print("-", tag)

        # Cerrar
        await browser.close()


# Ejecutar el script async
asyncio.run(run())
