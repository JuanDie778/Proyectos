import asyncio
import csv
from datetime import datetime
from playwright.async_api import async_playwright
import random
import re
import pandas as pd
import os
import langid

class LinkedInUSABIMScraper:
    def __init__(self):
        self.search_query = "Bim Modeler"
        self.location_filter = "Estados Unidos"
        self.location_code = "%5B%22urn%3Ali%3Afs_geo%3A103644278%22%5D"
        self.results = []
        self.max_posts = 10
        self.consolidated_csv = "linkedin_usa_bim_posts_consolidated.csv"
        self.consolidated_excel = "linkedin_usa_bim_posts_consolidated.xlsx"

    async def setup_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--start-maximized']
        )
        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            viewport={'width': 1366, 'height': 768}
        )
        self.page = await self.context.new_page()

    async def login_to_linkedin(self, email: str, password: str):
        try:
            print("🔐 Iniciando sesión en LinkedIn...")
            await self.page.goto('https://www.linkedin.com/login', wait_until='networkidle')
            await asyncio.sleep(random.uniform(1, 3))
            
            await self.page.fill('#username', email)
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            await self.page.fill('#password', password)
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            await self.page.click('button[type="submit"]')
            
            await self.page.wait_for_load_state('networkidle')
            print("✅ Sesión iniciada correctamente")
            return True
        except Exception as e:
            print(f"❌ Error en login: {e}")
            return False

    async def handle_popups(self):
        """Cerrar pop-ups emergentes"""
        try:
            await asyncio.sleep(2)
            selectors = ['.artdeco-modal__dismiss', '.msg-overlay-bubble-header__control', 'button[aria-label="Dismiss"]']
            for selector in selectors:
                try:
                    close_btn = await self.page.query_selector(selector)
                    if close_btn:
                        await close_btn.click()
                        await asyncio.sleep(1)
                except:
                    continue
        except Exception as e:
            print(f"⚠️ Error manejando pop-ups: {e}")

    def detect_language(self, text):
        """Detectar idioma del texto"""
        if text == "N/A" or len(text.strip()) < 10:
            return "unknown"
        
        try:
            lang, confidence = langid.classify(text)
            return lang
        except:
            return "unknown"

    def is_desired_language(self, text):
        """Verificar si el texto está en un idioma deseado"""
        if text == "N/A":
            return True
            
        undesired_languages = ['ar', 'hi', 'ur', 'bn', 'pa']
        
        lang = self.detect_language(text)
        
        if lang == "unknown":
            arabic_chars = bool(re.search('[\u0600-\u06FF]', text))
            hindi_chars = bool(re.search('[\u0900-\u097F]', text))
            
            if arabic_chars or hindi_chars:
                return False
            return True
        
        return lang not in undesired_languages

    async def verify_author_location(self, author_url):
        """Verificar la ubicación del autor visitando su perfil"""
        if author_url == "N/A":
            return False
            
        try:
            new_page = await self.context.new_page()
            await new_page.goto(author_url, wait_until='networkidle')
            await asyncio.sleep(2)
            
            location_selectors = [
                '.text-body-small.inline.t-black--light.break-words',
                '.pv-top-card--list-bullet li',
                '.pv-top-card-v2--list-bullet li',
                '.inline-show-more-text--is-collapsed'
            ]
            
            location = "N/A"
            for selector in location_selectors:
                location_element = await new_page.query_selector(selector)
                if location_element:
                    location = await location_element.inner_text()
                    if location and location.strip():
                        break
            
            await new_page.close()
            
            usa_indicators = [
                'usa', 'united states', 'estados unidos', 'us', 'u.s.', 'u.s.a.',
                'new york', 'california', 'texas', 'florida', 'chicago', 'los angeles',
                'san francisco', 'boston', 'seattle', 'washington', 'miami',
                'ny', 'ca', 'tx', 'fl', 'il', 'wa', 'ma'
            ]
            
            location_lower = location.lower()
            return any(indicator in location_lower for indicator in usa_indicators)
            
        except Exception as e:
            print(f"    ⚠️ Error verificando ubicación del autor: {e}")
            return False

    def verify_content_location(self, post_data):
        """Verificar ubicación basada en el contenido del post"""
        text_content = f"{post_data['author_title']} {post_data['post_text']} {post_data['location']}".lower()
        
        usa_keywords = [
            'usa', 'united states', 'estados unidos', 'us', 'u.s.', 'u.s.a.',
            'new york', 'california', 'texas', 'florida', 'chicago', 'los angeles',
            'san francisco', 'boston', 'seattle', 'washington', 'miami',
            'ny', 'ca', 'tx', 'fl', 'il', 'wa', 'ma'
        ]
        
        return any(keyword in text_content for keyword in usa_keywords)

    async def search_and_extract_posts(self):
        try:
            print(f"🔍 Buscando publicaciones para: '{self.search_query}' en {self.location_filter}")
            
            search_url = (
                f"https://www.linkedin.com/search/results/content/?"
                f"keywords={self.search_query.replace(' ', '%20')}"
                f"&facetGeoRegion={self.location_code}"
                f"&facetContentType=%22posts%22"
            )
            
            await self.page.goto(search_url, wait_until='networkidle')
            await asyncio.sleep(random.uniform(3, 5))
            
            await self.handle_popups()
            
            try:
                await self.page.wait_for_selector('.feed-shared-update-v2', timeout=10000)
            except:
                print("⚠️ No se encontraron publicaciones después de 10 segundos")
            
            for i in range(3):
                await self.page.evaluate('''
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    })
                ''')
                await asyncio.sleep(random.uniform(2, 4))
            
            print("📝 Extrayendo publicaciones...")
            posts = await self.page.query_selector_all('.feed-shared-update-v2')
            
            if not posts:
                print("⚠️ No se encontraron publicaciones")
                return False
                
            print(f"✅ Encontradas {len(posts)} publicaciones. Procesando primeras {self.max_posts}...")
            
            for i, post in enumerate(posts[:self.max_posts]):
                try:
                    print(f"  🔍 Procesando publicación {i+1}/{min(len(posts), self.max_posts)}")
                    post_data = await self.extract_post_data(post)
                    
                    if not self.is_desired_language(post_data['post_text']):
                        print(f"  ⚠️ Publicación en idioma no deseado - Descartada")
                        continue
                    
                    location_verified = self.verify_content_location(post_data)
                    
                    if not location_verified and post_data['author_url'] != "N/A":
                        print(f"  🔍 Verificando ubicación en perfil del autor...")
                        location_verified = await self.verify_author_location(post_data['author_url'])
                    
                    if location_verified:
                        self.results.append(post_data)
                        print(f"  ✅ Publicación añadida - URL: {post_data['post_url']}")
                    else:
                        print(f"  ⚠️ Publicación fuera de EE.UU. - Descartada")
                except Exception as e:
                    print(f"  ⚠️ Error en publicación {i+1}: {e}")
            
            return True
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return False

    def extract_url_from_data_urn(self, data_urn):
        if data_urn and data_urn.startswith('urn:li:activity:'):
            activity_id = data_urn.split(':')[-1]
            return f"https://www.linkedin.com/feed/update/{activity_id}/"
        return None

    async def extract_post_data(self, post):
        author_name = "N/A"
        try:
            author_element = await post.query_selector('.update-components-actor__name')
            author_name = await author_element.inner_text() if author_element else "N/A"
        except:
            pass
        
        author_title = "N/A"
        try:
            title_element = await post.query_selector('.update-components-actor__description')
            author_title = await title_element.inner_text() if title_element else "N/A"
        except:
            pass
        
        author_url = "N/A"
        try:
            selectors = [
                'a.app-aware-link[href*="/in/"]',
                '.update-components-actor__container a',
                '.update-components-actor__name-link'
            ]
            
            for selector in selectors:
                link_element = await post.query_selector(selector)
                if link_element:
                    author_url = await link_element.get_attribute('href')
                    if author_url:
                        author_url = author_url.split('?')[0]
                        break
        except:
            pass
        
        post_url = "N/A"
        try:
            strategies = [
                ('a.app-aware-link[href*="/feed/update/"]', None),
                ('a[data-id*="share-"]', None),
                ('.update-components-actor__container a', lambda url: url if "/feed/update/" in url else None),
                ('.feed-shared-update-v2__content', None),
                ('.feed-shared-update-v2', self.extract_url_from_data_urn)
            ]
            
            for selector, processor in strategies:
                try:
                    element = await post.query_selector(selector)
                    if element:
                        if processor:
                            url = processor(await element.get_attribute('href') or await element.get_attribute('data-urn'))
                            if url:
                                post_url = url
                                break
                        else:
                            url = await element.get_attribute('href')
                            if url and "/feed/update/" in url:
                                post_url = url.split('?')[0]
                                break
                except:
                    continue
                    
        except Exception as e:
            print(f"    ⚠️ Error extrayendo URL de publicación: {e}")
        
        post_text = "N/A"
        try:
            text_element = await post.query_selector('.update-components-text')
            if text_element:
                post_text = await text_element.inner_text()
                post_text = post_text[:500]
        except:
            pass
        
        post_date = "N/A"
        try:
            date_element = await post.query_selector('.update-components-actor__sub-description')
            if date_element:
                post_date = await date_element.inner_text()
                post_date = re.sub(r'•.*', '', post_date).strip()
        except:
            pass
        
        location = "N/A"
        try:
            location_element = await post.query_selector('.update-components-actor__sub-description .update-components-actor__distance')
            if location_element:
                location = await location_element.inner_text()
        except:
            pass
        
        return {
            'author_name': author_name,
            'author_title': author_title,
            'author_url': author_url,
            'post_url': post_url,
            'post_text': post_text,
            'post_date': post_date,
            'location': location,
            'scraped_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_unique_id(self, row):
        """Genera un ID único para cada publicación basado en múltiples campos"""
        # Si tenemos una URL válida, usarla como identificador principal
        if row.get('post_url', 'N/A') != 'N/A' and pd.notna(row.get('post_url')):
            return f"url_{row['post_url']}"
        
        # Si no hay URL, usar combinación de otros campos
        author_name = str(row.get('author_name', 'N/A'))
        post_text = str(row.get('post_text', 'N/A'))[:100]
        post_date = str(row.get('post_date', 'N/A'))
        
        # Crear un hash basado en la combinación de campos
        combined_string = f"{author_name}_{post_text}_{post_date}"
        return f"hash_{hash(combined_string)}"

    def remove_duplicates_smart(self, df):
        """Elimina duplicados de manera inteligente"""
        df = df.copy()
        
        if 'post_url' in df.columns:
            # Crear columna de ID único
            df['unique_id'] = df.apply(
                lambda row: self.generate_unique_id(row), 
                axis=1
            )
            
            # Eliminar duplicados manteniendo el último
            df = df.drop_duplicates(subset=['unique_id'], keep='last')
            
            # Eliminar la columna temporal
            df = df.drop(columns=['unique_id'])
        
        return df

    async def save_to_csv_and_excel(self):
        if not self.results:
            print("❌ No hay datos para guardar")
            return
            
        new_df = pd.DataFrame(self.results)
        
        individual_csv = f"linkedin_usa_bim_posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        new_df.to_csv(individual_csv, index=False, encoding='utf-8')
        print(f"✅ Archivo individual guardado en {individual_csv}")
        
        # Guardar/actualizar archivo consolidado CSV
        if os.path.exists(self.consolidated_csv):
            existing_df = pd.read_csv(self.consolidated_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = self.remove_duplicates_smart(combined_df)
            combined_df.to_csv(self.consolidated_csv, index=False, encoding='utf-8')
            print(f"✅ Archivo consolidado CSV actualizado: {self.consolidated_csv}")
        else:
            new_df.to_csv(self.consolidated_csv, index=False, encoding='utf-8')
            print(f"✅ Nuevo archivo consolidado CSV creado: {self.consolidated_csv}")
        
        # Guardar/actualizar archivo consolidado Excel
        if os.path.exists(self.consolidated_excel):
            existing_df = pd.read_excel(self.consolidated_excel)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = self.remove_duplicates_smart(combined_df)
            combined_df.to_excel(self.consolidated_excel, index=False)
            print(f"✅ Archivo consolidado Excel actualizado: {self.consolidated_excel}")
        else:
            new_df.to_excel(self.consolidated_excel, index=False)
            print(f"✅ Nuevo archivo consolidado Excel creado: {self.consolidated_excel}")
        
        print(f"📊 Publicaciones nuevas: {len(new_df)}")
        if os.path.exists(self.consolidated_csv):
            consolidated_df = pd.read_csv(self.consolidated_csv)
            print(f"📊 Total de publicaciones en consolidado: {len(consolidated_df)}")

    async def run_scraper(self, email: str, password: str):
        try:
            await self.setup_browser()
            if await self.login_to_linkedin(email, password):
                await self.handle_popups()
                if await self.search_and_extract_posts():
                    await self.save_to_csv_and_excel()
        except Exception as e:
            print(f"❌ Error general: {e}")
        finally:
            await self.browser.close()
            await self.playwright.stop()

async def main():
    print("=" * 60)
    print("🔍 LINKEDIN BIM SCRAPER - FILTRO ESTADOS UNIDOS")
    print(f"📝 Buscando: 'Bim Modeler'")
    print(f"📍 Ubicación: Estados Unidos")
    print("=" * 60)
    
    scraper = LinkedInUSABIMScraper()
    
    email = "ussaapontejuandiego@gmail.com"
    password = "juandi778"
    
    if email == "tu_email@example.com":
        print("❌ Configura tus credenciales reales")
        return
    
    await scraper.run_scraper(email, password)
    print("🏁 Proceso completado")

if __name__ == "__main__":
    asyncio.run(main())