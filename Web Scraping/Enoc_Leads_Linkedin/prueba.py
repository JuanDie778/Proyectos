import asyncio
import csv
from datetime import datetime
from playwright.async_api import async_playwright
import random
import re
import pandas as pd
import os
import langid
import time
from functools import wraps

# ==================== CLASES DE APOYO ====================

class ErrorHandler:
    """Manejador avanzado de errores y reintentos"""
    
    @staticmethod
    def async_retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
        """Decorador para reintentos con backoff exponencial en funciones async"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                retries, current_delay = 0, delay
                while retries < max_retries:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        retries += 1
                        if retries >= max_retries:
                            print(f"❌ Error después de {max_retries} intentos: {e}")
                            raise
                        
                        print(f"⚠️ Intento {retries}/{max_retries} fallido. Reintentando en {current_delay}s... Error: {e}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def handle_linkedin_errors(page_content):
        """Detecta errores específicos de LinkedIn en el contenido de la página"""
        error_patterns = [
            "Es culpa nuestra",
            "Something went wrong",
            "Please try again later",
            "Try again",
            "authwall",
            "captcha",
            "security check"
        ]
        
        content_lower = page_content.lower()
        return any(error in content_lower for error in error_patterns)

class ExecutionMetrics:
    """Sistema de métricas y logging de ejecución"""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'countries_processed': {},
            'terms_processed': {},
            'posts_found': 0,
            'posts_added': 0,
            'posts_filtered': 0,
            'errors_count': 0,
            'retries_count': 0
        }
        self.log_file = f"scraper_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def start_timer(self):
        """Inicia el temporizador de ejecución"""
        self.metrics['start_time'] = time.time()
        self._log("🚀 Iniciando ejecución del scraper")
    
    def end_timer(self):
        """Finaliza el temporizador y calcula la duración"""
        self.metrics['end_time'] = time.time()
        self.metrics['total_duration'] = self.metrics['end_time'] - self.metrics['start_time']
        self._log(f"⏰ Duración total: {self.metrics['total_duration']:.2f} segundos")
    
    def increment(self, metric_name, value=1, details=None):
        """Incrementa una métrica específica"""
        if metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], dict) and details:
                if details not in self.metrics[metric_name]:
                    self.metrics[metric_name][details] = 0
                self.metrics[metric_name][details] += value
            else:
                self.metrics[metric_name] += value
    
    def _log(self, message, level="INFO"):
        """Registra un mensaje en el log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        
        # Guardar en archivo
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def generate_report(self):
        """Genera un reporte detallado de la ejecución"""
        report = [
            "\n" + "="*60,
            "📊 REPORTE DE EJECUCIÓN DEL SCRAPER",
            "="*60,
            f"⏰ Duración total: {self.metrics['total_duration']:.2f} segundos",
            f"🌎 Países procesados: {len(self.metrics['countries_processed'])}",
            f"🔍 Términos procesados: {len(self.metrics['terms_processed'])}",
            f"📝 Publicaciones encontradas: {self.metrics['posts_found']}",
            f"✅ Publicaciones añadidas: {self.metrics['posts_added']}",
            f"🚫 Publicaciones filtradas: {self.metrics['posts_filtered']}",
            f"❌ Errores: {self.metrics['errors_count']}",
            f"🔄 Reintentos: {self.metrics['retries_count']}",
            "="*60
        ]
        
        # Detalles por país
        if self.metrics['countries_processed']:
            report.append("\n📈 Detalles por país:")
            for country, count in self.metrics['countries_processed'].items():
                report.append(f"   {country}: {count} publicaciones")
        
        # Detalles por término
        if self.metrics['terms_processed']:
            report.append("\n📈 Detalles por término:")
            for term, count in self.metrics['terms_processed'].items():
                report.append(f"   '{term}': {count} publicaciones")
        
        report_text = "\n".join(report)
        self._log(report_text)
        
        # Guardar reporte en archivo
        report_file = f"scraper_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text

class AntiScrapingManager:
    """Gestor avanzado de medidas anti-scraping"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        ]
        
        self.viewports = [
            {'width': 1920, 'height': 1080},
            {'width': 1366, 'height': 768},
            {'width': 1440, 'height': 900},
            {'width': 1536, 'height': 864},
            {'width': 1280, 'height': 720}
        ]
    
    def get_random_user_agent(self):
        """Obtiene un user agent aleatorio"""
        return random.choice(self.user_agents)
    
    def get_random_viewport(self):
        """Obtiene un viewport aleatorio"""
        return random.choice(self.viewports)
    
    async def human_like_delay(self, min_delay=1, max_delay=3):
        """Genera delays más humanos con variaciones"""
        base_delay = random.uniform(min_delay, max_delay)
        # Añadir micro-pausas ocasionales
        if random.random() < 0.3:
            base_delay += random.uniform(0.5, 2.0)
        await asyncio.sleep(base_delay)
    
    async def human_like_interaction(self, page):
        """Simula interacciones humanas realistas"""
        try:
            viewport = page.viewport_size
            # Movimiento de mouse natural
            for _ in range(random.randint(2, 5)):
                x = random.randint(100, viewport['width'] - 100)
                y = random.randint(100, viewport['height'] - 100)
                
                # Movimiento en curva
                steps = random.randint(3, 8)
                for i in range(steps):
                    curve_x = x + random.randint(-50, 50)
                    curve_y = y + random.randint(-30, 30)
                    await page.mouse.move(curve_x, curve_y)
                    await asyncio.sleep(random.uniform(0.05, 0.2))
                
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
                # Clic aleatorio ocasional
                if random.random() < 0.3:
                    await page.mouse.click(x, y)
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                    
        except Exception as e:
            print(f"⚠️ Error en interacción humana: {e}")
    
    async def human_like_scroll(self, page):
        """Scroll más realista con patrones variables"""
        scroll_patterns = [
            {"amount": random.randint(800, 1200), "delay": random.uniform(0.3, 0.7)},
            {"amount": random.randint(300, 500), "delay": random.uniform(1.0, 1.5)},
            {"amount": random.randint(500, 700), "delay": random.uniform(0.5, 1.0)}
        ]
        
        for pattern in scroll_patterns:
            await page.evaluate(f'''
                window.scrollBy({{
                    top: {pattern['amount']},
                    behavior: 'smooth'
                }})
            ''')
            await asyncio.sleep(pattern['delay'])
            await self.human_like_interaction(page)

# ==================== CLASE PRINCIPAL ====================

class LinkedInBIMScraper:
    def __init__(self):
        # Lista ampliada de términos de búsqueda
        self.search_terms = [
            "BIM Modeler", "BIM Manager", "BIM Coordinator", 
            "Revit Specialist", "3D Modeler", "VDC Engineer",
            "BIM Technician", "BIM Specialist", "BIM Designer",
            "Digital Construction", "Virtual Design Construction",
            "AEC Technology", "Building Information Modeling"
        ]
        
        # Países y sus códigos de ubicación en LinkedIn
        self.countries = {
            "United States": {
                "code": "%5B%22urn%3Ali%3Afs_geo%3A103644278%22%5D",
                "location_names": ["USA", "EEUU", "United States", "Estados Unidos"]
            },
            "Colombia": {
                "code": "%5B%22urn%3Ali%3Afs_geo%3A100876405%22%5D", 
                "location_names": ["Colombia", "CO", "Colombia"]
            }
        }
        
        self.results = []
        self.max_posts_per_term = 12  # Límite por término de búsqueda
        self.max_pages = 2  # Límite de páginas a procesar por término
        self.consolidated_csv = "linkedin_bim_posts_consolidated.csv"
        self.consolidated_excel = "linkedin_bim_posts_consolidated.xlsx"
        
        # Nuevos componentes
        self.error_handler = ErrorHandler()
        self.metrics = ExecutionMetrics()
        self.anti_scraping = AntiScrapingManager()

    async def setup_browser(self):
        self.playwright = await async_playwright().start()
        
        # Configuración anti-detection mejorada
        viewport = self.anti_scraping.get_random_viewport()
        user_agent = self.anti_scraping.get_random_user_agent()
        
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=[
                '--start-maximized',
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-dev-shm-usage'
            ]
        )
        
        self.context = await self.browser.new_context(
            user_agent=user_agent,
            viewport=viewport,
            java_script_enabled=True,
            permissions=['geolocation']
        )
        
        # Inyectar scripts anti-detección
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            window.chrome = {
                runtime: {},
            };
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        """)
        
        self.page = await self.context.new_page()

    @ErrorHandler.async_retry(max_retries=3, delay=2, backoff=2)
    async def login_to_linkedin(self, email: str, password: str):
        try:
            self.metrics._log("🔐 Iniciando sesión en LinkedIn...")
            await self.page.goto('https://www.linkedin.com/login', wait_until='networkidle')
            await self.anti_scraping.human_like_delay(1, 3)
            
            # Verificar si ya estamos logueados
            if await self.page.query_selector('input[aria-label="Buscar"]'):
                self.metrics._log("✅ Ya se encuentra logueado")
                return True
            
            # Verificar errores de LinkedIn
            content = await self.page.content()
            if self.error_handler.handle_linkedin_errors(content):
                self.metrics._log("⚠️ Error de LinkedIn detectado, reintentando...")
                raise Exception("Error de LinkedIn detectado")
            
            await self.page.fill('#username', email)
            await self.anti_scraping.human_like_delay(0.5, 1.5)
            
            await self.page.fill('#password', password)
            await self.anti_scraping.human_like_delay(0.5, 1.5)
            
            await self.page.click('button[type="submit"]')
            
            # Esperar con verificación de login exitoso
            try:
                await self.page.wait_for_selector('input[aria-label="Buscar"]', timeout=15000)
                self.metrics._log("✅ Sesión iniciada correctamente")
                return True
            except:
                # Verificar si hay error de login
                error_element = await self.page.query_selector('.alert-error')
                if error_element:
                    error_text = await error_element.inner_text()
                    raise Exception(f"Error en login: {error_text}")
                raise Exception("Timeout en login")
                
        except Exception as e:
            self.metrics._log(f"❌ Error en login: {e}", "ERROR")
            raise

    async def handle_popups(self):
        """Cerrar pop-ups emergentes"""
        try:
            await self.anti_scraping.human_like_delay(1, 2)
            selectors = ['.artdeco-modal__dismiss', '.msg-overlay-bubble-header__control', 'button[aria-label="Dismiss"]']
            for selector in selectors:
                try:
                    close_btn = await self.page.query_selector(selector)
                    if close_btn and await close_btn.is_visible():
                        await close_btn.click()
                        await self.anti_scraping.human_like_delay(0.5, 1)
                except:
                    continue
        except Exception as e:
            self.metrics._log(f"⚠️ Error manejando pop-ups: {e}", "WARNING")

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

    async def verify_author_location(self, author_url, country):
        """Verificar la ubicación del autor visitando su perfil"""
        if author_url == "N/A":
            return False
            
        try:
            new_page = await self.context.new_page()
            await new_page.goto(author_url, wait_until='networkidle')
            await self.anti_scraping.human_like_delay(2, 3)
            
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
            
            # Indicadores específicos para cada país
            country_indicators = {
                "United States": [
                    'usa', 'united states', 'estados unidos', 'us', 'u.s.', 'u.s.a.',
                    'new york', 'california', 'texas', 'florida', 'chicago', 'los angeles',
                    'san francisco', 'boston', 'seattle', 'washington', 'miami',
                    'ny', 'ca', 'tx', 'fl', 'il', 'wa', 'ma'
                ],
                "Colombia": [
                    'colombia', 'co', 'bogota', 'medellin', 'cali', 'barranquilla', 'cartagena',
                    'antioquia', 'cundinamarca', 'valle', 'atlantico', 'santander', 'boyaca',
                    'nariño', 'cordoba', 'tolima', 'huila', 'cauca', 'cesar', 'magdalena',
                    'meta', 'quindio', 'risaralda', 'norte de santander', 'bolivar', 'amazonas'
                ]
            }
            
            location_lower = location.lower()
            return any(indicator in location_lower for indicator in country_indicators[country])
            
        except Exception as e:
            self.metrics._log(f"⚠️ Error verificando ubicación del autor: {e}", "WARNING")
            return False

    def verify_content_location(self, post_data, country):
        """Verificar ubicación basada en el contenido del post"""
        text_content = f"{post_data['author_title']} {post_data['post_text']}".lower()
        
        # Palabras clave específicas para cada país
        country_keywords = {
            "United States": [
                'usa', 'united states', 'estados unidos', 'us', 'u.s.', 'u.s.a.',
                'new york', 'california', 'texas', 'florida', 'chicago', 'los angeles',
                'san francisco', 'boston', 'seattle', 'washington', 'miami',
                'ny', 'ca', 'tx', 'fl', 'il', 'wa', 'ma'
            ],
            "Colombia": [
                'colombia', 'co', 'bogota', 'medellin', 'cali', 'barranquilla', 'cartagena',
                'antioquia', 'cundinamarca', 'valle', 'atlantico', 'santander', 'boyaca',
                'nariño', 'cordoba', 'tolima', 'huila', 'cauca', 'cesar', 'magdalena'
            ]
        }
        
        return any(keyword in text_content for keyword in country_keywords[country])

    async def go_to_next_page(self):
        """Intenta navegar a la siguiente página de resultados"""
        try:
            # Buscar botón de siguiente página
            next_selectors = [
                'button[aria-label="Siguiente"]',
                'button[aria-label="Next"]',
                '.artdeco-pagination__button--next',
                'li.active + li.artdeco-pagination__indicator'
            ]
            
            for selector in next_selectors:
                next_button = await self.page.query_selector(selector)
                if next_button:
                    # Verificar si el botón está habilitado
                    is_disabled = await next_button.get_attribute('disabled')
                    if not is_disabled:
                        await next_button.click()
                        await self.page.wait_for_load_state('networkidle')
                        await self.anti_scraping.human_like_delay(3, 5)
                        return True
                    else:
                        return False
            
            return False
            
        except Exception as e:
            self.metrics._log(f"⚠️ Error navegando a siguiente página: {e}", "WARNING")
            return False

    async def process_search_term(self, search_term, country, country_info):
        """Procesa un término de búsqueda con paginación para un país específico"""
        try:
            self.metrics._log(f"🔍 Buscando publicaciones para: '{search_term}' en {country}")
            
            search_url = (
                f"https://www.linkedin.com/search/results/content/?"
                f"keywords={search_term.replace(' ', '%20')}"
                f"&facetGeoRegion={country_info['code']}"
                f"&facetContentType=%22posts%22"
            )
            
            await self.page.goto(search_url, wait_until='networkidle')
            await self.anti_scraping.human_like_delay(3, 5)
            
            await self.handle_popups()
            
            # Contador de páginas procesadas
            pages_processed = 0
            
            while pages_processed < self.max_pages:
                self.metrics._log(f"📄 Procesando página {pages_processed + 1}")
                
                try:
                    await self.page.wait_for_selector('.feed-shared-update-v2', timeout=10000)
                except:
                    self.metrics._log("⚠️ No se encontraron publicaciones después de 10 segundos")
                    break
                
                # Scroll para cargar más contenido
                await self.anti_scraping.human_like_scroll(self.page)
                
                # Extraer publicaciones de la página actual
                posts = await self.page.query_selector_all('.feed-shared-update-v2')
                self.metrics.increment('posts_found', len(posts))
                
                if not posts:
                    self.metrics._log("⚠️ No se encontraron publicaciones en esta página")
                    break
                    
                self.metrics._log(f"✅ Encontradas {len(posts)} publicaciones. Procesando...")
                
                posts_processed = 0
                for i, post in enumerate(posts):
                    # Verificar límite por término
                    current_term_count = len([r for r in self.results if r['search_term'] == search_term and r['country'] == country])
                    if current_term_count >= self.max_posts_per_term:
                        self.metrics._log(f"⚡ Límite de {self.max_posts_per_term} publicaciones alcanzado para '{search_term}' en {country}")
                        return True
                    
                    try:
                        self.metrics._log(f"  🔍 Procesando publicación {i+1}/{len(posts)}")
                        post_data = await self.extract_post_data(post)
                        # Añadir término de búsqueda, país y ubicación filtrada
                        post_data['search_term'] = search_term
                        post_data['country'] = country
                        post_data['location'] = random.choice(country_info['location_names'])
                        
                        if not self.is_desired_language(post_data['post_text']):
                            self.metrics._log(f"  ⚠️ Publicación en idioma no deseado - Descartada")
                            self.metrics.increment('posts_filtered')
                            continue
                        
                        location_verified = self.verify_content_location(post_data, country)
                        
                        if not location_verified and post_data['author_url'] != "N/A":
                            self.metrics._log(f"  🔍 Verificando ubicación en perfil del autor...")
                            location_verified = await self.verify_author_location(post_data['author_url'], country)
                        
                        if location_verified:
                            self.results.append(post_data)
                            posts_processed += 1
                            self.metrics.increment('posts_added')
                            self.metrics.increment('countries_processed', 1, country)
                            self.metrics.increment('terms_processed', 1, search_term)
                            self.metrics._log(f"  ✅ Publicación añadida")
                        else:
                            self.metrics._log(f"  ⚠️ Publicación fuera de {country} - Descartada")
                            self.metrics.increment('posts_filtered')
                    except Exception as e:
                        self.metrics._log(f"  ⚠️ Error en publicación {i+1}: {e}", "WARNING")
                        self.metrics.increment('errors_count')
                
                # Intentar ir a la siguiente página
                if not await self.go_to_next_page():
                    self.metrics._log("✅ No hay más páginas disponibles")
                    break
                
                pages_processed += 1
                
            return True
        except Exception as e:
            self.metrics._log(f"❌ Error procesando búsqueda '{search_term}' en {country}: {e}", "ERROR")
            self.metrics.increment('errors_count')
            return False

    async def search_and_extract_posts(self):
        try:
            self.metrics._log(f"🔍 Iniciando búsqueda con {len(self.search_terms)} términos en {len(self.countries)} países...")
            
            for country_name, country_info in self.countries.items():
                self.metrics._log(f"\n{'='*60}")
                self.metrics._log(f"🌎 PAÍS: {country_name}")
                self.metrics._log(f"{'='*60}")
                
                for i, search_term in enumerate(self.search_terms):
                    self.metrics._log(f"\n  🔍 Término {i+1}/{len(self.search_terms)}: '{search_term}'")
                    
                    success = await self.process_search_term(search_term, country_name, country_info)
                    
                    if not success:
                        self.metrics._log(f"  ⚠️ Error procesando término: '{search_term}'")
                        self.metrics.increment('errors_count')
                    
                    # Pausa entre términos de búsqueda
                    if i < len(self.search_terms) - 1:
                        pause_time = random.randint(8, 15)
                        self.metrics._log(f"  ⏳ Pausa de {pause_time} segundos antes del próximo término...")
                        await asyncio.sleep(pause_time)
                
                # Pausa más larga entre países
                if country_name != list(self.countries.keys())[-1]:
                    pause_time = random.randint(15, 25)
                    self.metrics._log(f"\n🌎 Pausa de {pause_time} segundos antes del próximo país...")
                    await asyncio.sleep(pause_time)
            
            return len(self.results) > 0
        except Exception as e:
            self.metrics._log(f"❌ Error en búsqueda: {e}", "ERROR")
            self.metrics.increment('errors_count')
            return False

    async def extract_post_data(self, post):
        # Extraer título del autor
        author_title = "N/A"
        try:
            title_element = await post.query_selector('.update-components-actor__description')
            author_title = await title_element.inner_text() if title_element else "N/A"
        except:
            pass
        
        # Extraer URL del autor (para verificación de ubicación)
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
        
        # Extraer texto del post
        post_text = "N/A"
        try:
            text_element = await post.query_selector('.update-components-text')
            if text_element:
                post_text = await text_element.inner_text()
                post_text = post_text[:500]  # Limitar longitud
        except:
            pass
        
        # Extraer fecha del post
        post_date = "N/A"
        try:
            date_element = await post.query_selector('.update-components-actor__sub-description')
            if date_element:
                post_date = await date_element.inner_text()
                post_date = re.sub(r'•.*', '', post_date).strip()
        except:
            pass
        
        return {
            'author_title': author_title,
            'post_text': post_text,
            'post_date': post_date,
            'scraped_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'author_url': author_url  # Mantenemos temporalmente para verificación
        }

    def remove_duplicates_smart(self, df):
        """Elimina duplicados de manera inteligente"""
        df = df.copy()
        
        # Crear columna de ID único basado en contenido
        df['unique_id'] = df.apply(
            lambda row: f"{row['author_title']}_{row['post_text'][:100]}_{row['post_date']}", 
            axis=1
        )
        
        # Eliminar duplicados manteniendo el último
        df = df.drop_duplicates(subset=['unique_id'], keep='last')
        
        # Eliminar la columna temporal
        df = df.drop(columns=['unique_id'])
        
        return df

    async def save_to_csv_and_excel(self):
        if not self.results:
            self.metrics._log("❌ No hay datos para guardar")
            return
            
        new_df = pd.DataFrame(self.results)
        
        # Eliminar la columna author_url (según lo solicitado)
        if 'author_url' in new_df.columns:
            new_df = new_df.drop(columns=['author_url'])
        
        # Reordenar columnas según lo solicitado
        column_order = ['search_term', 'country', 'location', 'author_title', 'post_text', 'post_date', 'scraped_date']
        new_df = new_df.reindex(columns=column_order)
        
        individual_csv = f"linkedin_bim_posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        new_df.to_csv(individual_csv, index=False, encoding='utf-8')
        self.metrics._log(f"✅ Archivo individual guardado en {individual_csv}")
        
        # Guardar/actualizar archivo consolidado CSV
        if os.path.exists(self.consolidated_csv):
            existing_df = pd.read_csv(self.consolidated_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = self.remove_duplicates_smart(combined_df)
            combined_df.to_csv(self.consolidated_csv, index=False, encoding='utf-8')
            self.metrics._log(f"✅ Archivo consolidado CSV actualizado: {self.consolidated_csv}")
        else:
            new_df.to_csv(self.consolidated_csv, index=False, encoding='utf-8')
            self.metrics._log(f"✅ Nuevo archivo consolidado CSV creado: {self.consolidated_csv}")
        
        # Guardar/actualizar archivo consolidado Excel
        if os.path.exists(self.consolidated_excel):
            existing_df = pd.read_excel(self.consolidated_excel)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = self.remove_duplicates_smart(combined_df)
            combined_df.to_excel(self.consolidated_excel, index=False)
            self.metrics._log(f"✅ Archivo consolidado Excel actualizado: {self.consolidated_excel}")
        else:
            new_df.to_excel(self.consolidated_excel, index=False)
            self.metrics._log(f"✅ Nuevo archivo consolidado Excel creado: {self.consolidated_excel}")
        
        self.metrics._log(f"📊 Publicaciones nuevas: {len(new_df)}")
        if os.path.exists(self.consolidated_csv):
            consolidated_df = pd.read_csv(self.consolidated_csv)
            self.metrics._log(f"📊 Total de publicaciones en consolidado: {len(consolidated_df)}")

    async def run_scraper(self, email: str, password: str):
        self.metrics.start_timer()
        try:
            await self.setup_browser()
            if await self.login_to_linkedin(email, password):
                await self.handle_popups()
                if await self.search_and_extract_posts():
                    await self.save_to_csv_and_excel()
        except Exception as e:
            self.metrics.increment('errors_count')
            self.metrics._log(f"Error general: {e}", "ERROR")
        finally:
            try:
                await self.browser.close()
                await self.playwright.stop()
                self.metrics._log("Navegador cerrado correctamente")
            except Exception as e:
                self.metrics._log(f"Error cerrando navegador: {e}", "ERROR")
            finally:
                self.metrics.end_timer()
                self.metrics.generate_report()

# ==================== EJECUCIÓN PRINCIPAL ====================

async def main():
    print("=" * 60)
    print("🔍 LINKEDIN BIM SCRAPER MEJORADO - MULTIPAÍS")
    print("📝 Términos de búsqueda:")
    for i, term in enumerate(["BIM Modeler", "BIM Manager", "BIM Coordinator", "Revit Specialist", "3D Modeler", "VDC Engineer", "BIM Technician", "BIM Specialist", "BIM Designer", "Digital Construction", "Virtual Design Construction", "AEC Technology", "Building Information Modeling"]):
        print(f"   {i+1}. {term}")
    print("🌎 Países: Estados Unidos, Colombia")
    print("📑 Paginación: Activada (2 páginas por término)")
    print("🛡️  Anti-scraping: Activado")
    print("📊 Métricas: Activadas")
    print("=" * 60)
    
    scraper = LinkedInBIMScraper()
    
    email = "ussaapontejuandiego@gmail.com"
    password = "juandi778"
    
    if email == "tu_email@example.com":
        print("❌ Configura tus credenciales reales")
        return
    
    await scraper.run_scraper(email, password)
    print("🏁 Proceso completado")

if __name__ == "__main__":
    asyncio.run(main())