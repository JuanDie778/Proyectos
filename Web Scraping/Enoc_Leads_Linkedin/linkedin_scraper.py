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
from typing import List, Dict, Optional

# ==================== CLASES DE APOYO SIMPLIFICADAS ====================

class ExecutionMetrics:
    """Sistema de métricas simplificado"""
    
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
            'india_posts_filtered': 0  # Nueva métrica para posts de India filtrados
        }
    
    def start_timer(self):
        self.metrics['start_time'] = time.time()
        self._log("🚀 Iniciando ejecución del scraper")
    
    def end_timer(self):
        self.metrics['end_time'] = time.time()
        self.metrics['total_duration'] = self.metrics['end_time'] - self.metrics['start_time']
        self._log(f"⏰ Duración total: {self.metrics['total_duration']:.2f} segundos")
    
    def increment(self, metric_name, value=1, details=None):
        if metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], dict) and details:
                if details not in self.metrics[metric_name]:
                    self.metrics[metric_name][details] = 0
                self.metrics[metric_name][details] += value
            else:
                self.metrics[metric_name] += value
    
    def _log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
    
    def generate_report(self):
        report = [
            "\n" + "="*60,
            "📊 REPORTE DE EJECUCIÓN DEL SCRAPER",
            "="*60,
            f"⏰ Duración total: {self.metrics['total_duration']:.2f} segundos",
            f"🌎 Países procesados: {len(self.metrics['countries_processed'])}",
            f"🔍 Términos procesados: {len(self.metrics['terms_processed'])}",
            f"📄 Publicaciones encontradas: {self.metrics['posts_found']}",
            f"✅ Publicaciones añadidas: {self.metrics['posts_added']}",
            f"🚫 Publicaciones filtradas: {self.metrics['posts_filtered']}",
            f"🇮🇳 Publicaciones de India filtradas: {self.metrics['india_posts_filtered']}",
            f"❌ Errores: {self.metrics['errors_count']}",
            "="*60
        ]
        
        if self.metrics['countries_processed']:
            report.append("\n📈 Detalles por país:")
            for country, count in self.metrics['countries_processed'].items():
                report.append(f"   {country}: {count} publicaciones")
        
        if self.metrics['terms_processed']:
            report.append("\n📈 Detalles por término:")
            for term, count in self.metrics['terms_processed'].items():
                report.append(f"   '{term}': {count} publicaciones")
        
        report_text = "\n".join(report)
        self._log(report_text)
        return report_text

class ErrorHandler:
    """Manejador simplificado de errores"""
    
    @staticmethod
    def async_retry(max_retries=3, delay=2, backoff=2, exceptions=(Exception,)):
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
                        
                        print(f"⚠️ Intento {retries}/{max_retries} fallido. Reintentando en {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# ==================== CLASE PRINCIPAL MEJORADA ====================

class LinkedInBIMScraperFixed:
    def __init__(self):
        # TÉRMINOS DE BÚSQUEDA
        self.search_terms = [
            "BIM Modeler", "BIM Manager", "BIM Coordinator", "BIM Specialist",
            "BIM Designer", "BIM Engineer", "BIM Architect", "Revit Specialist",
            "Revit Modeler", "Revit Designer", "3D BIM", "VDC Engineer",
            "Building Information Modeling", "BIM Consultant", "BIM Project Manager"
        ]
        
        # PAÍSES
        self.countries = {
            "United States": {
                "code": "%5B%22urn%3Ali%3Afs_geo%3A103644278%22%5D",
                "location_names": ["USA", "EEUU", "United States", "Estados Unidos"]
            },
            "Canada": {
                "code": "%5B%22urn%3Ali%3Afs_geo%3A101174742%22%5D",
                "location_names": ["Canada", "Canadá"]
            },
            "United Kingdom": {
                "code": "%5B%22urn%3Ali%3Afs_geo%3A101165590%22%5D",
                "location_names": ["UK", "United Kingdom", "Reino Unido"]
            },
            "Spain": {
                "code": "%5B%22urn%3Ali%3Afs_geo%3A105646813%22%5D",
                "location_names": ["Spain", "España"]
            },
            "Colombia": {
                "code": "%5B%22urn%3Ali%3Afs_geo%3A100876405%22%5D",
                "location_names": ["Colombia", "CO", "Bogotá", "Medellín", "Cali"]
            }
        }
        
        self.results = []
        self.max_posts_per_term = 40  # Límite más conservador
        self.max_pages = 5  # Páginas más conservadoras
        self.max_posts_total = 1000  # Límite total más realista
        
        self.consolidated_csv = "linkedin_bim_posts_consolidated.csv"
        self.consolidated_excel = "linkedin_bim_posts_consolidated.xlsx"
        
        # Componentes simplificados
        self.error_handler = ErrorHandler()
        self.metrics = ExecutionMetrics()

    def is_india(self, text):
        """Detectar si el texto contiene indicadores de la India"""
        if text == "N/A" or not text:
            return False
            
        india_indicators = [
            'india', 'indian', 'bangalore', 'mumbai', 'delhi', 'chennai', 
            'hyderabad', 'kolkata', 'pune', 'bengaluru', 'noida', 'gurgaon',
            'ahmedabad', 'india\'s', 'from india', 'based in india', 'located in india',
            'bombay', 'madras', 'calcutta', 'new delhi', 'india based', 'working in india',
            '🇮🇳', 'indianapolis'  # Indianapolis podría ser falso positivo, pero es poco común
        ]
        
        text_lower = text.lower()
        # Verificar múltiples indicadores para reducir falsos positivos
        india_count = sum(1 for indicator in india_indicators if indicator in text_lower)
        return india_count >= 2  # Requerir al menos 2 indicadores para mayor precisión

    async def setup_browser(self):
        """Configuración de navegador SIMPLIFICADA que funciona"""
        self.playwright = await async_playwright().start()
        
        # Configuración SIMPLE como el código base que funciona
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--start-maximized']  # Solo lo básico
        )
        
        # Configuración básica sin scripts anti-detección problemáticos
        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            viewport={'width': 1366, 'height': 768}
        )
        
        self.page = await self.context.new_page()

    @ErrorHandler.async_retry(max_retries=3, delay=2, backoff=2)
    async def login_to_linkedin(self, email: str, password: str):
        try:
            self.metrics._log("🔐 Iniciando sesión en LinkedIn...")
            await self.page.goto('https://www.linkedin.com/login', wait_until='networkidle')
            await asyncio.sleep(random.uniform(1, 3))
            
            # Verificar si ya estamos logueados
            try:
                await self.page.wait_for_selector('input[aria-label="Buscar"]', timeout=5000)
                self.metrics._log("✅ Ya se encuentra logueado")
                return True
            except:
                pass  # Continuar con el login
            
            await self.page.fill('#username', email)
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            await self.page.fill('#password', password)
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
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
        """Cerrar pop-ups emergentes - versión simplificada"""
        try:
            await asyncio.sleep(2)
            selectors = ['.artdeco-modal__dismiss', '.msg-overlay-bubble-header__control', 'button[aria-label="Dismiss"]']
            for selector in selectors:
                try:
                    close_btn = await self.page.query_selector(selector)
                    if close_btn and await close_btn.is_visible():
                        await close_btn.click()
                        await asyncio.sleep(1)
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
        """Verificar la ubicación del autor visitando su perfil - usa author_url SIN eliminarlo"""
        if author_url == "N/A":
            return False
            
        try:
            new_page = await self.context.new_page()
            await new_page.goto(author_url, wait_until='networkidle')
            await asyncio.sleep(2)
            
            location_selectors = [
                '.text-body-small.inline.t-black--light.break-words',
                '.pv-top-card--list-bullet li',
                '.pv-top-card-v2--list-bullet li'
            ]
            
            location = "N/A"
            for selector in location_selectors:
                location_element = await new_page.query_selector(selector)
                if location_element:
                    location = await location_element.inner_text()
                    if location and location.strip():
                        break
            
            await new_page.close()
            
            # Verificar primero si es India
            if self.is_india(location):
                return False
            
            # Indicadores para cada país
            country_indicators = {
                "United States": ['usa', 'united states', 'estados unidos', 'us', 'u.s.', 'new york', 'california', 'texas'],
                "Canada": ['canada', 'canadá', 'toronto', 'vancouver', 'montreal'],
                "United Kingdom": ['uk', 'united kingdom', 'london', 'manchester', 'england'],
                "Spain": ['spain', 'españa', 'madrid', 'barcelona', 'valencia'],
                "Colombia": ['colombia', 'bogota', 'bogotá', 'medellin', 'medellín', 'cali', 'barranquilla', 'cartagena']
            }
            
            location_lower = location.lower()
            return any(indicator in location_lower for indicator in country_indicators.get(country, []))
            
        except Exception as e:
            self.metrics._log(f"⚠️ Error verificando ubicación del autor: {e}", "WARNING")
            return False

    def verify_content_location(self, post_data, country):
        """Verificar ubicación basada en el contenido del post"""
        text_content = f"{post_data['author_title']} {post_data['post_text']}".lower()
        
        # Verificar primero si es India
        if self.is_india(text_content):
            return False
        
        country_keywords = {
            "United States": ['usa', 'united states', 'estados unidos', 'us', 'u.s.', 'new york', 'california', 'texas'],
            "Canada": ['canada', 'canadá', 'toronto', 'vancouver', 'montreal'],
            "United Kingdom": ['uk', 'united kingdom', 'london', 'manchester', 'england'],
            "Spain": ['spain', 'españa', 'madrid', 'barcelona', 'valencia'],
            "Colombia": ['colombia', 'bogota', 'bogotá', 'medellin', 'medellín', 'cali', 'barranquilla', 'cartagena']
        }
        
        return any(keyword in text_content for keyword in country_keywords.get(country, []))

    async def go_to_next_page(self):
        """Navegar a la siguiente página - versión simplificada"""
        try:
            next_selectors = [
                'button[aria-label="Siguiente"]',
                'button[aria-label="Next"]',
                '.artdeco-pagination__button--next'
            ]
            
            for selector in next_selectors:
                next_button = await self.page.query_selector(selector)
                if next_button:
                    is_disabled = await next_button.get_attribute('disabled')
                    if not is_disabled:
                        await next_button.click()
                        await self.page.wait_for_load_state('networkidle')
                        await asyncio.sleep(random.uniform(3, 5))
                        return True
                    else:
                        return False
            
            return False
            
        except Exception as e:
            self.metrics._log(f"⚠️ Error navegando a siguiente página: {e}", "WARNING")
            return False

    async def process_search_term(self, search_term, country, country_info):
        """Procesa un término de búsqueda - versión mejorada pero simplificada"""
        try:
            if len(self.results) >= self.max_posts_total:
                self.metrics._log(f"⚡ Límite global de {self.max_posts_total} publicaciones alcanzado")
                return False
                
            self.metrics._log(f"🔍 Buscando publicaciones para: '{search_term}' en {country}")
            
            search_url = (
                f"https://www.linkedin.com/search/results/content/?"
                f"keywords={search_term.replace(' ', '%20')}"
                f"&facetGeoRegion={country_info['code']}"
                f"&facetContentType=%22posts%22"
            )
            
            await self.page.goto(search_url, wait_until='networkidle')
            await asyncio.sleep(random.uniform(3, 5))
            
            await self.handle_popups()
            
            pages_processed = 0
            
            while pages_processed < self.max_pages:
                if len(self.results) >= self.max_posts_total:
                    self.metrics._log(f"⚡ Límite global alcanzado")
                    return True
                    
                self.metrics._log(f"📄 Procesando página {pages_processed + 1}")
                
                try:
                    await self.page.wait_for_selector('.feed-shared-update-v2', timeout=10000)
                except:
                    self.metrics._log("⚠️ No se encontraron publicaciones")
                    break
                
                # Scroll simple
                for i in range(3):
                    await self.page.evaluate('''
                        window.scrollTo({
                            top: document.body.scrollHeight,
                            behavior: 'smooth'
                        })
                    ''')
                    await asyncio.sleep(random.uniform(2, 4))
                
                posts = await self.page.query_selector_all('.feed-shared-update-v2')
                self.metrics.increment('posts_found', len(posts))
                
                if not posts:
                    break
                    
                self.metrics._log(f"✅ Encontradas {len(posts)} publicaciones. Procesando...")
                
                for i, post in enumerate(posts):
                    if len(self.results) >= self.max_posts_total:
                        return True
                        
                    current_term_count = len([r for r in self.results if r['search_term'] == search_term and r['country'] == country])
                    if current_term_count >= self.max_posts_per_term:
                        self.metrics._log(f"⚡ Límite por término alcanzado para '{search_term}' en {country}")
                        return True
                    
                    try:
                        self.metrics._log(f"  🔍 Procesando publicación {i+1}/{len(posts)}")
                        post_data = await self.extract_post_data(post)
                        post_data['search_term'] = search_term
                        post_data['country'] = country
                        post_data['location'] = random.choice(country_info['location_names'])
                        
                        # Verificar si es de India
                        combined_text = f"{post_data['author_title']} {post_data['post_text']} {post_data['location']}"
                        if self.is_india(combined_text):
                            self.metrics._log(f"  🇮🇳 Publicación de India detectada y filtrada")
                            self.metrics.increment('posts_filtered')
                            self.metrics.increment('india_posts_filtered')
                            continue
                        
                        if not self.is_desired_language(post_data['post_text']):
                            self.metrics._log(f"  ⚠️ Publicación en idioma no deseado")
                            self.metrics.increment('posts_filtered')
                            continue
                        
                        location_verified = self.verify_content_location(post_data, country)
                        
                        if not location_verified and post_data.get('author_url', "N/A") != "N/A":
                            location_verified = await self.verify_author_location(post_data['author_url'], country)
                        
                        if location_verified:
                            self.results.append(post_data)  # MANTENER TODAS LAS URLs
                            self.metrics.increment('posts_added')
                            self.metrics.increment('countries_processed', 1, country)
                            self.metrics.increment('terms_processed', 1, search_term)
                            self.metrics._log(f"  ✅ Publicación añadida - Post: {post_data.get('post_url', 'N/A')}")
                        else:
                            self.metrics._log(f"  ⚠️ Publicación fuera de {country}")
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
        """Búsqueda principal - versión mejorada"""
        try:
            self.metrics._log(f"🔍 Iniciando búsqueda con {len(self.search_terms)} términos en {len(self.countries)} países...")
            
            for country_name, country_info in self.countries.items():
                self.metrics._log(f"\n{'='*60}")
                self.metrics._log(f"🌎 PAÍS: {country_name}")
                self.metrics._log(f"{'='*60}")
                
                for i, search_term in enumerate(self.search_terms):
                    if len(self.results) >= self.max_posts_total:
                        self.metrics._log(f"⚡ Límite global alcanzado")
                        return True
                        
                    self.metrics._log(f"\n  🔍 Término {i+1}/{len(self.search_terms)}: '{search_term}'")
                    
                    success = await self.process_search_term(search_term, country_name, country_info)
                    
                    if not success:
                        self.metrics._log(f"  ⚠️ Error procesando término: '{search_term}'")
                    
                    # Guardar cada 25 publicaciones
                    if len(self.results) % 25 == 0 and len(self.results) > 0:
                        await self.save_to_csv_and_excel()
                    
                    # Pausa entre términos
                    if i < len(self.search_terms) - 1:
                        pause_time = random.randint(5, 10)
                        self.metrics._log(f"  ⏳ Pausa de {pause_time} segundos...")
                        await asyncio.sleep(pause_time)
                
                # Pausa entre países
                if country_name != list(self.countries.keys())[-1]:
                    pause_time = random.randint(10, 15)
                    self.metrics._log(f"\n🌎 Pausa de {pause_time} segundos antes del próximo país...")
                    await asyncio.sleep(pause_time)
            
            return len(self.results) > 0
        except Exception as e:
            self.metrics._log(f"❌ Error en búsqueda: {e}", "ERROR")
            self.metrics.increment('errors_count')
            return False

    def extract_url_from_data_urn(self, data_urn):
        """Extraer URL del post desde data-urn"""
        if data_urn and data_urn.startswith('urn:li:activity:'):
            activity_id = data_urn.split(':')[-1]
            return f"https://www.linkedin.com/feed/update/{activity_id}/"
        return None

    async def extract_post_data(self, post):
        """Extraer datos del post - COMPLETO con nombre, URLs y toda la información"""
        # Extraer NOMBRE del autor
        author_name = "N/A"
        try:
            author_element = await post.query_selector('.update-components-actor__name')
            author_name = await author_element.inner_text() if author_element else "N/A"
        except:
            pass
        
        # Extraer título del autor
        author_title = "N/A"
        try:
            title_element = await post.query_selector('.update-components-actor__description')
            author_title = await title_element.inner_text() if title_element else "N/A"
        except:
            pass
        
        # Extraer URL del PERFIL del autor
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
        
        # Extraer URL del POST
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
            self.metrics._log(f"⚠️ Error extrayendo URL de publicación: {e}", "WARNING")
        
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
        
        # Extraer ubicación mostrada en el post
        location = "N/A"
        try:
            location_element = await post.query_selector('.update-components-actor__sub-description .update-components-actor__distance')
            if location_element:
                location = await location_element.inner_text()
        except:
            pass
        
        return {
            'author_name': author_name,        # RECUPERADO
            'author_title': author_title,
            'author_url': author_url,          # MANTENIDO - URL al perfil
            'post_url': post_url,              # RECUPERADO - URL al post
            'post_text': post_text,
            'post_date': post_date,
            'location': location,              # RECUPERADO
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
        """Elimina duplicados de manera inteligente MANTENIENDO concatenación"""
        if len(df) == 0:
            return df
            
        df = df.copy()
        
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
        """CONCATENACIÓN CORREGIDA - Funciona como el código original"""
        if not self.results:
            self.metrics._log("❌ No hay datos para guardar")
            return
            
        new_df = pd.DataFrame(self.results)
        
        # Reordenar columnas INCLUYENDO todas las URLs
        column_order = ['search_term', 'country', 'location', 'author_name', 'author_title', 
                       'author_url', 'post_url', 'post_text', 'post_date', 'scraped_date']
        new_df = new_df.reindex(columns=column_order)
        
        # Archivo individual (como respaldo)
        individual_csv = f"linkedin_bim_posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        new_df.to_csv(individual_csv, index=False, encoding='utf-8')
        self.metrics._log(f"✅ Archivo individual guardado en {individual_csv}")
        
        # CONCATENACIÓN CORRECTA - Archivo consolidado CSV
        if os.path.exists(self.consolidated_csv):
            self.metrics._log("📂 Leyendo archivo consolidado existente...")
            existing_df = pd.read_csv(self.consolidated_csv)
            self.metrics._log(f"📊 Publicaciones existentes: {len(existing_df)}")
            
            # CONCATENAR nuevos datos con existentes
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            self.metrics._log(f"📊 Total después de concatenar: {len(combined_df)}")
            
            # Eliminar duplicados
            combined_df = self.remove_duplicates_smart(combined_df)
            self.metrics._log(f"📊 Total después de eliminar duplicados: {len(combined_df)}")
            
            # Guardar archivo concatenado
            combined_df.to_csv(self.consolidated_csv, index=False, encoding='utf-8')
            self.metrics._log(f"✅ Archivo consolidado CSV ACTUALIZADO: {self.consolidated_csv}")
        else:
            # Crear nuevo archivo consolidado
            new_df.to_csv(self.consolidated_csv, index=False, encoding='utf-8')
            self.metrics._log(f"✅ Nuevo archivo consolidado CSV creado: {self.consolidated_csv}")
        
        # CONCATENACIÓN CORRECTA - Archivo consolidado Excel
        if os.path.exists(self.consolidated_excel):
            self.metrics._log("📂 Leyendo archivo Excel consolidado existente...")
            existing_df = pd.read_excel(self.consolidated_excel)
            
            # CONCATENAR nuevos datos con existentes
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Eliminar duplicados
            combined_df = self.remove_duplicates_smart(combined_df)
            
            # Guardar archivo concatenado
            combined_df.to_excel(self.consolidated_excel, index=False)
            self.metrics._log(f"✅ Archivo consolidado Excel ACTUALIZADO: {self.consolidated_excel}")
        else:
            # Crear nuevo archivo consolidado
            new_df.to_excel(self.consolidated_excel, index=False)
            self.metrics._log(f"✅ Nuevo archivo consolidado Excel creado: {self.consolidated_excel}")
        
        # Reporte de estado
        self.metrics._log(f"📊 Publicaciones nuevas agregadas: {len(new_df)}")
        if os.path.exists(self.consolidated_csv):
            final_df = pd.read_csv(self.consolidated_csv)
            self.metrics._log(f"📊 TOTAL de publicaciones en archivo consolidado: {len(final_df)}")
            self.metrics._log(f"💾 Archivo principal: {self.consolidated_csv}")
            self.metrics._log(f"💾 Archivo Excel: {self.consolidated_excel}")

    async def run_scraper(self, email: str, password: str):
        """Ejecutar scraper - versión mejorada"""
        self.metrics.start_timer()
        
        max_attempts = 2  # Reducir intentos
        attempt = 0
        
        while attempt < max_attempts:
            try:
                attempt += 1
                self.metrics._log(f"🔄 Intento {attempt}/{max_attempts}")
                
                await self.setup_browser()
                if await self.login_to_linkedin(email, password):
                    await self.handle_popups()
                    if await self.search_and_extract_posts():
                        await self.save_to_csv_and_excel()
                        break
                        
            except Exception as e:
                self.metrics.increment('errors_count')
                self.metrics._log(f"Error en intento {attempt}: {e}", "ERROR")
                
                if attempt >= max_attempts:
                    self.metrics._log("❌ Todos los intentos fallidos", "ERROR")
                else:
                    wait_time = 30 * attempt
                    self.metrics._log(f"⏳ Esperando {wait_time} segundos antes de reintentar...")
                    await asyncio.sleep(wait_time)
                    
            finally:
                try:
                    await self.browser.close()
                    await self.playwright.stop()
                    self.metrics._log("Navegador cerrado correctamente")
                except Exception as e:
                    self.metrics._log(f"Error cerrando navegador: {e}", "ERROR")
        
        self.metrics.end_timer()
        self.metrics.generate_report()

# ==================== EJECUCIÓN PRINCIPAL ====================

async def main():
    print("=" * 80)
    print("🔍 LINKEDIN BIM SCRAPER MEJORADO - VERSIÓN HÍBRIDA")
    print("🔍 Términos de búsqueda: 15 términos BIM optimizados")
    print("🌎 Países: 5 países principales")
    print("📄 Paginación: 3 páginas por término")
    print("📊 Límite: 500 publicaciones máximas")
    print("🛡️ Anti-scraping: Simplificado y funcional")
    print("🇮🇳 Filtro India: Activado - Detecta y filtra publicaciones de India")
    print("📈 Métricas: Activadas")
    print("=" * 80)
    
    scraper = LinkedInBIMScraperFixed()
    
    # CAMBIAR ESTAS CREDENCIALES POR LAS TUYAS
    email = "ussaapontejuandiego@gmail.com"
    password = "juandi778"
    
    if email == "tu_email@example.com":
        print("❌ Configura tus credenciales reales")
        return
    
    # Ejecutar el scraper
    await scraper.run_scraper(email, password)
    
    # Mensaje final
    print("=" * 80)
    print("🎉 PROCESO COMPLETADO")
    print(f"📊 Publicaciones obtenidas: {len(scraper.results)}")
    print(f"🇮🇳 Publicaciones de India filtradas: {scraper.metrics.metrics['india_posts_filtered']}")
    print("💾 Datos guardados en archivos CSV y Excel")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())