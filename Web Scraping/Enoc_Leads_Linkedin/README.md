# LinkedIn BIM Scraper - Generador de Leads para Enoc

## Descripción General del Proyecto

Este proyecto es una herramienta avanzada de web scraping diseñada para científicos de datos especializados en la generación de leads y análisis de datos en el sector de la construcción y modelado de información de edificaciones (BIM). El scraper automatiza la extracción de publicaciones de LinkedIn relacionadas con profesionales BIM, aplicando técnicas sofisticadas de procesamiento de texto, clasificación geográfica y gestión de datos para crear un conjunto de datos limpio y valioso para análisis posteriores.

Desde una perspectiva de ciencia de datos, este proyecto demuestra la aplicación práctica de metodologías de extracción de datos web, filtrado inteligente basado en contenido textual, verificación de ubicación mediante análisis semántico y estrategias de deduplicación para mantener la integridad de los datos. El enfoque se centra en la calidad sobre la cantidad, implementando filtros multi-nivel para asegurar que los datos recopilados sean relevantes y precisos para el mercado objetivo.

## Características Principales

### Extracción Inteligente de Datos
- **Búsqueda Multi-Término**: 15 términos especializados en BIM (BIM Modeler, BIM Manager, BIM Coordinator, etc.)
- **Cobertura Geográfica**: 5 países estratégicos (Estados Unidos, Canadá, Reino Unido, España, Colombia)
- **Paginación Automática**: Navegación inteligente a través de múltiples páginas de resultados
- **Extracción Completa**: Captura de nombre del autor, título profesional, URLs de perfil y publicación, texto del post, fecha y ubicación

### Procesamiento de Texto Avanzado
- **Detección de Idioma**: Utiliza la biblioteca `langid` para identificar idiomas no deseados (árabe, hindi, urdu, etc.)
- **Filtrado de Contenido**: Eliminación automática de publicaciones irrelevantes basadas en análisis textual
- **Verificación Semántica**: Análisis de contenido para confirmar relevancia geográfica

### Clasificación y Filtrado Geográfico
- **Filtro Anti-India**: Algoritmo especializado para detectar y excluir publicaciones de origen indio
- **Verificación de Ubicación**: Doble validación mediante análisis de contenido y visita a perfiles de autor
- **Mapeo de Indicadores**: Base de datos de indicadores geográficos por país para clasificación precisa

### Gestión de Datos Profesional
- **Deduplicación Inteligente**: Sistema de hashing basado en URLs y contenido para eliminar duplicados
- **Consolidación de Datos**: Concatenación automática de nuevos datos con archivos existentes
- **Múltiples Formatos**: Exportación a CSV y Excel con codificación UTF-8
- **Métricas de Ejecución**: Seguimiento detallado de rendimiento y estadísticas de procesamiento

### Automatización Robusta
- **Manejo de Errores**: Decorador de reintento con backoff exponencial
- **Anti-Detección**: Técnicas de simulación humana (pausas aleatorias, navegación natural)
- **Gestión de Sesiones**: Autenticación automática y manejo de pop-ups
- **Límites de Seguridad**: Controles de velocidad y límites de extracción para evitar bloqueos

## Instalación

### Requisitos del Sistema
- Python 3.8+
- Navegador Chromium (incluido con Playwright)
- Conexión a internet estable

### Dependencias
```bash
pip install playwright pandas langid asyncio
```

### Configuración Inicial
```bash
# Instalar navegadores de Playwright
playwright install chromium

# Clonar o descargar el proyecto
# Colocar en el directorio deseado
```

## Uso

### Configuración de Credenciales
Edite las variables de email y password en la función `main()` del archivo `linkedin_scraper.py`:

```python
email = "tu_email@ejemplo.com"
password = "tu_password_seguro"
```

### Ejecución del Scraper
```bash
python linkedin_scraper.py
```

### Parámetros de Configuración
- `max_posts_per_term`: Límite de publicaciones por término de búsqueda (por defecto: 40)
- `max_pages`: Número máximo de páginas por búsqueda (por defecto: 5)
- `max_posts_total`: Límite global de publicaciones (por defecto: 1000)

### Archivos de Salida
- `linkedin_bim_posts_consolidated.csv`: Archivo principal consolidado
- `linkedin_bim_posts_consolidated.xlsx`: Versión Excel del archivo consolidado
- Archivos individuales con timestamp para respaldo

## Arquitectura del Sistema

### Arquitectura General
```
LinkedInBIMScraperFixed
├── ExecutionMetrics: Sistema de métricas y logging
├── ErrorHandler: Manejo de errores con reintentos
├── LinkedInBIMScraperFixed: Clase principal
│   ├── setup_browser(): Configuración de Playwright
│   ├── login_to_linkedin(): Autenticación
│   ├── search_and_extract_posts(): Bucle principal de búsqueda
│   ├── process_search_term(): Procesamiento por término
│   ├── extract_post_data(): Extracción de datos del DOM
│   ├── save_to_csv_and_excel(): Persistencia de datos
│   └── verify_author_location(): Verificación geográfica
```

### Flujo de Datos
1. **Inicialización**: Configuración de términos de búsqueda y países objetivo
2. **Autenticación**: Login automático en LinkedIn
3. **Búsqueda Iterativa**: Procesamiento por país y término
4. **Extracción**: Captura de datos de publicaciones
5. **Filtrado**: Aplicación de filtros de idioma y ubicación
6. **Verificación**: Validación geográfica mediante análisis de contenido y perfiles
7. **Consolidación**: Deduplicación y guardado en archivos consolidados

### Componentes Clave
- **Playwright**: Automatización de navegador para interacción con LinkedIn
- **Pandas**: Manipulación y consolidación de datos
- **Langid**: Detección automática de idiomas
- **Asyncio**: Programación asíncrona para eficiencia

## Metodología de Ciencia de Datos

### 1. Recolección de Datos
- **Web Scraping Estructurado**: Extracción sistemática de datos de redes sociales
- **Muestreo Estratégico**: Enfoque en términos específicos del dominio BIM
- **Cobertura Geográfica**: Muestreo representativo de mercados objetivo

### 2. Procesamiento de Texto
- **Detección de Idioma**: Clasificación automática usando modelos de lenguaje
- **Filtrado Basado en Contenido**: Análisis semántico para relevancia
- **Normalización**: Limpieza y estandarización de datos textuales

### 3. Clasificación Geográfica
- **Análisis Semántico**: Identificación de indicadores geográficos en texto
- **Verificación Cruzada**: Validación mediante múltiples fuentes (contenido + perfil)
- **Reglas de Decisión**: Algoritmos de clasificación basados en conocimiento experto

### 4. Gestión de Calidad de Datos
- **Deduplicación**: Técnicas de hashing para identificar registros únicos
- **Consistencia**: Mantenimiento de integridad referencial en concatenaciones
- **Validación**: Verificación de completitud y precisión de datos

### 5. Métricas y Monitoreo
- **KPI de Rendimiento**: Seguimiento de tasas de éxito, errores y cobertura
- **Logging Estructurado**: Registro detallado para análisis post-ejecución
- **Reportes Automáticos**: Generación de informes de ejecución

### 6. Escalabilidad y Mantenimiento
- **Arquitectura Modular**: Diseño orientado a objetos para facilidad de mantenimiento
- **Configuración Parametrizable**: Ajustes flexibles sin modificar código
- **Gestión de Errores**: Recuperación automática de fallos

## Contribuciones

### Cómo Contribuir
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### Áreas de Mejora Sugeridas
- **Machine Learning**: Implementar modelos de clasificación para mejor filtrado
- **NLP Avanzado**: Usar transformers para análisis más sofisticado de texto
- **Base de Datos**: Migrar a sistemas de base de datos para mayor escalabilidad
- **API Integration**: Crear endpoints REST para integración con otros sistemas
- **Visualización**: Dashboards para análisis exploratorio de datos

### Directrices de Código
- Seguir PEP 8 para estilo de código Python
- Documentar funciones con docstrings
- Mantener cobertura de pruebas
- Usar type hints para mejor legibilidad

## Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo LICENSE para más detalles.

## Contacto

Para preguntas o colaboraciones, contactar al equipo de desarrollo.

---

**Nota**: Este proyecto está diseñado exclusivamente para fines de investigación y análisis de datos. Asegurarse de cumplir con los términos de servicio de LinkedIn y las leyes de protección de datos aplicables en su jurisdicción.