# PLWordNet Handler

Biblioteka do pracy z polskim słownikiem semantycznym PLWordNet. 
Umożliwia łatwy dostęp do danych słownikowych, zarządzanie połączeniami 
z bazą danych oraz przetwarzanie struktur Słowosieci.

**Aktualna wersja `README.md` jest w trakcie tworzenia, niektóre opisy 
mogą nie być aktualne**

## Opis

PLWordNet Handler to kompleksowa biblioteka Python do pracy z danymi 
z polskiej Słowosieci. Biblioteka oferuje wysokopoziomowy interfejs 
API do pobierania jednostek leksykalnych, relacji między nimi
oraz dodatkowych informacji ze Słowosieci. Wspiera połączenie 
z bazą danych i zapewnia funkcjonalności do analizy komentarzy 
oraz przykładów użycia jednostek leksykalnych.

## Główne funkcjonalności

- **Dostęp do danych PLWordNet**: Pobieranie jednostek leksykalnych i relacji semantycznych
- **Elastyczne połączenia**: Połączenie do MySQL, konwersja do grafu `networkx`
- **Parsowanie komentarzy**: Analiza złożonych komentarzy z anotacjami m.in. sentyment
- **Integracja z Wikipedią**: Wzbogacanie danych o opisy z Wikipedii
- **Aplikacje konsolowe**: Gotowe narzędzia do pracy z danymi

## Główne moduły

### `plwordnet_handler.api`
Główny interfejs API do pracy z PLWordNet. Zawiera abstrakcyjną klasę 
bazową `PlWordnetAPIBase` oraz implementacje dla różnych typów baz danych.

### `plwordnet_handler.connectors`
Moduły odpowiedzialne za połączenia z bazami danych:
- **`connector_i.py`** - Interfejs dla wszystkich typów połączeń
- **`db_connector.py`** - Implementacja połączeń z bazami danych MySQL
- **`mysql.py`** - Niskopoziomowe połączenia MySQL

### `plwordnet_handler.api.data`
Struktury danych reprezentujące elementy PLWordNet:
- **`lu.py`** - Klasa `LexicalUnit` reprezentująca jednostki leksykalne
- **`comment.py`** - Parser komentarzy z adnotacjami sentymentalnymi i przykładami użycia

### `plwordnet_handler.config`
Zarządzanie konfiguracją aplikacji:
- **`config.py`** - Klasa `DbSQLConfig` do konfiguracji połączeń z bazami danych

### `plwordnet_handler.external`
Integracje z zewnętrznymi źródłami danych:
- **`wikipedia.py`** - Klasa `WikipediaExtractor` do pobierania opisów z Wikipedii

### `apps`
Gotowe aplikacje konsolowe:
- **`plwordnet-mysql.py`** - Aplikacja do pracy z bazą MySQL PLWordNet


## Instalacja

### Wymagania
- Python 3.6 lub nowszy
- Dostęp do bazy danych PLWordNet (MySQL/PostgreSQL)

### Instalacja z repozytorium

```bash
# Klonowanie repozytorium
git clone https://github.com/yourusername/plwordnet-handler.git
cd plwordnet-handler

# Instalacja w trybie deweloperskim
pip install -e .
```

### Instalacja standardowa
``` bash
pip install .
```

### Instalacja bezpośrednio z GitHub
``` bash
pip install git+https://github.com/radlab-dev-group/radlab-plwordnet.git
```


## Szybki start

### Konfiguracja bazy danych
Utwórz plik konfiguracyjny `config.json`:
``` json
{
    "host": "localhost",
    "port": 3306,
    "user": "username",
    "password": "password",
    "database": "plwordnet"
}
```

### Podstawowe użycie
``` python
from plwordnet_handler.connectors.db_connector import PlWordnetAPIMySQLDbConnector
from plwordnet_handler.api.plwordnet_i import PlWordnetAPIBase

# Utworzenie połączenia
connector = PlWordnetAPIMySQLDbConnector("config.json")

# Użycie context managera
with connector as conn:
    # Pobieranie jednostek leksykalnych
    lexical_units = conn.get_lexical_units(limit=10)

    # Pobieranie relacji leksykalnych
    relations = conn.get_lexical_relations(limit=5)

    # Przetwarzanie wyników
    for unit in lexical_units:
        print(f"Lemma: {unit.lemma}, POS: {unit.pos}")
```

### Parsowanie komentarzy
``` python
from plwordnet_handler.api.data.comment import parse_plwordnet_comment

comment_text = "##D: definicja słowa ##A1: {radość; pozytywne} - s [przykład użycia]"
parsed = parse_plwordnet_comment(comment_text)

print(f"Definicja: {parsed.definition}")
print(f"Adnotacje: {len(parsed.sentiment_annotations)}")
```

### Aplikacje konsolowe
Po instalacji dostępne są aplikacje konsolowe:
``` bash
# Uruchomienie aplikacji MySQL
plwordnet-cli
```
Przykładowe wywołanie do konwersji bazy MySQL do grafów ze ściąganiem linków Wikipedii:
``` bash
plwordnet-cli \
    --convert-to-nx-graph \
    --nx-graph-dir resources/plwordnet \
    --extract-wikipedia-articles \
    --db-config resources/plwordnet-mysql-db.json 
```
**Uwaga**, proces pobierania artykułów może być bardzo długi!

## Dokumentacja API
Szczegółowa dokumentacja API dostępna jest w kodzie źródłowym. 
Każda klasa i metoda zawiera docstring z opisem parametrów i wartości zwracanych.

## Struktura projektu
``` 
plwordnet-handler/
├── plwordnet_handler/          # Główny pakiet biblioteki
│   ├── api/                    # Interfejsy API
│   ├── connectors/             # Połączenia z bazami danych
│   ├── config/                 # Konfiguracja
│   └── external/               # Integracje zewnętrzne
├── apps/                       # Aplikacje konsolowe
├── resources/                  # Zasoby i pliki konfiguracyjne
├── requirements.txt            # Zależności
├── setup.py                   # Konfiguracja instalacji
└── README.md                  # Ten plik
```

## Licencja
Apache 2.0 License - szczegóły w pliku LICENSE.
