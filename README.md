# PLWordNet Handler

Biblioteka do pracy z polskim słownikiem semantycznym PLWordNet. 
Umożliwia łatwy dostęp do danych słownikowych, zarządzanie połączeniami 
z bazą danych oraz przetwarzanie struktur językowych.

## Opis

PLWordNet Handler to kompleksowa biblioteka Python do pracy z danymi 
z polskiego słownika semantycznego PLWordNet. Biblioteka oferuje 
wysokopoziomowy interfejs API do pobierania jednostek leksykalnych, 
relacji między słowami oraz dodatkowych informacji językowych. 
Wspiera różne typy połączeń z bazą danych i zapewnia narzędzia do analizy 
komentarzy oraz przykładów użycia.

## Główne funkcjonalności

- **Dostęp do danych PLWordNet**: Pobieranie jednostek leksykalnych i relacji semantycznych
- **Elastyczne połączenia**: Obsługa różnych typów baz danych (MySQL, PostgreSQL)
- **Parsowanie komentarzy**: Analiza złożonych komentarzy z adnotacjami sentymentalnymi
- **Integracja z Wikipedią**: Wzbogacanie danych o opisy z Wikipedii
- **Wizualizacja**: Tworzenie grafów relacji semantycznych
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
plwordnet-mysql
```

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

## Współpraca
Zapraszamy do współpracy! Prosimy o:
1. Forkowanie repozytorium
2. Utworzenie feature branch
3. Implementację zmian z testami
4. Utworzenie Pull Request

## Wsparcie
W przypadku problemów lub pytań prosimy o utworzenie Issue w repozytorium GitHub.
