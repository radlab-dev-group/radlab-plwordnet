from plwordnet_handler.api.data.wikipedia import (
    extract_wikipedia_description,
    WikipediaExtractor,
)

url = "https://pl.wikipedia.org/wiki/Diana_(1899)"

with WikipediaExtractor(max_sentences=10) as extractor:
    info = extractor.get_article_info(url)
    print(f"Title: {info['title']}")
    print(f"Description: {info['description']}")


# # Utility function
# description = extract_wikipedia_description(url, max_sentences=1)
# print(f"Short description: {description}")
#
# # Multiple URLs
# urls = [
#     "https://pl.wikipedia.org/wiki/Armia",
#     "https://en.wikipedia.org/wiki/Army"
# ]
# results = extractor.extract_multiple_descriptions(urls)
# for url, desc in results.items():
#     print(f"{url}: {desc}")
