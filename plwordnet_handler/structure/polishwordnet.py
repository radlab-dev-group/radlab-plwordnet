from typing import Optional

from plwordnet_handler.api.plwordnet import PlWordnetAPI
from plwordnet_handler.connectors.db_connector import PlWordnetAPIMySQLDbConnector


class PolishWordnet:
    """
    Class representing a Słowosieć structure
    """

    def __init__(
        self,
        connector=None,
        db_config_path: Optional[str] = None,
        extract_wiki_articles: bool = False,
        use_memory_cache: bool = False,
    ):
        """
        Initialize PolishWordnet with PlWordnetAPI.

        Args:
            db_config_path: Optional path to a database configuration file.
            extract_wiki_articles: Whether to extract Wikipedia articles.
        """
        # Create connector with provided or default configuration
        if connector is None:
            if db_config_path:
                connector = PlWordnetAPIMySQLDbConnector(
                    db_config_path=db_config_path
                )
            else:
                raise Exception("Connector or database config must be provided.")

        self.api = PlWordnetAPI(
            connector=connector,
            extract_wiki_articles=extract_wiki_articles,
            use_memory_cache=use_memory_cache,
        )

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying PlWordnetAPI instance
        or use implementation from this class.

        Args:
            name: Name of the attribute to access

        Returns:
            The requested attribute from the PlWordnetAPI instance or self
        """
        if name in self.api.DELEGATED_METHODS:
            return getattr(self.api, name)

        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __enter__(self):
        """
        Context manager entry - establish connection.
        """
        if self.api.connect():
            return self
        else:
            raise ConnectionError("Failed to load connector")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - close connection.
        """
        self.api.disconnect()
