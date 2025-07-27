import logging
import networkx

from abc import ABC
from typing import Optional, List, Dict, Any, Tuple

from plwordnet_handler.api.data.comment import CommentParser
from plwordnet_handler.api.data.lu import LexicalUnit, LexicalUnitMapper
from plwordnet_handler.api.data.lu_relations import (
    LexicalUnitRelation,
    LexicalUnitRelationMapper,
)

from plwordnet_handler.connectors.db.config import DbSQLConfig
from plwordnet_handler.connectors.db.mysql import MySQLDbConnection
from plwordnet_handler.connectors.connector_i import PlWordnetConnectorInterface


class _PlWordnetAPIMySQLDbConnectorBase(PlWordnetConnectorInterface, ABC):
    """
    Database connector implementation for plWordnet API using MySQL.
    """

    def __init__(self, db_config_path: str):
        """
        Initialize plWordnet database connector.

        Args:
            db_config_path: Path to JSON file with database configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If required configuration keys are missing
        """
        self.db_config_path = db_config_path
        self.config: Optional[DbSQLConfig] = None
        self.connection: Optional[MySQLDbConnection] = None
        self.logger = logging.getLogger(__name__)

        # Initialize configuration and connection
        self.__load_config()
        self.__initialize_connection()

    def connect(self) -> bool:
        """
        Establish connection to the database.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        if not self.connection:
            self.logger.error("Connection not initialized")
            return False

        success = self.connection.connect()
        if success:
            self.logger.info("Successfully connected to plWordnet database")
        else:
            self.logger.error("Failed to connect to plWordnet database")

        return success

    def disconnect(self) -> None:
        """
        Close database connection.
        """
        if self.connection:
            self.connection.disconnect()
            self.logger.info("Disconnected from plWordnet database")

    def is_connected(self) -> bool:
        """
        Check if the database connection is active.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connection and self.connection.is_connected()

    def _execute_select_query(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a database query.

        Args:
            query: SQL query to execute
            params: Optional parameters for the query

        Returns:
            List of dictionaries with query results, or None if error
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        try:
            results = self.connection.execute_select_query(
                query=query, params=params
            )
            self.logger.debug(
                f"Query executed successfully, returned "
                f"{len(results) if results else 0} results"
            )
            return results
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return None

    def __load_config(self) -> None:
        """
        Load MySQL configuration from a JSON file.
        """
        try:
            self.config = DbSQLConfig.from_json_file(config_path=self.db_config_path)
            self.logger.info(
                f"Configuration loaded successfully for database: {self.config.database}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def __initialize_connection(self) -> None:
        """
        Initialize database connection.
        """
        if not self.config:
            raise ValueError("Configuration not loaded")

        try:
            self.connection = MySQLDbConnection(**self.config.to_dict())
            self.logger.info("Database connection initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection: {e}")
            raise


class _PlWordnetAPIMySQLDbConnectorQueries(_PlWordnetAPIMySQLDbConnectorBase, ABC):
    """
    Abstract base class containing SQL queries and helper methods
    for PlWordnet MySQL database operations.

    Contains predefined SQL queries and methods for executing queries
    with optional result limiting functionality.
    """

    LIMIT_QUERY = "LIMIT"
    GET_ALL_LU = "LEXICAL_UNITS"
    GET_ALL_LU_RELS = "LEXICAL_UNITS_RELATIONS"
    Q = {
        GET_ALL_LU: "SELECT * FROM lexicalunit",
        GET_ALL_LU_RELS: "SELECT * FROM lexicalrelation",
    }

    def _limit_query(self, query: str, limit: int):
        """
        Adds a LIMIT clause to the SQL query.

        Args:
            query: SQL query to modify
            limit: Maximum number of results (must be > 0)

        Returns:
            str: SQL query with added LIMIT clause

        Raises:
            AssertionError: If limit <= 0
        """
        assert limit > 0
        return f"{query} {self.LIMIT_QUERY} {limit}"

    def _execute_query_with_limit_opt(
        self, query: str, params: Optional[Tuple] = None, limit: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a SELECT query with optional result limiting.

        Args:
            query: SQL query to execute
            params: Optional query parameters
            limit: Optional limit for number of results

        Returns:
            List of dictionaries with query results or None on error
        """
        if limit:
            query = self._limit_query(query=query, limit=limit)
        return self._execute_select_query(query=query, params=params)


class PlWordnetAPIMySQLDbConnector(_PlWordnetAPIMySQLDbConnectorQueries):
    """
    Main MySQL database connector class for PlWordnet API.

    Implements the PlWordnetConnectorInterface and provides
    concrete methods for retrieving lexical units and lexical
    relations from a MySQL database.
    """

    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Retrieves lexical units from the database.

        Args:
            limit: Optional limit for number of results

        Returns:
            List of LexicalUnit objects or None if no data available
        """
        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_LU],
            limit=limit,
        )
        if not data_list:
            return None
        return LexicalUnitMapper.map_from_dict_list(data_list=data_list)

    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Retrieves lexical relations from the database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of LexicalUnitRelation objects or None on error
        """
        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_LU_RELS],
            limit=limit,
        )
        if not data_list:
            return None
        return LexicalUnitRelationMapper.map_from_dict_list(data_list=data_list)

    def to_nx_multi_di_graph(
        self, extract_wiki_articles: bool
    ) -> networkx.MultiDiGraph or None:
        """
        Converts database data to NetworkX MultiDiGraph format.

        Args:
            extract_wiki_articles: Whether to extract wiki articles

        Returns:
            NetworkX MultiDiGraph or None (currently returns None - in development)

        Raises:
            ValueError: If not connected to the database
        """
        if not self.is_connected():
            raise ValueError("Not connected to database")

        all_lu = self.get_lexical_units()
        all_lu_rels = self.get_lexical_relations()

        print("Number of lexical units:", len(all_lu))
        print("Number of lexical units relations:", len(all_lu_rels))
        #
        # parser = CommentParser()
        # for lu in self.get_lexical_units(limit=1000):
        #     print("LU:", lu)
        #     print(" EMO ANNOTATIONS")
        #     print("   -> emotions")
        #     print("\t", parser.get_all_emotions(lu.comment))
        #     print("   -> categories")
        #     print("\t", parser.get_all_categories(lu.comment))
        #
        #     print(" ADDITIONAL INFO")
        #     print("   -> ext_url: ", lu.comment.external_url_description)
        #     print("   -> base_domain: ", lu.comment.base_domain)
        #     print("   -> definition: ", lu.comment.definition)
        #     print("   -> usage_examples: ")
        #     for e in lu.comment.usage_examples:
        #         print("  \t - ", e)
        #     print("-" * 50)
        return None
