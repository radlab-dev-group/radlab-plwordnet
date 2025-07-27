from abc import ABC, abstractmethod
from typing import Optional, List

from plwordnet_handler.api.data.lu import LexicalUnit
from plwordnet_handler.api.data.rel_type import RelationType
from plwordnet_handler.api.data.lu_relations import LexicalUnitRelation


class PlWordnetConnectorInterface(ABC):
    """
    Abstract interface for plWordnet database connectors.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection using connector.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection from connector.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if a connection is active.

        Returns:
            bool: True if connected, False otherwise
        """
        pass

    @abstractmethod
    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Get lexical units

        Args:
            limit: Optional limit for number of results

        Returns:
            List of lexical units or None if error
        """
        pass

    @abstractmethod
    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Get lexical relations

        Args:
            limit: Optional limit for number of results

        Returns:
            List of lexical relations or None if error
        """
        pass

    @abstractmethod
    def get_relation_types(
        self, limit: Optional[int] = None
    ) -> Optional[List[RelationType]]:
        """
        Get types of relations

        Args:
            limit: Optional limit for a number of results

        Returns:
            List of relation types or None if error occurred
        """
        pass

    def __enter__(self):
        """
        Context manager entry - establish connection.
        """
        if self.connect():
            return self
        else:
            raise ConnectionError("Failed to establish database connection")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - close connection.
        """
        self.disconnect()
