from typing import Optional, List, Dict, Any

from plwordnet_handler.api.data.lu import LexicalUnit
from plwordnet_handler.api.plwordnet_i import PlWordnetAPIBase
from plwordnet_handler.connectors.connector_i import PlWordnetConnectorInterface


class PlWordnetAPI(PlWordnetAPIBase):
    """
    Main API class for Polish Wordnet operations.
    """

    def __init__(self, connector: PlWordnetConnectorInterface):
        super().__init__(connector)

    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Get lexical units from the wordnet database.

        Args:
            limit: Optional limit for number of results

        Returns:
            List of lexical units or None if error
        """
        return self.connector.get_lexical_units(limit=limit)

    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get lexical relations from the wordnet database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical relations or None if an error occurred
        """
        return self.connector.get_lexical_relations(limit=limit)
