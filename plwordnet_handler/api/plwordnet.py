import networkx
from typing import Optional, List

from plwordnet_handler.structure.elems.lu import LexicalUnit
from plwordnet_handler.structure.elems.rel_type import RelationType
from plwordnet_handler.structure.elems.lu_relations import LexicalUnitRelation

from plwordnet_handler.api.plwordnet_i import PlWordnetAPIBase
from plwordnet_handler.connectors.connector_i import PlWordnetConnectorInterface


class PlWordnetAPI(PlWordnetAPIBase):
    """
    Main API class for Polish Wordnet operations.
    """

    def __init__(
        self,
        connector: PlWordnetConnectorInterface,
        extract_wiki_articles: bool = False,
    ):
        """
        Args:
             connector: connector interface for plWordnet (PlWordnetConnectorInterface)
             extract_wiki_articles: whether to extract wiki articles
        """
        super().__init__(connector)
        self.extract_wiki_articles = extract_wiki_articles

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
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Get lexical relations from the wordnet database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical relations or None if an error occurred
        """
        return self.connector.get_lexical_relations(limit=limit)

    def get_relation_types(
        self, limit: Optional[int] = None
    ) -> Optional[List[RelationType]]:
        """
        Get relation types from the wordnet database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of relation types or None if an error occurred
        """
        return self.connector.get_relation_types(limit=limit)

    def to_nx_multi_di_graph(
        self, extract_wiki_articles: bool, limit: Optional[int] = None
    ) -> networkx.MultiDiGraph or None:
        return self.connector.to_nx_multi_di_graph(
            extract_wiki_articles=extract_wiki_articles, limit=limit
        )
