import logging
import os
import pickle
import networkx as nx
from typing import Optional, Dict, List

from plwordnet_handler.structure.elems.lu import LexicalUnit
from plwordnet_handler.structure.polishwordnet import PolishWordnet


class DBToGraphMapper:
    G_UAS = "units_and_synsets"
    G_LU = "lexical_units"
    G_SYN = "synsets"
    GRAPH_TYPES = [G_UAS, G_LU, G_SYN]

    GRAPH_DIR = "nx/graphs"

    """
    Class for mapping Polish Wordnet database elements to NetworkX MultiDiGraph.
    """

    def __init__(self, polish_wordnet: PolishWordnet):
        """
        Initialize the mapper with a PolishWordnet instance.

        Args:
            polish_wordnet: PolishWordnet instance to use for data retrieval
        """
        self.polish_wordnet = polish_wordnet
        self.logger = logging.getLogger(__name__)

        self._graphs = {}

    def get_synset_graph(self, limit: Optional[int] = None) -> nx.MultiDiGraph:
        """
        Create a graph where synsets are nodes and synset relations are edges.

        Args:
            limit: Optional limit for the number of results

        Returns:
            NetworkX MultiDiGraph with synsets as nodes and relations as edges

        Raises:
            ValueError: If synsets or synset relations data is None
        """
        self.logger.info("Creating synset graph")

        synsets = self.polish_wordnet.get_synsets(limit=limit)
        if synsets is None:
            raise ValueError("Synsets data is None")

        synset_relations = self.polish_wordnet.get_synset_relations(limit=limit)
        if synset_relations is None:
            raise ValueError("Synset relations data is None")

        relation_types = self.polish_wordnet.get_relation_types(limit=limit)
        if relation_types is None:
            raise ValueError("Relation types data is None")

        # Create relation type mapping
        rel_type_map = {rt.ID: rt for rt in relation_types}

        # Create graph
        graph = nx.MultiDiGraph()

        # Add synset nodes
        for synset in synsets:
            graph.add_node(synset.ID, data=synset.to_dict())

        # Add synset relation edges
        for relation in synset_relations:
            rel_type = rel_type_map.get(relation.REL_ID)
            edge_data = {
                "relation_id": relation.REL_ID,
                "relation_type": rel_type.name if rel_type else None,
                "data": relation.to_dict(),
            }
            graph.add_edge(relation.PARENT_ID, relation.CHILD_ID, **edge_data)

        self.logger.info(
            f"Created synset graph with {graph.number_of_nodes()} "
            f"nodes and {graph.number_of_edges()} edges"
        )

        self._graphs[self.G_SYN] = graph
        return graph

    def get_lexical_unit_graph(self, limit: Optional[int] = None) -> nx.MultiDiGraph:
        """
        Create a graph where lexical units are nodes
        and lexical unit relations are edges.

        Args:
            limit: Optional limit for the number of results

        Returns:
            NetworkX MultiDiGraph with lexical units as nodes and relations as edges

        Raises:
            ValueError: If lexical units or lexical unit relations data is None
        """
        self.logger.info("Creating lexical unit graph")

        # Get lexical units
        lexical_units = self.polish_wordnet.get_lexical_units(limit=limit)
        if lexical_units is None:
            raise ValueError("Lexical units data is None")

        # Get lexical unit relations
        lu_relations = self.polish_wordnet.get_lexical_relations(limit=limit)
        if lu_relations is None:
            raise ValueError("Lexical unit relations data is None")

        # Get relation types for mapping
        relation_types = self.polish_wordnet.get_relation_types(limit=limit)
        if relation_types is None:
            raise ValueError("Relation types data is None")

        # Create relation type mapping
        rel_type_map = {rt.ID: rt for rt in relation_types}

        # Create graph
        graph = nx.MultiDiGraph()

        # Add lexical unit nodes
        for lu in lexical_units:
            graph.add_node(lu.ID, data=lu.to_dict())

        # Add lexical unit relation edges
        for relation in lu_relations:
            rel_type = rel_type_map.get(relation.REL_ID)
            edge_data = {
                "relation_id": relation.REL_ID,
                "relation_type": rel_type.name if rel_type else None,
                "data": relation.to_dict(),
            }
            graph.add_edge(relation.PARENT_ID, relation.CHILD_ID, **edge_data)

        self.logger.info(
            f"Created lexical unit graph with {graph.number_of_nodes()} "
            f"nodes and {graph.number_of_edges()} edges"
        )

        self._graphs[self.G_LU] = graph
        return graph

    def get_synset_with_units_graph(
        self, limit: Optional[int] = None
    ) -> nx.MultiDiGraph:
        """
        Create a graph where synsets are nodes containing lists of lexical units,
        and synset relations are edges.

        Args:
            limit: Optional limit for the number of results

        Returns:
            NetworkX MultiDiGraph with synsets as nodes
            (containing unit lists) and relations as edges

        Raises:
            ValueError: If required data is None
        """
        self.logger.info("Creating synset with units graph")

        # Get all required data
        synsets = self.polish_wordnet.get_synsets(limit=limit)
        if synsets is None:
            raise ValueError("Synsets data is None")

        synset_relations = self.polish_wordnet.get_synset_relations(limit=limit)
        if synset_relations is None:
            raise ValueError("Synset relations data is None")

        lexical_units = self.polish_wordnet.get_lexical_units(limit=limit)
        if lexical_units is None:
            raise ValueError("Lexical units data is None")

        units_and_synsets = self.polish_wordnet.get_units_and_synsets(limit=limit)
        if units_and_synsets is None:
            raise ValueError("Units and synsets data is None")

        relation_types = self.polish_wordnet.get_relation_types(limit=limit)
        if relation_types is None:
            raise ValueError("Relation types data is None")

        # Create mappings
        rel_type_map = {rt.ID: rt for rt in relation_types}
        lu_map = {lu.ID: lu for lu in lexical_units}

        # Create synset to units mapping
        synset_units_map: Dict[int, List[LexicalUnit]] = {}
        for unit_synset in units_and_synsets:
            synset_id = unit_synset.SYN_ID
            unit_id = unit_synset.LEX_ID

            if synset_id not in synset_units_map:
                synset_units_map[synset_id] = []

            if unit_id in lu_map:
                synset_units_map[synset_id].append(lu_map[unit_id])

        # Create graph
        graph = nx.MultiDiGraph()

        # Add synset nodes with unit lists
        for synset in synsets:
            synset_data = synset.to_dict()
            units_in_synset = synset_units_map.get(synset.ID, [])
            synset_data["units"] = [unit.to_dict() for unit in units_in_synset]
            graph.add_node(synset.ID, data=synset_data)

        # Add synset relation edges
        for relation in synset_relations:
            rel_type = rel_type_map.get(relation.REL_ID)
            edge_data = {
                "relation_id": relation.REL_ID,
                "relation_type": rel_type.name if rel_type else None,
                "data": relation.to_dict(),
            }
            graph.add_edge(relation.PARENT_ID, relation.CHILD_ID, **edge_data)

        self.logger.info(
            f"Created synset with units graph with {graph.number_of_nodes()} "
            f"nodes and {graph.number_of_edges()} edges"
        )
        self._graphs[self.G_UAS] = graph
        return graph

    def prepare_all_graphs(self, limit: Optional[int] = None):
        """
        Prepares all available NetworkX graph types from Polish Wordnet data.

        Checks which graph types haven't been created yet and generates them
        based on database data. Supports three graph types:
        - Synset graph (G_SYN): nodes are synsets, edges are synset relations
        - Lexical unit graph (G_LU): nodes are lexical units,
          edges are lexical relations
        - Synsets with lexical units graph (G_UAS): nodes are synsets containing
        lists of lexical units

        Args:
            limit (Optional[int], optional): Optional limit for the number of records
                fetched from a database for each data type. Useful for testing or
                working with data samples. If None, all available data is fetched.
                Defaults to None.

        Raises:
            ValueError: When an unknown graph type is encountered.
        """

        for g_type in self.GRAPH_TYPES:
            if g_type not in self._graphs:
                if g_type == self.G_SYN:
                    self.get_synset_graph(limit=limit)
                elif g_type == self.G_LU:
                    self.get_lexical_unit_graph(limit=limit)
                elif g_type == self.G_UAS:
                    self.get_synset_with_units_graph(limit=limit)
                else:
                    raise ValueError(f"Unknown graph type: {g_type}")

    def store_to_dir(self, out_dir_path: str):
        """
        Store all graphs from _graphs to directory as separate pickle files.

        Creates a subdirectory GRAPH_DIR within out_dir_path and saves each graph
        from _graphs as a separate pickle file.

        Args:
            out_dir_path: Path to the output directory
        """
        graphs_dir = os.path.join(out_dir_path, self.GRAPH_DIR)
        os.makedirs(graphs_dir, exist_ok=True)

        self.logger.info(f"Storing graphs to directory: {graphs_dir}")
        for graph_type, graph in self._graphs.items():
            file_path = os.path.join(graphs_dir, f"{graph_type}.pickle")
            with open(file_path, "wb") as f:
                pickle.dump(graph, f)
            self.logger.info(f"Saved graph '{graph_type}' to {file_path}")
        self.logger.info(f"Successfully stored {len(self._graphs)} graphs")
