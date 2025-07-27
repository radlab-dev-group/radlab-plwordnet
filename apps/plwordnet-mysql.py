DEFAULT_DB_CFG_PATH = "resources/plwordnet-mysql-db.json"
DEFAULT_NX_OUT_FILE = "resources/plwordnet-nx-multidigraph.pickle"

EXAMPLE_USAGE = f"""
Example usage:

python plwordnet-mysql.py \\
        --db-config {DEFAULT_DB_CFG_PATH} \\
        --extract-wikipedia-articles \\
        --convert-to-nx-graph \\
        --nx-graph-file {DEFAULT_NX_OUT_FILE}
"""


import sys
import argparse

from plwordnet_handler.api.plwordnet import PlWordnetAPI
from plwordnet_handler.connectors.db_connector import PlWordnetAPIMySQLDbConnector


def prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Polish Wordnet MySQL handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE,
    )

    parser.add_argument(
        "--db-config",
        dest="db_config",
        type=str,
        default=DEFAULT_DB_CFG_PATH,
        help="Path to JSON file with database configuration",
    )

    parser.add_argument(
        "--extract-wikipedia-articles",
        dest="extract_wikipedia_articles",
        action="store_true",
        help="Extract Wikipedia articles as additional LU description",
    )

    parser.add_argument(
        "--convert-to-nx-graph",
        dest="convert_to_nx",
        action="store_true",
        help="Convert to NX graph as nx.MultiDiGraph",
    )
    parser.add_argument(
        "--nx-graph-file",
        dest="nx_graph_file",
        type=str,
        default=DEFAULT_NX_OUT_FILE,
        help=f"Path to NX graph file, if not given "
        f"{DEFAULT_NX_OUT_FILE} will be used",
    )

    return parser


def dump_to_networkx_file(args, extract_wiki_articles: bool) -> int:
    mysql_connector = PlWordnetAPIMySQLDbConnector(args.db_config)
    mysql_connector.connect()

    # TODO: extract_wiki_articles
    nx_graph = mysql_connector.to_nx_multi_di_graph()

    # TODO: Store to args.nx_graph_file
    mysql_connector.disconnect()

    return 1


def main(argv=None):
    args = prepare_parser().parse_args(argv)
    if args.convert_to_nx:
        return dump_to_networkx_file(
            args=args, extract_wiki_articles=args.extract_wikipedia_articles
        )

    # TODO: args.extract_wikipedia_articles
    api = PlWordnetAPI(connector=PlWordnetAPIMySQLDbConnector(args.db_config))
    # api.connect()
    # if not api.is_connected():
    #     raise Exception("Connection to MySQL failed")
    # print("Connected to MySQL")
    #
    # l_us = api.to_nx_multi_di_graph(limit=10)
    # print("Lexical units:")
    # for lu in l_us:
    #     print("\t->", lu)
    #
    # l_u_rels = api.get_lexical_relations(limit=10)
    # print("Lexical relations:")
    # for rel in l_u_rels:
    #     print("\t->", rel)
    #
    # api.disconnect()

    return 0


if __name__ == "__main__":
    main()
