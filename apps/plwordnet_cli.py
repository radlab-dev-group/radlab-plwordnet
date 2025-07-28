import sys
import logging
import argparse

from plwordnet_handler.connectors.db.db_to_nx import DBToGraphMapper
from plwordnet_handler.structure.polishwordnet import PolishWordnet

# from plwordnet_handler.api.plwordnet import PlWordnetAPI
# from plwordnet_handler.connectors.db_connector import PlWordnetAPIMySQLDbConnector

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NX_OUT_DIR = "../resources/plwordnet"
DEFAULT_DB_CFG_PATH = "../resources/plwordnet-mysql-db.json"

EXAMPLE_USAGE = f"""
Example usage:

python plwordnet-cli \\
        --db-config {DEFAULT_DB_CFG_PATH} \\
        --extract-wikipedia-articles \\
        --convert-to-nx-graph \\
        --nx-graph-dir {DEFAULT_NX_OUT_DIR} \\
        --log-level {DEFAULT_LOG_LEVEL}
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../plwordnet_cli.log"),
    ],
)

logger = logging.getLogger(__name__)


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
        "--nx-graph-dir",
        dest="nx_graph_dir",
        type=str,
        default=DEFAULT_NX_OUT_DIR,
        help=f"Path to NX graph directory, if not given "
        f"{DEFAULT_NX_OUT_DIR} will be used",
    )

    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        required=False,
        help="Limit the number of results to check app is proper working.",
    )

    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Set the logging level",
    )

    parser.add_argument(
        "--show-progress-bar",
        dest="show_progress_bar",
        action="store_true",
        help="Show progress bar",
    )

    return parser


def dump_to_networkx_file(args) -> int:
    logger.info("Starting NetworkX graph generation")
    try:
        with PolishWordnet(
            db_config_path=args.db_config,
            extract_wiki_articles=args.extract_wikipedia_articles,
            connector=None,
            use_memory_cache=True,
            show_progress_bar=args.show_progress_bar,
        ) as pl_wn:
            g_mapper = DBToGraphMapper(polish_wordnet=pl_wn)
            logger.info("Converting to NetworkX MultiDiGraph...")
            g_mapper.prepare_all_graphs(limit=args.limit)
            g_mapper.store_to_dir(out_dir_path=args.nx_graph_dir)

        logger.info("NetworkX graph generation completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during NetworkX graph generation: {e}")
        return 1


def prepare_logger(args) -> None:
    logging.getLogger().setLevel(getattr(logging, args.log_level))


def main(argv=None):
    args = prepare_parser().parse_args(argv)
    prepare_logger(args=args)

    logger.info("Starting plwordnet-cli")
    logger.info(f"Arguments: {vars(args)}")

    if args.convert_to_nx:
        return dump_to_networkx_file(args=args)

    # api = PlWordnetAPI(
    #     connector=PlWordnetAPIMySQLDbConnector(args.db_config),
    #     extract_wiki_articles=args.extract_wikipedia_articles,
    # )
    # api.connect()
    # if not api.is_connected():
    #     raise Exception("Connection to MySQL failed")
    # print("Connected to MySQL")
    #
    # rel_types = api.get_relation_types(limit=10)
    # print("Relation types:")
    # for rt in rel_types:
    #     print("\t->", rt)
    #
    # api.disconnect()
    #
    # return 0
    logger.warning("Main functionality not implemented yet!")
    raise NotImplementedError("Not implemented yet!")


if __name__ == "__main__":
    sys.exit(main())
