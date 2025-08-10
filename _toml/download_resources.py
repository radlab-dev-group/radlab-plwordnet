import os
import sys
import tarfile
import argparse
import urllib.request

from pathlib import Path
from dataclasses import dataclass


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class _SetupConfiguration:
    REPO_URL = "https://github.com/radlab-dev-group/radlab-plwordnet"

    @dataclass
    class SetupConfig:
        handler_module_name: str = "plwordnet_handler"
        radlab_resources: str = (
            "https://resources.radlab.dev/persistant/plwn_handler"
        )
        remote_base_resources_dir: str = "resources"
        remote_base_graphs_dir: str = "plwordnet/graphs"
        remote_test_graph_name: str = "slowosiec_test.gz"
        remote_full_graph_name: str = "slowosiec_full.gz"

        @property
        def full_plwn_graph_url(self):
            return (
                f"{self.radlab_resources}/"
                f"{self.remote_base_resources_dir}/"
                f"{self.remote_base_graphs_dir}/"
                f"{self.remote_full_graph_name}"
            )

        @property
        def test_plwn_graph_url(self):
            return (
                f"{self.radlab_resources}/"
                f"{self.remote_base_resources_dir}/"
                f"{self.remote_base_graphs_dir}/"
                f"{self.remote_test_graph_name}"
            )


class _GraphDownloadMixin:
    """
    Mixin for downloading graphs during installation.
    """

    ENV_DOWN_TEST_PLWN = "PLWORDNET_DOWNLOAD_TEST"
    ENV_DOWN_FULL_PLWN = "PLWORDNET_DOWNLOAD_FULL"

    def download_graphs_if_requested(self):
        """
        Download graphs if environment variables are set.
        """
        config = _SetupConfiguration.SetupConfig()

        download_test = os.environ.get(self.ENV_DOWN_TEST_PLWN, "").lower() in (
            "1",
            "true",
            "yes",
        )
        download_full = os.environ.get(self.ENV_DOWN_FULL_PLWN, "").lower() in (
            "1",
            "true",
            "yes",
        )

        if not (download_test or download_full):
            print("No graph download requested.")
            return

        # Determine target directory
        if hasattr(self, "install_lib") and self.install_lib:
            target_dir = self.install_lib
        else:
            # Fallback for develop mode
            import site

            target_dir = site.getsitepackages()[0]

        graphs_dir = os.path.join(
            target_dir, "plwordnet_handler", "resources", "graphs"
        )
        os.makedirs(graphs_dir, exist_ok=True)
        print(f"Target graphs directory: {graphs_dir}")

        if download_test:
            print("Downloading test graph data...")
            self._download_and_extract_graph(
                config.test_plwn_graph_url,
                graphs_dir,
                "plwn_test_graph",
            )

        if download_full:
            print("Downloading full graph data...")
            self._download_and_extract_graph(
                config.full_plwn_graph_url,
                graphs_dir,
                "plwn_full_graph",
            )

    @staticmethod
    def _download_and_extract_graph(url, destination_path, graph_type):
        """
        Download and extract a gzipped graph file.
        """
        archive_path = graph_type + ".gz"
        try:
            print(f"Downloading {graph_type} from {url}...")
            urllib.request.urlretrieve(url, archive_path)

            print(f"Extracting {graph_type}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=destination_path)

            if os.path.exists(archive_path):
                os.remove(archive_path)

            print(f"Successfully installed {graph_type} to {destination_path}")
        except Exception as e:
            print(f"Error downloading {graph_type}: {e}")
            for path in [archive_path, destination_path]:
                if os.path.exists(path):
                    os.remove(path)


def main():
    parser = argparse.ArgumentParser(description="Pobierz zasoby plWordNet")
    parser.add_argument(
        "--test", action="store_true", help="Pobierz testową wersję grafu"
    )
    parser.add_argument(
        "--full", action="store_true", help="Pobierz pełną wersję grafu"
    )

    args = parser.parse_args()

    # Jeśli nie podano argumentów, sprawdź zmienne środowiskowe
    if not args.test and not args.full:
        if os.environ.get("PLWORDNET_DOWNLOAD_TEST"):
            args.test = True
        if os.environ.get("PLWORDNET_DOWNLOAD_FULL"):
            args.full = True

    if not args.test and not args.full:
        print(
            "Podaj --test lub --full albo ustaw PLWORDNET_DOWNLOAD_TEST/PLWORDNET_DOWNLOAD_FULL"
        )
        return 1

    os.environ[_GraphDownloadMixin.ENV_DOWN_TEST_PLWN] = str(args.test)
    os.environ[_GraphDownloadMixin.ENV_DOWN_FULL_PLWN] = str(args.full)

    downloader = _GraphDownloadMixin()
    downloader.download_graphs_if_requested()

    print("Pobieranie zakończone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
