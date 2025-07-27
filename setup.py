import os

from setuptools import setup, find_packages


def _read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []


def _read_long_description():
    readme_files = ["README.md", "README.rst", "README.txt"]
    for readme_file in readme_files:
        readme_path = os.path.join(os.path.dirname(__file__), readme_file)
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
    return "PLWordNet Handler - narzędzie do pracy z polskim słownikiem semantycznym"


setup(
    name="plwordnet-handler",
    version="1.0.0",
    description="Narzędzie do pracy ze Słowosiecią",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="RadLab Team",
    author_email="pawel@radlab.dev",
    url="https://github.com/radlab-dev-group/radlab-plwordnet",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.rst"],
        "resources": ["*"],
    },
    install_requires=_read_requirements(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "plwordnet-mysql=apps.plwordnet-mysql:main",
        ],
    },
    keywords="nlp, wordnet, polish, linguistics, semantic",
    zip_safe=False,
)
