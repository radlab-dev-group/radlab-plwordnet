#!/bin/bash

plwordnet-milvus \
  --milvus-config=../resources/milvus-config-pk.json \
  --embedder-config=../resources/embedder-config.json \
  --device="cuda:1" \
  --log-level=INFO \
  --prepare-database \
  --prepare-base-embeddings-lu \
  --prepare-base-mean-empty-embeddings-lu \
  --prepare-base-embeddings-synset \
  --nx-graph-dir=/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_full/nx/graphs \
  --relgat-mapping-directory=../resources/aligned-dataset-identifiers/ \
  --relgat-dataset-directory=../resources/aligned-dataset-identifiers/dataset \
  --export-relgat-dataset \
  --export-relgat-mapping
