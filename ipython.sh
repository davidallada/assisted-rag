#!/bin/bash
docker compose --profile ipython up -d
docker compose exec assisted-rag-ipython ipython