#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

mkdir -p datasets
curl -L --fail -o datasets/complete.zip "https://zenodo.org/records/23104607/files/complete.zip?download=1"
unzip -q datasets/complete.zip -d datasets
rm datasets/complete.zip