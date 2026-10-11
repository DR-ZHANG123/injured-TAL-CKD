#!/usr/bin/env bash
# Stage 00e：GSE195460（DKD 与对照肾皮质单核 RNA；cellranger v4 统一重计数的 filtered_feature_bc_matrix.h5，
# 对照 6 名、DKD 5 名）及系列矩阵（eGFR、肾小球硬化、IFTA）。只下载 RNA 计数，不下载 ATAC fragments。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/raw/GSE195460"; mkdir -p "$OUT" "$ROOT/results/00_download"
SER="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE195nnn/GSE195460"
SMP="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5837nnn"
get() { [ -s "$OUT/$2" ] || curl -fSL --retry 5 -C - -o "$OUT/$2" "$1/$2"; }
for s in Control1 Control2 Control3 Control4 Control5 DN1 DN2 DN3; do
  get "$SER/suppl" "GSE195460_${s}_filtered_feature_bc_matrix.h5"
done
get "$SMP/GSM5837792/suppl" GSM5837792_Control6_filtered_feature_bc_matrix.h5
get "$SMP/GSM5837797/suppl" GSM5837797_DN4_filtered_feature_bc_matrix.h5
get "$SMP/GSM5837799/suppl" GSM5837799_DN5_filtered_feature_bc_matrix.h5
get "$SER/matrix" GSE195460_series_matrix.txt.gz
( cd "$OUT" && md5sum *.h5 GSE195460_series_matrix.txt.gz ) > "$ROOT/results/00_download/gse195460_md5.tsv"
