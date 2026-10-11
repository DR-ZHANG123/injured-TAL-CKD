#!/usr/bin/env bash
# Stage 00c：GSE211785（Susztak 队列：对照 / 高血压肾病 / DKD 的 sc+snRNA；不含 KPMP 样本的 PreSCVI 原始计数）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/raw/gse211785"; mkdir -p "$OUT" "$ROOT/results/00_download"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE211nnn/GSE211785/suppl"
for f in GSE211785_Susztak_SC_SN_ATAC_merged_PreSCVI_final.h5ad.gz GSE211785_scRNA-seq_snRNA-seq_snATAC-seq_metadata.txt.gz; do
  [ -s "$OUT/$f" ] || curl -fSL --retry 5 -C - -o "$OUT/$f" "$BASE/$f"
done
[ -s "$OUT/susztak_preSCVI.h5ad" ] || gunzip -c "$OUT/GSE211785_Susztak_SC_SN_ATAC_merged_PreSCVI_final.h5ad.gz" > "$OUT/susztak_preSCVI.h5ad"
( cd "$OUT" && md5sum *.gz ) > "$ROOT/results/00_download/gse211785_md5.tsv"
