#!/usr/bin/env bash
# Stage 00d：GSE271188（肾小球疾病活检单核：DKD、IgAN、MN、MCD、对照）与 GSE286184（同一超系列 GSE286186 的 Xenium）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for g in GSE271188 GSE286184; do
  OUT="$ROOT/data/raw/$g"; mkdir -p "$OUT"
  stub="${g:0:6}nnn"
  [ -s "$OUT/${g}_RAW.tar" ] || curl -fSL --retry 5 -C - -o "$OUT/${g}_RAW.tar" "https://ftp.ncbi.nlm.nih.gov/geo/series/$stub/$g/suppl/${g}_RAW.tar"
  curl -fsSL -o "$OUT/${g}_series_matrix.txt.gz" "https://ftp.ncbi.nlm.nih.gov/geo/series/$stub/$g/matrix/${g}_series_matrix.txt.gz" || true
  ( cd "$OUT" && tar -xf "${g}_RAW.tar" )
done
( cd "$ROOT/data/raw" && md5sum GSE271188/GSE271188_RAW.tar GSE286184/GSE286184_RAW.tar ) > "$ROOT/results/00_download/gse271188_md5.tsv"
