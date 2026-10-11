#!/usr/bin/env bash
# Stage 00b：KPMP 图谱 v2（CZ CELLxGENE collection 9c9d04c4-8899-417f-bb6f-6107dcadf14f，开放获取）
# 人肾单核（7ff0197b…，1,388,643 核）与单细胞（91f31e05…，348,984 细胞）h5ad。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/data/raw/kpmp"; mkdir -p "$OUT" "$ROOT/results/00_download"
declare -A F=(
  [kpmp_v2_snRNA.h5ad]="https://datasets.cellxgene.cziscience.com/06b8d9c2-0bde-48bd-88d9-730993fc95e2.h5ad"
  [kpmp_v2_scRNA.h5ad]="https://datasets.cellxgene.cziscience.com/255ce81a-eb4e-48ff-9f18-e50455ccb24d.h5ad"
)
for f in "${!F[@]}"; do
  curl -fSL --retry 5 -C - -o "$OUT/$f" "${F[$f]}"
done
( cd "$OUT" && md5sum *.h5ad ) > "$ROOT/results/00_download/kpmp_md5.tsv"
