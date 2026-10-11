#!/usr/bin/env bash
# Stage 00f：下载 S-LDSC 参考文件（1000G Phase 3 EUR plink、等位基因频率、baselineLD v2.2、HapMap3 权重与 SNP 列表），
# 来源 Zenodo 10515792（ldsc 资源镜像），按记录公布的 MD5 核对后解压到 data/raw/ldsc/。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REF="$ROOT/data/raw/ldsc"; mkdir -p "$REF" "$ROOT/results/00_download"
BASE="https://zenodo.org/records/10515792/files"
cat > "$REF/.md5" <<'MD5'
b261e0caf06a003e7522938e01b3d349  1000G_Phase3_baselineLD_v2.2_ldscores.tgz
ac29686ffd5b6378789857a522ebca77  1000G_Phase3_frq.tgz
a7773ab485827b533cb300c76356d76b  1000G_Phase3_plinkfiles.tgz
a98ac0f089ee285177544a3e6e721ca3  1000G_Phase3_weights_hm3_no_MHC.tgz
65a34c68833eb4a764d0707b5505b508  hm3_no_MHC.list.txt
MD5
cd "$REF"
while read -r _ f; do
  [ -s "$f" ] || curl -fSL --retry 5 -C - -o "$f" "$BASE/$f?download=1"
done < .md5
md5sum -c .md5 | tee "$ROOT/results/00_download/ldsc_md5.tsv"
for t in *.tgz; do tar -xzf "$t" --skip-old-files; done
