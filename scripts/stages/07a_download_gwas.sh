#!/usr/bin/env bash
# Stage 07a：下载 GWAS 汇总统计（CKDGen eGFR / UACR / 糖尿病分层）、MAGMA 二进制与参考文件、HGNC 基因表。
# URL 全部来自 config/run.yaml 的 genetics 段；结果 MD5 写入 results/00_download/gwas_md5.tsv。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
GWAS="$ROOT/data/raw/gwas"; TOOLS="$ROOT/data/tools/magma"; ANNOT="$ROOT/data/raw/annot"
mkdir -p "$GWAS" "$TOOLS" "$ANNOT" results/00_download

# 1) 从 config 读出 trait -> file/url（同一文件只下一次）
python - <<'PY' > "$GWAS/.urls.tsv"
import sys; sys.path.insert(0, "scripts/lib")
from repro import load_config
g = load_config()["genetics"]
seen = set()
for t, v in g["traits"].items():
    if v["file"] not in seen:
        seen.add(v["file"]); print(v["file"], v["url"], sep="\t")
m = g["magma"]
for name, key in [("magma_v1.10_static.zip", "url_binary"), ("NCBI37.3.zip", "url_geneloc"), ("g1000_eur.zip", "url_ref")]:
    print("TOOLS/" + name, m[key], sep="\t")
print("ANNOT/hgnc_complete_set.txt", g["eqtl"]["hgnc_url"], sep="\t")
PY

while IFS=$'\t' read -r f url; do
  case "$f" in
    TOOLS/*) out="$TOOLS/${f#TOOLS/}";;
    ANNOT/*) out="$ANNOT/${f#ANNOT/}";;
    *)       out="$GWAS/$f";;
  esac
  [ -s "$out" ] || curl -fSL --retry 5 -C - -o "$out" "$url"
done < "$GWAS/.urls.tsv"

# 2) 解压 MAGMA
( cd "$TOOLS" && for z in magma_v1.10_static.zip NCBI37.3.zip g1000_eur.zip; do unzip -n -q "$z"; done && chmod +x magma )

# 3) 与 CKDGen 发布的 MD5 核对（Teumer2019 提供 MD5SUM）
curl -fsSL -o "$GWAS/Teumer2019_MD5SUM" https://ckdgen.imbi.uni-freiburg.de/files/Teumer2019/MD5SUM
( cd "$GWAS" && grep -E "$(ls formatted_*.gz | paste -sd'|')" Teumer2019_MD5SUM | md5sum -c - )

# 4) 汇总表：trait, file, url, build, N(中位), ancestry, md5
python - <<'PY'
import sys, gzip, hashlib, statistics
sys.path.insert(0, "scripts/lib")
from pathlib import Path
from repro import load_config, md5, write_provenance, ROOT
g = load_config()["genetics"]
rows, inputs = [], []
for t, v in g["traits"].items():
    p = ROOT / g["gwas_dir"] / v["file"]; inputs.append(p)
    ns = []
    with gzip.open(p, "rt") as fh:
        sep = None if v["format"] == "ckdgen" else "\t"
        hdr = fh.readline().rstrip("\n").split(sep)
        col = "n_total_sum" if v["format"] == "ckdgen" else f"n.{v['stratum']}"
        i = hdr.index(col)
        for k, line in enumerate(fh):
            if k >= 200000: break
            f = line.rstrip("\n").split(sep)
            if len(f) != len(hdr): continue
            x = f[i]
            if x not in ("NA", ""): ns.append(float(x))
    rows.append([t, v["file"], v["url"], "GRCh37", f"{statistics.median(ns):.0f}", f"{max(ns):.0f}",
                 v["ancestry"], v["source"], md5(p)])
tools = ROOT / g["tools_dir"]
for f in ["magma_v1.10_static.zip", "NCBI37.3.zip", "g1000_eur.zip"]:
    rows.append(["magma_resource", f, g["magma"]["url_" + {"magma_v1.10_static.zip": "binary", "NCBI37.3.zip": "geneloc", "g1000_eur.zip": "ref"}[f]],
                 "GRCh37", "", "", "EUR" if "g1000" in f else "", "MAGMA v1.10 resources (cncr.nl/research/magma)", md5(tools / f)])
out = ROOT / "results/00_download/gwas_md5.tsv"
with open(out, "w") as fh:
    fh.write("trait\tfile\turl\tbuild\tN_median_first200k\tN_max_first200k\tancestry\tsource\tmd5\n")
    for r in rows: fh.write("\t".join(r) + "\n")
write_provenance("07a_download_gwas", inputs, [out], extra={"note": "GWAS + MAGMA resources"})
print(open(out).read())
PY
