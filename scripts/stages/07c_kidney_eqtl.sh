#!/usr/bin/env bash
# Stage 07c：下载公开人肾 eQTL 汇总统计（URL 来自 config/run.yaml genetics.eqtl），记录 MD5 与是否为全量统计。
#  - GTEx v8 肾皮质（eQTL Catalogue QTD000261，全量 cis 统计，GRCh38，N=73）→ coloc
#  - Susztak 实验室肾小管 / 肾小球 / meta eQTL（仅 FDR<0.01 显著对，GRCh37）→ SMR + HEIDI
# NephQTL2（NEPTUNE GLOM/TUB）只在 hugeampkpn.org 网页端浏览，没有可脚本下载的全量文件，记录为不可用。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/data/raw/eqtl"; mkdir -p "$OUT" results/00_download

python - <<'PY' > "$OUT/.urls.tsv"
import sys; sys.path.insert(0, "scripts/lib")
from repro import load_config
for k, v in load_config()["genetics"]["eqtl"].items():
    if isinstance(v, dict):
        print(k, v.get("file", v["url"].rsplit("/", 1)[-1]), v["url"], sep="\t")
PY

while IFS=$'\t' read -r key f url; do
  [ -s "$OUT/$f" ] || curl -fSL --retry 10 -C - -o "$OUT/$f" "$url"
  case "$f" in QTD*) [ -s "$OUT/$f.tbi" ] || curl -fsSL -o "$OUT/$f.tbi" "$url.tbi";; esac
done < "$OUT/.urls.tsv"
for z in "$OUT"/*.gz; do gzip -t "$z"; done

python - <<'PY'
import sys; sys.path.insert(0, "scripts/lib")
from repro import load_config, md5, write_provenance, ROOT
e = load_config()["genetics"]["eqtl"]
out = ROOT / "results/00_download/eqtl_md5.tsv"
rows, files = [], []
for k, v in e.items():
    if not isinstance(v, dict):
        continue
    f = ROOT / "data/raw/eqtl" / v.get("file", v["url"].rsplit("/", 1)[-1]); files.append(f)
    rows.append([k, f.name, v["url"], v.get("build", "GRCh37"), str(v["n"]),
                 "full" if v["full"] else "significant_pairs_only", v["source"], md5(f)])
rows.append(["nephqtl2", "", "https://www.nephqtl2.org/", "", "", "not_downloadable",
             "NephQTL2 (NEPTUNE glomerular/tubulointerstitial eQTL); browser only, no bulk download", ""])
with open(out, "w") as fh:
    fh.write("dataset\tfile\turl\tbuild\tN\tcontent\tsource\tmd5\n")
    for r in rows: fh.write("\t".join(r) + "\n")
write_provenance("07c_kidney_eqtl", [], files + [out])
print(open(out).read())
PY
