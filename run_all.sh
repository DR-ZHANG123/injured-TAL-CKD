#!/usr/bin/env bash
# 总控：按依赖顺序运行全部分析 stage 与出图脚本。
# 用法：bash run_all.sh [--resume|--force] [--skip=stageA,stageB]
#   --resume（默认）：results/<stage>/PROVENANCE.json 已存在则跳过该 stage
#   --force：全部重跑
# 环境：tal-ckd（environment.yml；切片阶段用 environment_slides.yml）；stage 17 读取全切片图像需要 openslide，在 tal-ckd-slides 环境中运行（见下）。
# stage 25（S-LDSC）通过 conda run 调用 ldsc 1.0.1（Python 2.7），需另建名为 ldsc 的环境（github.com/bulik/ldsc 的 environment.yml；切片阶段用 environment_slides.yml）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
MODE=resume; SKIP=""
for a in "$@"; do
  case "$a" in
    --force) MODE=force ;;
    --resume) MODE=resume ;;
    --skip=*) SKIP="${a#--skip=}" ;;
  esac
done
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tal-ckd
mkdir -p logs

# stage 名 | 结果目录（用于 --resume 判断）| 命令
STAGES=(
  "00b_download_kpmp|00_download|bash scripts/stages/00b_download_kpmp.sh"
  "00c_download_gse211785|00_download|bash scripts/stages/00c_download_gse211785.sh"
  "00d_download_gse271188|00_download|bash scripts/stages/00d_download_gse271188.sh"
  "03_kpmp_pseudobulk|03_kpmp_pseudobulk|python scripts/stages/03_kpmp_pseudobulk.py --mod snRNA scRNA"
  "05_markers|05_markers|python scripts/stages/05_markers.py"
  "04_kpmp_dkd_vs_hkd|04_kpmp_dkd_vs_hkd|python scripts/stages/04_kpmp_dkd_vs_hkd.py"
  "04b_kpmp_adjusted|04b_kpmp_adjusted|python scripts/stages/04b_kpmp_adjusted.py"
  "07a_download_gwas|07a_download_gwas|bash scripts/stages/07a_download_gwas.sh"
  "07b_magma|07b_magma|python scripts/stages/07b_magma_genes.py"
  "07c_kidney_eqtl|07c_kidney_eqtl|bash scripts/stages/07c_kidney_eqtl.sh"
  "07d_coloc|07d_coloc|python scripts/stages/07d_coloc_kidney.py"
  "07e_smr|07e_smr|python scripts/stages/07e_smr_kidney.py"
  "07f_genetic_convergence|07f_genetic_convergence|python scripts/stages/07f_genetic_convergence.py"
  "08_gse211785_replication|08_gse211785_replication|python scripts/stages/08_gse211785_replication.py"
  "09_gse271188_replication|09_gse271188_replication|python scripts/stages/09_gse271188_replication.py"
  "10_xenium_annotate|10_xenium_annotate|python scripts/stages/10_xenium_annotate.py"
  "10e_xenium_tal|10e_xenium_tal|python scripts/stages/10e_xenium_tal.py"
  "10f_xenium_umod|10f_xenium_umod|python scripts/stages/10f_xenium_umod.py"
  "11a0_wsi_embed|11a0_wsi_embed|conda run -n tal-ckd-slides python scripts/stages/11a0_wsi_embed.py"
  "11a_port_wsi_embeddings|11a_port_wsi_embeddings|python scripts/stages/11a_port_wsi_embeddings.py"
  "11b_histology|11_histology|python scripts/stages/11b_histology.py"
  "13_gwas_cell_enrichment|13_gwas_cell_enrichment|python scripts/stages/13_gwas_cell_enrichment.py"
  "14_genetic_direction|14_genetic_direction|python scripts/stages/14_genetic_direction.py"
  "15_key_genes|15_key_genes|python scripts/stages/15_key_genes.py"
  "15b_key_genes_label_free|15b_key_genes_label_free|python scripts/stages/15b_key_genes_label_free.py"
  "16_fig_tables|16_fig_tables|python scripts/stages/16_fig_tables.py"
  "17_fig45_tile_scores|17_fig45_tiles|conda run -n tal-ckd-slides python scripts/stages/17_fig45_tile_scores.py"
  "00e_download_gse195460|00_download|bash scripts/stages/00e_download_gse195460.sh"
  "18_programme_genetics|18_programme_genetics|python scripts/stages/18_programme_genetics.py"
  "18b_lineage_genetics|18b_lineage_genetics|python scripts/stages/18b_lineage_genetics.py"
  "18c_programme_power|18c_programme_power|python scripts/stages/18c_programme_power.py"
  "19a_gse211785_relabel|19a_gse211785_relabel|python scripts/stages/19a_gse211785_relabel.py"
  "19b_gse195460|19b_gse195460|python scripts/stages/19b_gse195460.py"
  "19c_label_transfer_circularity|19c_label_transfer_circularity|python scripts/stages/19c_label_transfer_circularity.py"
  "20a_slide_headers|20a_slide_headers|conda run -n tal-ckd-slides python scripts/stages/20a_slide_headers.py"
  "20_slide_robustness|20_slide_robustness|python scripts/stages/20_slide_robustness.py"
  "21_confounders|21_confounders|python scripts/stages/21_confounders.py"
  "22_tal_states|22_tal_states|python scripts/stages/22_tal_states.py"
  "23_fig_tables_v2|23_fig_tables_v2|python scripts/stages/23_fig_tables_v2.py"
  "24_replication_summary|24_replication_summary|python scripts/stages/24_replication_summary.py"
  "00f_download_ldsc|00f_download_ldsc|bash scripts/stages/00f_download_ldsc.sh"
  "25_sldsc|25_sldsc|python scripts/stages/25_sldsc.py"
)

for entry in "${STAGES[@]}"; do
  IFS='|' read -r name res cmd <<< "$entry"
  if [[ ",$SKIP," == *",$name,"* ]]; then echo "[skip] $name"; continue; fi
  if [[ "$MODE" == resume && -s "results/$res/PROVENANCE.json" && "$name" != 00* ]]; then
    echo "[done] $name"; continue
  fi
  echo "[run ] $name"
  eval "$cmd" 2>&1 | tee "logs/${name}.log"
done

echo "[figs] figure scripts (read results tables only)"
for f in Fig1_framework Fig2_tal_abundance Fig3_tal_function Fig4_genetics Fig5_slide Fig6_clinical graphical_abstract \
         FigS1_clinical_robustness FigS2_dkd_vs_hkd FigS3_external_replication FigS5_genetics \
         FigS6_slide_robustness FigS4_xenium_tal; do
  python "scripts/figures/$f.py" 2>&1 | tee "logs/fig_${f}.log"
done
python scripts/figures/finalize_sizes.py 2>&1 | tee logs/fig_finalize_sizes.log
