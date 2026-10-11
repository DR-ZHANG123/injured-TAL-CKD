"""build_additional_files: Additional files 2-5 (xlsx) assembled from result tables under results/.

Each workbook starts with a README sheet (title, description, one plain-English note per sheet, glossary of the codes
that appear in the tables). Values are copied, never recomputed; individual identifiers are replaced by sequential
codes; columns holding trait, gene-set, eQTL or slide read-out codes get a readable companion column.
Usage: python scripts/build_additional_files.py [--out submission/03_additional_files]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# ---------------------------------------------------------------- readable names for codes used in result tables
TRAITS = {"egfr_EA": "eGFR (creatinine), all individuals of European ancestry",
          "egfr_DM": "eGFR (creatinine), individuals with diabetes",
          "egfr_noDM": "eGFR (creatinine), individuals without diabetes",
          "uacr_EA": "Urine albumin-to-creatinine ratio, all individuals of European ancestry",
          "uacr_DM": "Urine albumin-to-creatinine ratio, individuals with diabetes"}
GENE_SETS = {"tal_healthy": "Genes of healthy TAL cells (healthy TAL state genes and genes lost in injured TAL cells)",
             "tal_healthy_specific": "Genes specific to healthy TAL states",
             "tal_injured_down": "Genes lost in injured TAL cells",
             "tal_injured_up": "Genes gained in injured TAL cells",
             "pt_injured_up": "Genes gained in failed-repair PT cells"}
EQTL = {"susztak_meta": "Whole-kidney eQTL meta-analysis", "susztak_tubule": "Microdissected tubule eQTL",
        "susztak_glom": "Microdissected glomerulus eQTL", "gtex_kidney_cortex": "GTEx kidney cortex eQTL"}
READOUTS = {"tal_injured_logit": "Injured TAL fraction (logit)", "injured_TAL": "Injured TAL fraction (logit)",
            "pt_injured_logit": "Injured PT fraction (logit)",
            "tubular_axis": "Mean of standardized injured PT and injured TAL fractions",
            "ecgc_axis": "Glomerular capillary endothelial marker score in EC-GC cells",
            "ecgc_axis_all": "Glomerular capillary endothelial marker score in all EC-GC states",
            "ECGC": "Glomerular capillary endothelial marker score in EC-GC cells",
            "direct_eGFR_model": "Slide model trained directly on eGFR", "direct_eGFR": "Slide model trained directly on eGFR",
            "embedding_PC1": "First principal component of slide embeddings",
            "embedding_PC2": "Second principal component of slide embeddings",
            "embedding_PC3": "Third principal component of slide embeddings",
            "pred": "Slide estimate, PAS and H&E models averaged", "pred_HE": "Slide estimate, H&E model",
            "pred_PAS": "Slide estimate, PAS model"}
READABLE = {"trait": ("trait_name", TRAITS), "gene_set": ("gene_set_name", GENE_SETS), "eqtl": ("eqtl_name", EQTL),
            "smr_tissue": ("smr_eqtl_name", EQTL), "target": ("readout_name", READOUTS),
            "readout": ("readout_name", READOUTS)}

GLOSSARY = [
    ["REF", "Reference tissue (KPMP healthy reference and reference categories)"],
    ["DKD / HKD", "Diabetic / hypertensive kidney disease (pathology-adjudicated KPMP categories)"],
    ["AKI", "Acute kidney injury"],
    ["DM_resilient / Diabetes Mellitus - Resilient", "Diabetes without chronic kidney disease"],
    ["snRNA / scRNA", "Single-nucleus / single-cell RNA sequencing"],
    ["aTAL / frTAL", "Adaptive / failed-repair thick ascending limb cells; injured TAL = aTAL + frTAL"],
    ["aPT / frPT", "Adaptive / failed-repair proximal tubule cells; injured PT = aPT + frPT"],
    ["EC-GC", "Glomerular capillary endothelial cells"],
    ["log2fc", "log2 fold change, injured versus healthy cells of the same participant or donor"],
    ["fdr", "Benjamini-Hochberg false discovery rate"],
    ["hedges_g_logit", "Hedges' g of logit fractions"],
    ["per SD / perSD", "Per standard deviation of the logit injured fraction or of the slide estimate"],
    ["p_hc3", "P value with heteroscedasticity-consistent (HC3) standard errors"],
    ["Participant and donor codes", "Sequential codes (P0001 ...) that replace original identifiers; the same "
                                    "identifier receives the same code in every sheet of a workbook"],
]

BOOKS = {
    "Additional_file_2.xlsx": {
        "title": "Injured cell-state fractions",
        "description": "Per-participant injured TAL and PT fractions in KPMP single-nucleus and single-cell data, including "
                       "acute kidney injury and diabetes without kidney disease; procurement, age and sex models; "
                       "independent cohorts GSE211785, GSE195460 and GSE271188; agreement between sequencing methods.",
        "sheets": [
            ("KPMP_TAL_states", "22_tal_states/participant_fractions.tsv",
             "Per participant and atlas: TAL cell counts, adaptive, failed-repair and injured TAL fractions, failed-repair "
             "share of injured TAL cells, injured fraction with a cortical-only denominator, diagnosis, KPMP condition "
             "category, age band, sex, procurement type and eGFR category midpoint."),
            ("KPMP_TAL_PT", "16_fig_tables/kpmp_donor_fractions.tsv",
             "Per participant and atlas: injured PT and injured TAL fractions with cell counts and eGFR category; "
             "independent_of_snRNA marks single-cell participants absent from the single-nucleus atlas."),
            ("KPMP_group_summary", "22_tal_states/group_summary.tsv",
             "Median and interquartile range of each fraction by diagnostic group, including AKI and diabetes without kidney disease."),
            ("KPMP_group_tests", "22_tal_states/group_contrasts.tsv",
             "Group comparisons: Hedges' g of logit fractions, Mann-Whitney P, and regression with age and sex."),
            ("procurement_groups", "21_confounders/procurement_groups.tsv",
             "Injured fractions of reference tissue by procurement type and of DKD and HKD."),
            ("procurement_tests", "21_confounders/procurement_contrasts.tsv",
             "Comparisons between procurement types and of DKD and HKD against percutaneous reference biopsies."),
            ("diagnosis_models", "21_confounders/dx_models.tsv",
             "Diagnosis effects on the logit injured fraction without covariates, with age and sex, with procurement, "
             "and among percutaneous biopsies."),
            ("cortical_denominator", "22_tal_states/cortical_vs_main.tsv",
             "Agreement of the injured TAL fraction with a cortical-only denominator and the main definition."),
            ("GSE211785_donors", "19a_gse211785_relabel/donor_fractions.tsv",
             "Injured TAL fraction per donor and library technology; analysis gives the labelling reference and whether "
             "the single-nucleus library was preferred."),
            ("GSE211785_tests", "19a_gse211785_relabel/abundance.tsv", "Group comparisons, one observation per donor."),
            ("GSE195460_donors", "19b_gse195460/donor_fractions.tsv",
             "Injured TAL fraction per donor with library chemistry, eGFR and interstitial fibrosis category."),
            ("GSE195460_tests", "19b_gse195460/abundance_clinical.tsv",
             "DKD versus control, with and without library chemistry as covariate, and clinical correlations."),
            ("GSE271188_donors", "16_fig_tables/gse271188_donor_fractions.tsv", "Injured TAL and PT fractions per participant."),
            ("GSE271188_tests", "16_fig_tables/gse271188_abundance.tsv", "Group comparisons (Hedges' g of logit fractions, Mann-Whitney P)."),
            ("agreement_by_lineage", "21_confounders/sn_sc_agreement_lineages.tsv",
             "Agreement of injured fractions measured by both atlases in the same participants: Spearman, Pearson (logit), "
             "intraclass correlation and Bland-Altman bias with limits of agreement."),
            ("agreement_rho_difference", "21_confounders/sn_sc_rho_difference.tsv",
             "Difference in Spearman rho between the TAL and each other lineage, with bootstrap confidence intervals."),
            ("agreement_participants", "16_fig_tables/sn_vs_sc_donors.tsv", "Injured fractions of the same participant by both methods."),
            ("agreement_TAL_states", "22_tal_states/sn_vs_sc_agreement.tsv",
             "Agreement of adaptive, failed-repair and pooled injured TAL measures between methods."),
            ("TAL_state_profiles", "22_tal_states/sn_sc_state_profile_concordance.tsv",
             "Correlation of state expression profiles (relative to healthy TAL) between the two atlases."),
        ]},
    "Additional_file_3.xlsx": {
        "title": "Gene programme of injured TAL cells",
        "description": "Discovery statistics for injured versus healthy TAL cells in KPMP single-nucleus data, replication in "
                       "KPMP single-cell participants, GSE271188, GSE211785 and GSE195460, replication without labelling "
                       "genes and with split gene halves, TAL function genes and urinary injury markers, EGF analyses, and in situ UMOD and neighbourhood measurements.",
        "sheets": [
            ("replication_summary", "24_replication_summary/replication_over_discovery_genes.tsv",
             "Correlation of log2 fold changes and replication rate over the 1212 discovery genes in each replication set."),
            ("discovery_genes", "15_key_genes/aTAL_frTAL_key_genes.tsv",
             "The 1212 discovery genes. log2fc and fdr: discovery; *_kpmp_sc and *_gse271188: replication sets; "
             "replicated_*: same direction and absolute log2 fold change of at least 0.5; egfr_*: rank association with "
             "eGFR in injured TAL cells; genetic_*: lines of genetic evidence."),
            ("in_situ_UMOD", "10f_xenium_umod/donor_umod.tsv",
             "Xenium (GSE286184): per donor, UMOD transcripts per 100 other panel transcripts in injured and healthy TAL cells, "
             "with TAL states reassigned without UMOD."),
            ("in_situ_UMOD_tests", "10f_xenium_umod/umod_paired_tests.tsv", "Wilcoxon signed-rank tests of the per-donor differences."),
            ("in_situ_neighbours", "10e_xenium_tal/donor_tal.tsv",
             "Xenium: per donor, injured TAL fraction and the share of injured and healthy TAL cells with a fibroblast or immune cell within 50 µm."),
            ("in_situ_neighbour_tests", "10e_xenium_tal/tal_paired_tests.tsv", "Wilcoxon signed-rank tests of the per-donor differences."),
            ("discovery_all_genes", "15_key_genes/aTAL_frTAL_discovery.tsv",
             "Paired comparison of injured and healthy TAL cells for all expressed genes (174 participants)."),
            ("replication_summary", "16_fig_tables/key_gene_replication.tsv", "Replication rates and fold-change correlations."),
            ("discovery_summary", "15_key_genes/summary.tsv",
             "Numbers of discovery genes and replication by injured state (injured TAL, failed-repair TAL, failed-repair PT)."),
            ("GSE211785_log2fc", "19a_gse211785_relabel/programme_lfc.tsv", "Injured versus healthy TAL cells per gene, donor-level paired comparison."),
            ("GSE211785_concordance", "19a_gse211785_relabel/programme_concordance.tsv",
             "Concordance with discovery by technology, with all genes and without labelling genes."),
            ("GSE195460_log2fc", "19b_gse195460/programme_lfc_all_state_labels.tsv", "Injured versus healthy TAL cells per gene, donor-level paired comparison."),
            ("GSE195460_concordance", "19b_gse195460/programme_concordance.tsv",
             "Concordance with discovery, with all genes and without labelling genes."),
            ("without_labelling_genes", "15b_key_genes_label_free/sensitivity.tsv",
             "Replication in KPMP single-cell participants and GSE271188 after removal of the genes used to assign labels."),
            ("split_halves_summary", "19c_label_transfer_circularity/summary.tsv",
             "Labels from one random half of the genes, replication scored on the other half (20 splits), for the injured TAL "
             "programme and for healthy state pairs; full_* columns give the values with all genes."),
            ("split_halves_runs", "19c_label_transfer_circularity/runs.tsv", "Results of each split."),
            ("TAL_function_and_markers", "22_tal_states/tal_gene_lookup.tsv",
             "TAL function genes and urinary injury markers in discovery and replication sets, for injured, adaptive and failed-repair TAL cells."),
            ("EGF_UMOD_vs_injury", "22_tal_states/tal_gene_participant_assoc.tsv",
             "Participant-level EGF and uromodulin expression in all TAL cells and in healthy TAL cells against the injured TAL fraction."),
            ("EGF_UMOD_participants", "22_tal_states/participant_tal_genes.tsv",
             "Participant-level EGF and uromodulin expression (log2 counts per million), diagnosis, eGFR and injured TAL fraction."),
            ("EGF_eGFR_models", "23_fig_tables_v2/egf_egfr_standardized.tsv",
             "Standardized association of TAL EGF expression with eGFR without covariates, with diagnosis, and with diagnosis "
             "and the injured TAL fraction."),
        ]},
    "Additional_file_4.xlsx": {
        "title": "Kidney-function genetics",
        "description": "Gene-level genetic evidence for eGFR and the urine albumin-to-creatinine ratio (MAGMA, colocalization "
                       "with kidney cortex eQTL, summary-data Mendelian randomization with the HEIDI test), competitive "
                       "gene-set tests with matched random gene sets and their detection rates, stratified LD score regression, and enrichment among genes specific to KPMP cell states.",
        "sheets": [
            ("evidence_by_gene", "07f_genetic_convergence/genetic_evidence_by_gene.tsv",
             "One row per trait and gene; n_lines = number of supporting lines of evidence."),
            ("evidence_summary", "07f_genetic_convergence/convergence_summary.tsv", "Genes per trait with one, two or three lines of evidence."),
            ("MAGMA_summary", "07b_magma/magma_summary.tsv", "Gene-level association summary per trait."),
            ("colocalization_summary", "07d_coloc/coloc_summary.tsv", "Colocalization with GTEx kidney cortex eQTL."),
            ("SMR_summary", "07e_smr/smr_summary.tsv", "Summary-data Mendelian randomization with the HEIDI test per trait and eQTL data set."),
            ("gene_sets", "18_programme_genetics/gene_sets.tsv", "Members of each tested gene set; in_magma marks genes with MAGMA statistics."),
            ("gene_set_tests", "18_programme_genetics/gene_set_tests.tsv",
             "MAGMA competitive gene-set tests (one-sided), standard model and conditioned on mean kidney expression."),
            ("matched_random_sets", "18_programme_genetics/empirical_null.tsv",
             "Mean gene Z of each set against 1000 random gene sets matched for kidney expression and gene length."),
            ("lineage_matched_sets", "18b_lineage_genetics/lineage_null.tsv",
             "Genetic signal of each healthy cell lineage (union of its subclass-specific genes) against the same 1000 expression- and length-matched random gene sets."),
            ("programme_power", "18c_programme_power/power.tsv",
             "Detection rate of the matched random-set test for gene sets of the size of the injured TAL programme: "
             "200 random subsets of TAL identity genes of that size, with the observed injured TAL programme for comparison."),
            ("SLDSC", "25_sldsc/sldsc_results.tsv",
             "Stratified LD score regression: heritability enrichment and coefficient (tau) of each gene-set annotation "
             "(gene body ± 100 kb) conditional on the baselineLD v2.2 model and an all-genes annotation; one-sided P for tau > 0."),
            ("programme_gene_evidence", "18_programme_genetics/supported_programme_genes.tsv",
             "Genes gained or lost in injured TAL cells with genetic support, and the direction of the SMR effect."),
            ("SMR_TAL_genes", "23_fig_tables_v2/genetic_support_genes.tsv",
             "TAL genes with SMR support: SMR effect (z_protective positive when higher expression accompanies better kidney "
             "function) and the change in injured TAL cells."),
            ("cell_state_enrichment", "13_gwas_cell_enrichment/enrichment.tsv",
             "Enrichment of association signals among the genes most specific to each KPMP single-nucleus cell state (one-sided P, rank, FDR)."),
            ("direction_tests", "14_genetic_direction/direction_tests.tsv",
             "Fraction of SMR genes whose expression change in disease points in the genetically adverse direction, with permutation P."),
        ]},
    "Additional_file_5.xlsx": {
        "title": "Slide estimate and clinical associations",
        "description": "Cross-validated slide estimates of the injured TAL fraction, comparison with pathologist descriptor "
                       "scores, robustness to scanner, acquisition year and colour, specificity controls, and associations "
                       "of the measured and slide-estimated injured TAL fraction with eGFR and albuminuria.",
        "sheets": [
            ("cross_validation", "11_histology/cv_prediction.tsv",
             "Cross-validated correlation, permutation P and partial correlations for each slide read-out and stain combination."),
            ("cv_by_participant", "11_histology/cv_predictions_by_participant.tsv", "Measured and cross-validated estimated values per participant."),
            ("within_diagnosis", "20_slide_robustness/within_diagnosis_cv.tsv",
             "Performance within diagnoses, with tubule-rich tiles only (tiles whose lumen fraction lies between 0.05 and 0.40 and whose eosin fraction is at least the slide median), by stain, and with models trained on DKD and HKD only."),
            ("temporal_validation", "20_slide_robustness/temporal_validation.tsv",
             "Training on earlier and testing on later biopsies, compared with random splits of the same sizes."),
            ("colour_removed", "20_slide_robustness/cv_colour_variants.tsv",
             "Cross-validated performance after colour statistics were regressed out of the embeddings."),
            ("descriptor_model", "20_slide_robustness/descriptor_baseline_cv.tsv",
             "Cross-validated performance of the six pathologist descriptor scores, the slide estimate and both together."),
            ("descriptor_increment", "20_slide_robustness/descriptor_delta_r.tsv", "Gain in correlation from adding the slide estimate to the descriptor scores."),
            ("descriptor_variance", "20_slide_robustness/readout_variance_explained_by_descriptors.tsv",
             "Variance of the slide estimate and of the measured fraction explained by descriptor scores and diagnosis."),
            ("acquisition_by_set", "20_slide_robustness/acquisition_by_set.tsv", "Slides by set, stain, scan year and scanner."),
            ("acquisition_tests", "20_slide_robustness/acquisition_shift_tests.tsv", "Differences in acquisition and colour between training and independent slides."),
            ("set_separability", "20_slide_robustness/set_classifier_auc.tsv", "Cross-validated AUROC separating training from independent slides."),
            ("stain_agreement", "20_slide_robustness/stain_agreement.tsv", "Agreement of the PAS and H&E estimates."),
            ("stain_tracking", "20_slide_robustness/stain_readout_tracking.tsv",
             "Correlation of the slide estimates with interstitial fibrosis, tubular atrophy and eGFR."),
            ("independent_estimates", "11_histology/out_of_sample_predictions.tsv",
             "Slide estimates in participants with slides but no single-cell data."),
            ("independent_eGFR_robustness", "20_slide_robustness/oos_egfr_robustness.tsv",
             "eGFR per standard deviation of the slide estimate in independent participants, adjusted for diagnosis and acquisition variables."),
            ("independent_all_readouts", "20_slide_robustness/oos_clinical_all_targets.tsv",
             "eGFR and albuminuria associations of slide read-outs for different targets in independent participants."),
            ("independent_controls", "20_slide_robustness/oos_egfr_null_comparison.tsv",
             "Position of each read-out among 1000 read-outs trained on permuted labels and 1000 random directions."),
            ("independent_increment", "20_slide_robustness/oos_incremental_value.tsv",
             "eGFR association of each read-out with and without descriptor scores and other read-outs."),
            ("direct_eGFR_model", "20_slide_robustness/direct_egfr_model.tsv", "Slide model trained directly on eGFR."),
            ("tile_score_scale", "20_slide_robustness/tile_score_scale.tsv",
             "Range of tile scores compared with slide estimates and measured fractions."),
            ("eGFR_models", "21_confounders/egfr_models.tsv",
             "eGFR per standard deviation of the measured injured TAL (and PT) fraction by population and adjustment, with HC3 P values."),
            ("albuminuria_models", "21_confounders/albuminuria_models.tsv",
             "Ordinal logistic regression of albuminuria category (KDIGO A1-A3); odds ratios per standard deviation."),
            ("fibrosis_models", "21_confounders/programme_fibrosis_models.tsv",
             "eGFR models within chronic kidney disease with interstitial fibrosis, and models of the injured TAL programme score."),
            ("all_diagnoses_models", "21_confounders/full_dx_sensitivity.tsv",
             "eGFR and albuminuria models with every adjudicated KPMP diagnosis as its own level."),
        ]},
}

ID_COLS = {"donor_id", "donor", "participant", "participant_id", "ParticipantID", "redcap_id", "gsm", "unit",
           "sample", "specimen", "library"}   # 个体标识列：以顺序编码代替（期刊要求不含个体参与者信息）


def pseudonymize(df: pd.DataFrame, codes: dict) -> pd.DataFrame:
    """把个体标识列替换为 P0001 式顺序编码；同一工作簿内同一标识得到同一编码，便于跨表对应。"""
    df = df.copy()
    for col in [c for c in df.columns if c in ID_COLS]:
        base = df[col].astype(str).str.split("|").str[0]
        df[col] = [codes.setdefault(v, f"P{len(codes) + 1:04d}") for v in base]
    return df


def readable(df: pd.DataFrame) -> pd.DataFrame:
    """首列无名时命名为 gene；性状、基因集、eQTL、切片读出代码旁插入可读名称列。"""
    df = df.rename(columns={c: "gene" for c in df.columns[:1] if str(c).startswith("Unnamed")})
    for col, (new, names) in READABLE.items():
        if col in df.columns and df[col].astype(str).isin(names).any() and new not in df.columns:
            df.insert(df.columns.get_loc(col) + 1, new, df[col].map(names))
    return df


def readme(spec: dict) -> pd.DataFrame:
    rows = [["Title", spec["title"]], ["Description", spec["description"]], ["", ""], ["Sheet", "Content"]]
    rows += [[name, note] for name, _, note in spec["sheets"]]
    rows += [["", ""], ["Code or column", "Meaning"]] + GLOSSARY
    used = {c for _, src, _ in spec["sheets"] for c in pd.read_csv(R / src, sep="\t", nrows=0).columns}
    if "trait" in used:
        rows += [["Trait names", "Readable trait names are given in the column trait_name next to each trait code"]]
    return pd.DataFrame(rows)


def build(out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for fn, spec in BOOKS.items():
        codes: dict = {}
        with pd.ExcelWriter(out / fn, engine="openpyxl") as xw:
            readme(spec).to_excel(xw, sheet_name="README", header=False, index=False)
            for name, src, _ in spec["sheets"]:
                df = readable(pseudonymize(pd.read_csv(R / src, sep="\t"), codes))
                df.to_excel(xw, sheet_name=name[:31], index=False)
        made[fn] = spec["title"]
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submission" / "03_additional_files"))
    for k, v in build(Path(ap.parse_args().out)).items():
        print(k, v)
