# Injured thick ascending limb cells in diabetic and hypertensive kidney disease

Code, parameters, result tables and figure source data for the article
"Injured thick ascending limb burden links differentiated cell identity, kidney-function genetics and renal
histology in diabetic and hypertensive kidney disease".

## Layout
- `config/run.yaml` — every analysis parameter; scripts read it and hardcode nothing.
- `scripts/stages/` — numbered analysis stages (download, participant-level profiles, state fractions, gene
  programme, genetics, slide model, clinical associations). Each writes tables and a `PROVENANCE.json`
  (input/output MD5, seed, parameter-file hash, environment) to `results/<stage>/`.
- `scripts/stages/11a0_wsi_embed.py` with `scripts/lib/{wsi,histo}.py` — tiling, tile quality control and
  Phikon-v2 embedding of KPMP whole-slide images.
- `scripts/figures/` — plotting only; reads `results/` and writes each panel together with its input table to
  `figures/source_data/`.
- `results/`, `figures/source_data/` — the tables behind every number and figure in the article.

## Reproduce
```bash
conda env create -f environment.yml            # environment "tal-ckd"
conda env create -f environment_slides.yml     # environment "tal-ckd-slides" (openslide, torch) for slide stages
# stratified LD score regression (stage 25) calls ldsc 1.0.1 in a Python 2.7 environment named "ldsc":
#   git clone https://github.com/bulik/ldsc && conda env create -f ldsc/environment.yml  (put ldsc/ on PATH)
bash run_all.sh --resume                        # skips stages whose PROVENANCE.json exists
bash run_all.sh --force                         # reruns everything
```
Raw data are public and are downloaded by the `00*_download*` stages (KPMP atlas v2 from CZ CELLxGENE Discover,
GEO series, CKDGen summary statistics, eQTL Catalogue, LD score regression references from Zenodo 10515792). KPMP whole-slide images, the slide manifest and the
pathologist descriptor scores come from the open-access tier of the KPMP repository
(https://atlas.kpmp.org/repository) and are placed under `data/raw/kpmp/`.

## Licence
MIT (see `LICENSE`).
