# DMAIC Case Studies — Dissertation Companion Repository

**Author:** Akshay Thummalapally  
**Date:** August 2025  
**Purpose:** Complete MATLAB implementations of the two DMAIC (Define–Measure–Analyze–Improve–Control) case studies from the dissertation *“Adapting Digital Tools and Lean Six Sigma to Industry 5.0”*.

This repository is organized for clarity, reproducibility, and transparency, so readers can easily verify analyses and replicate results.

---

## 📂 Repository Structure

```
/code/               # MATLAB scripts & shared configuration
   ├─ steel_plates_dmaic.m    # Case Study 1: Steel Plates Fault Analysis (Sustainability)
   ├─ cmapss_dmaic.m          # Case Study 2: C-MAPSS Predictive Maintenance (Resilience)
   └─ configcode.m            # Central configuration used by both scripts

/data/               # Raw input datasets (not tracked in Git)
/output/             # Generated results: figures (.png), tables (.csv), logs (.txt)
README.md            # This file
.gitignore           # Ensures /data and /output are ignored by Git
```

---

## 📊 Case Studies

### Case Study 1 — Steel Plates Faults (Sustainability Pillar)
- **Script:** `code/steel_plates_dmaic.m`
- **Dataset:** `SteelPlatesFaults_Clean.csv` (place in `/data/`)
- **Outputs:**
  - Pareto chart of fault categories (`Figure_A_Pareto_Chart.png`)
  - Correlation heatmap (`Figure_B_Correlation_Heatmap.png`)
  - Confusion matrix & ROC curve for logistic regression (`Figure_C_*`, `Figure_D_*`)
  - p-Chart for process control (`Figure_E_p_Chart.png`)
  - Tables A–D: features, regression results, DOE simulation, Cp/Cpk analysis

### Case Study 2 — NASA C-MAPSS FD001 (Resilience Pillar)
- **Script:** `code/cmapss_dmaic.m`
- **Datasets:**  
  - `train_FD001.txt`  
  - `test_FD001.txt`  
  - `RUL_FD001.txt`  
  *(place all in `/data/`)*
- **Outputs:**
  - Sensor degradation plots (`Figure_A_Sensor_Degradation.png`)
  - Failure histogram (`Figure_B_Failure_Histogram.png`)
  - Predicted vs. Actual RUL scatter (`Figure_C_Predicted_vs_Actual.png`)
  - SPC chart of residuals (`Figure_D_SPC_Residuals.png`)
  - Tables A–B: feature correlation, resilience improvement

---

## ⚙️ Configuration

All parameters are managed in **`code/configcode.m`**:
- Dataset filenames and paths
- Random seed for reproducibility (`cfg.seed = 42`)
- Analysis parameters (top-K features, thresholds, splits, batch size)

Both scripts use this shared config, ensuring consistent, replicable runs.

---

## ▶️ How to Run

1. **Requirements**
    - MATLAB R2021a or newer
    - Statistics and Machine Learning Toolbox
    - (Optional for Case Study 2: Predictive Maintenance Toolbox)
2. **Prepare Datasets**
    - Place required datasets into `/data/`.
    - `/data/` is ignored by Git; download datasets manually (UCI & NASA links below).
3. **Run Scripts**
    In MATLAB, from the repo root:
    ```matlab
    % Steel Plates case study
    run(fullfile('code','steel_plates_dmaic.m'))

    % C-MAPSS case study
    run(fullfile('code','cmapss_dmaic.m'))
    ```
4. **Check Outputs**
    - Figures (`.png`), tables (`.csv`), and logs (`.txt`) appear in `/output/`.
    - Filenames match dissertation conventions: *Figure A–E*, *Table A–D*, etc.

---

## 📖 Mapping to Dissertation

- **Figures:**  
  *Figure A–E* → `Figure_A_*` … `Figure_E_*` in `/output/`
- **Tables:**  
  *Table A–D* → `Table_A_*` … `Table_D_*` in `/output/`
- **Console metrics** (RMSE, MAE, MTBF, Cp/Cpk, etc.) are written to summary `.txt` files in `/output/`.

Every number, plot, and table in Chapter 4 can be reproduced directly from the code.

---

## 🔄 Reproducibility

- Fixed RNG seed (`cfg.seed = 42`) for stable results
- Shared configuration for consistent defaults
- Deterministic output filenames for easy traceability

---

## 📝 Notes

- `/data/` and `/output/` are excluded from Git via `.gitignore`
- Download datasets from:
    - **Steel Plates:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults)
    - **C-MAPSS FD001:** [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

✅ With this README, reader can:
- Instantly see repo structure
- Know which dataset goes where
- Run each study with one command
- Cross-check outputs against dissertation figures/tables

---
