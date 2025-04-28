# Non-Adherence vs Composite Adverse Outcomes: Codebase for CHIL 2025  

This repository contains the code and analysis for our paper on non-adherence and composite adverse outcomes, accepted at CHIL 2025. It includes implementations of causal meta-learners for survival analysis, experiment scripts, and result analyses.  

## Repository Structure  

### **Main Codebase** (`causal_meta_learners/`)  
Contains core Python scripts for causal inference and survival analysis:  

- `causal_inference_modeling.py`:  
  - `get_meta_learner_results()`: Runs causal survival experiments across different setups.  
  - Positivity check for verifying covariate balance.  
- `experiment_setup.py`: Prepares data for causal survival modeling.  
- `meta_learners.py`: Implements causal inference meta-learners adapted for survival analysis.  
- `survival_models.py`: Implements survival analysis models.  
- `utils.py`: Utility functions for data handling and modeling.  

### **Experiments & Analysis Notebooks** (`causal_survival_experiments_notebooks/`)  
Jupyter notebooks for running experiments and analyzing results:

(Pre-run cell outputs are saved to showcase the results presented in the paper)  

- `survival_experiment_random_seed.ipynb`: Main experiments across multiple random seeds.  
- `survival_experiment_random_seed_ablation.ipynb`: Ablation study excluding county-provided risk scores.  
- `survival_experiment_new_data.ipynb`: Analysis of unadjusted survival curves and ITEs.  
- `causal_survival_forest_experiment_random_seed.ipynb`: Causal survival forest implementation in R via Python.  
- `causal_survival_forest_experiment_random_seed_ablation.ipynb`: Ablation version of the above.  
- `survival_experiment_results_analysis.ipynb`: Result analysis, including figures used in the paper.  
- `survival_experiment_results_analysis_ablation.ipynb`: Analysis of ablation study results.  
- `expanded_data_analysis.ipynb`: Cohort analysis, preprocessing, and dataset-related figures for the paper.  

### **Additional Folders & Files**  
- `data_splits/`: Folder containg our raw data (Empty - Data removed due to IRB restrictions of sharing our code.)  
- `pycox/` & `torchtuples/`: External packages for survival modeling.  
- `environment.yaml`: Conda environment file with dependencies.  
- `notebook_tutorial/`: Folder containing the jupyter notebook that serves as the tutorial for a step-by-step approach of using our method for extending causal meta-learners to survival analysis modeling.  

## Usage Instructions  

1. **Set up the environment**  
   ```bash
   conda env create -f environment.yaml
   conda activate <env_name>
   ```  
2. **Run Experiments**  
   - Use `causal_survival_experiments_notebooks/` for step-by-step guidance on running models and analyzing results.  
   - Modify `causal_meta_learners/causal_inference_modeling.py` for custom experiment configurations.  
