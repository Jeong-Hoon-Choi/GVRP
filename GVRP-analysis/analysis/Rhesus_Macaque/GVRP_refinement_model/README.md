# GVRP Refinement Model Evaluation on Rhesus Macaque in Quantity Aspect

This path contains the source code for evaluating the GVRP refinement on Rhesus Macaque in quantity aspect. The descriptions for each code are as follows.

## Source Code

- **common_parameter.py**: This source code includes the definitions of all the libraries and common parameters required in this path.

- **preprocess_for_GVRP_rhesus_macaque.py**: These are the modules required to encode the VCF files of rhesus macaque individuals located in data/Rhesus_Macaque/vcf/ into a format suitable for evaluation with the GVRP refinement model.

- **GVRP_refinement_model_apply_rhesus_macaque.py**: This is the source code for preprocessing VCF files from data/Rhesus_Macaque/vcf/, encoding them into training data, and evaluating the GVRP refinement model with this data. The generated preprocessed files are saved in data/Rhesus_Macaque/csv/, and the evaluation results are saved in result/Rhesus_Macaque/csv/.

- **ROC_for_rhesus_macaque_individual.py**: This is the code for loading the GVRP refinement model trained on human VCF data and plotting the ROC curve for each rhesus macaque individual data. The results are saved in the GVRP-analysis directory under result/Rhesus_Macaque/roc.

- **ROC_for_total_rhesus_macaque.py**: This is the code for loading the GVRP refinement model trained on human VCF data and plotting the ROC curve for all rhesus macaque data. The results are saved in the GVRP-analysis directory under result/Rhesus_Macaque/roc.
