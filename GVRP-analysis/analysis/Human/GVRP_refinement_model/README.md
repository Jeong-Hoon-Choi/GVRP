# GVRP Analysis on Human

This path contains the source code for training and evaluating the GVRP refinement on human variants. The descriptions for each code are as follows.

## Source Code

- **common_parameter.py**: This source code includes the definitions of all the libraries and common parameters required in this path.

- **common_module.py**: This source code contains modules commonly used in the training and evaluation processes for Human variant data. It includes data loading/splitting, model, and metric modules.

- **preprocess_for_GVRP_human.py**: This is the source code for encoding training data from human VCF files called using DeepVariant, includes convert vcf to csv, extract the features. Thre results save into the data/human/csv directory in GVRP-analysis.

- **fft.py**: This is the source code for the MLP (feed-forward network) used in training.

- **ftt.py**: This is the source code for the FT-Transformer based on PyTorch used in training.

- **GVRP_refinement_model_training_on_human.py**: This is the source code for training models on features extracted from human VCF files. The models are trained on HG001 and HG002 data using XGB, LGBM, RF, NB, kNN, MLP, and FTT. The trained models are saved in the GVRP-analysis directory under model/"TRAINING_TEST_DATA"/"VARIANT_TYPE"+"MODEL_NAME".

- **GVRP_refinement_model_inference_on_human.py**: This is the code for loading and evaluating the GVRP refinement model trained on human VCF data. The results are saved in the GVRP-analysis directory under result/human/csv.

- **ROC_for_human.py**: This is the code for loading the GVRP refinement model trained on human VCF data and plotting the ROC curve. The results are saved in the GVRP-analysis directory under result/human/roc.
