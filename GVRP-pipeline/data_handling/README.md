# GVRP-pipeline Handling The Data

This path contains the source code for encoding the the converted csv to apply in GVRP refinement model and decodes csv into vcf after apply GVRP refinement model.

## Source Code

- **before_apply_model.py**: This is the source code that contains the modules which extracts the features for use in training.

- **affter_apply_model.py**: This is the source code that contains the modules for filtering variants based on the training results and converting the filtered results back to VCF format.
