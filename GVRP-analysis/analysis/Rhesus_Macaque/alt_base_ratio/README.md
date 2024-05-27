# GVRP Refinement Model Evaluation on Rhesus Macaque in Quality Aspect

This path contains the source code for evaluating the GVRP refinement on Rhesus Macaque in quality aspect. The descriptions for each code are as follows.

## Source Code

- **common_parameter.py**: This source code includes the definitions of all the libraries and common parameters required in this path.

- **common_module.py**: This source code contains modules commonly used in the process for analysis alt base ratio, includes smapling the variants, get alt base ratio, plot the KDE distribution and conduct the statistical analysis.

- **alt_base_ratio_of_rhesus_macaque_individual.py**: This is the source code for sampling variants for each rhesus macaque individual and comparing the alt base ratio between labels and GVRP. It also includes the analysis of the alt base ratio for GVRP newly detected variants. The results are saved in the GVRP-analysis directory under result/Rhesus_Macaque/"VARIANT_TYPE"/"INDIVIDUAL_ID"/.

- **concat_sampling_variants.py**: This is the source code for integrating the sampled variants from each individual, using the previously sampled results.

- **alt_base_ratio_of_whole_rhesus_macaque.py**: This is the source code for sampling variants for all rhesus macaque, comparing the alt base ratio between labels and GVRP, and analyzing the alt base ratio for GVRP newly detected variants. The results are saved in the GVRP-analysis directory under result/Rhesus_Macaque/"VARIANT_TYPE"/total/.
