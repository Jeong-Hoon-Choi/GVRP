# Genome Variant Refinement Pipeline (GVRP)

The Genome Variant Refinement Pipeline (GVRP) is a Python-based tool developed to refine Variant Call Format (VCF) files produced by DeepVariant. In GVRP we separate in "analysis" and "pipeline" for usage. In analysis, we provide the source code that can reenact the results in paper. On the other hand, in pipeline, you can use the GVRP as the pipeline for refining the variants in VCF file. It allows for the processing of VCF files by offering functionalities to input, output, and optionally delete variants from the VCF files.

## Requirements

The GVRP is written in Python 3.11. All necessary libraries required to run GVRP are listed in the `requirements.txt` file in this directory.

## Installation

Before running GVRP, ensure you have Python 3.11 installed on your system. Clone this repository to your local machine. To install the required libraries, navigate to the directory containing `requirements.txt` and run the following command:

```bash
pip install -r requirements.txt
```

This command will install all the necessary Python libraries listed in `requirements.txt`.

## Usage

GVRP can be run from the command line using the following options:

- `-i`/`--input`: Specifies the input path of the Variant Call Format (VCF) file from DeepVariant.
- `-o`/`--output`: Specifies the path where the refined VCF file will be output.
- `-d`/`--delete` (optional): Specifies the path where the list of deleted variants in VCF format will be saved.

### Example Command

```bash
python3 GVRP.py -i your_input_vcf_path.vcf -o your_output_vcf_path.vcf -d your_deleted_vcf_path.vcf
```

## Example Files

Example VCF files can be accessed through Google Drive:

- **Human VCF Example**: The files are available at [this link](https://drive.google.com/drive/folders/1CddCgFMFvMPaHo6t_eZ5k2TbM_S29Zhi?usp=sharing). These include VCF files for GIAB's HG001 and HG002 with 60x coverage genome sequences, preprocessed as non-human species genome sequence, which skipped local realignment and base recalibration.  Then call variants with DeepVariant, and seperated in variant types.

- **Rhesus Macaque VCF Example**: The VCF file for two individuals of Rhesus macaque (MMUL.IN-35060 and MMUL.CH-35082) can be found at [this link](https://drive.google.com/drive/folders/1o18M1YF37rrGnsJMMEFwbJOBEI_zZsRp?usp=drive_link). These VCF files contain genome variant called by DeepVariant respectively.

## ReadMe

For detailed descriptions of each directory, please refer to the readme file within each directory.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (No. 2021R1A2C2010775) and the Institute of Information & Communications Technology Planning & Evaluation (IITP) under the Artificial Intelligence Convergence Innovation Human Resources Development (IITP-2023-00254177)