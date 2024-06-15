# GVRP-pipeline

GVRP-pipeline is a Python-based tool developed to refine Variant Call Format (VCF) files produced by DeepVariant. It allows for the processing of VCF files by offering functionalities to input, output, and optionally delete variants from the VCF files.

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
