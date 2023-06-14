# Multiple instance learning improves Ames mutagencity prediction for problematic molecular species
A Github repository to accompany the paper of the same name

## Simply overview
This repository contains all the code required to build and analyze the models presented in the associated paper (this will be linked when published). Folders contain the data required in the main files. The main code in contained in the jupyter notebooks (.ipynb) executed in from 1 to 6. Additionally the excel sheet the results have been manipulated in to generate the figures has been supplied.

## Contents
#### Folders
* MIL_functions:  containes 5 python files that contain the defintion of the functions used in the core code. These are defined here to allow easier reading of the core code using functions describing the action performed.
* data:           This folder contains the required input data for the generation of the models. Raw contained the raw data taken from the OECD toolbox, Ates, Bruschweiler and Hansen. The folder also contains the calculated metabolites and the data encoded with MACCS keys and the Morgan fingerprint.
* model_results:  This folder contains the results of the models. Internal and external refer to the internal and external validation of the aromatic amines data while hansen refers to the hansen dataset. The exported_results folder contains the results in a more readable format. For interpreting the results please go to module 4 or 6 of the core code.
* tpot models:    this folder contains the saved tpot model pipelines in the format generated by tpot.
#### Core code
* 1-data_selection.ipynb      : Reads from the raw input data to generate an sdf and csv file of the molecules in the datasets. Data was cleaned and merged into a unified format. Only molecules that could be passed in RDKit in this notebook were kept in the the dataset.
* 2-data_encoding.ipynb       : Encoding for the datasets is precalculated to save on time and speed up repeated usage of these encoded forms.
* 3-model_building.ipynb      : Training and testing of all the models featured.
* 4a-initial_results_analysis : Reading and formatting of the results including calculation of the metrics of performance and printing these results to be available for analysis in excel.
* 4b-PCA_graphing.ipynb       : Generates the principal component analysis (PCA) graphs of the two datasets base on molecules MACCS keys.
* 5-data_output.ipynb         : Reading and formatting of the results and the outputting them in a usable csv format.
* 6-morgan_analysis.ipynb     : Analysis of the polynomial kernal based normalised set kernel (NSK) on the Morgan fingerprints (PKNMo) model. Features calculation of predictive capability metrics on specific molecular groups defined in the paper. Molecular groups not seen here were analysed using substructures in DataWarrior.
#### Other
* MIL figures : excel spreadsheet featuring the transformed data for the presented results and figures

## Dataset information
The datasets used in the paper are supplied in this github.
The "Aromatic amine" dataset (n=457): contains the aromatic amines found in the collected datasets.
  found at: ./data/raw/selected_molecules.csv
The adapted Hansen dataset (n=6505): This is the Hansen dataset after the removal of the molecules whose smiles couldn't be read by RDKit.
  found at: ./data/raw/hansen_raw/Hansen_all_molecules.csv

## Other notes
* There are quite a few packages required to actually run the code given here. If you have experience in python or similar languages this shouldn't be a problem. If not then most packages are popular and how to install them should be available online. The exception is Gary Doran's MIL package which the github link associated will will be printed to the terminal if you need it and don't have it.
* All model building and data splitting sections should be given a set seed so results should be reproducible. If not the results gathered are those saved in the repository.
