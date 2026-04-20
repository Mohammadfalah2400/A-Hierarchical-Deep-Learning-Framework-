## 📄 Project Overview

This repository contains the implementation and supporting materials for the work:

**"Hierarchical Deep Learning for De Novo Molecular Structure Prediction from EI-MS Spectra"**

This work was developed in collaboration with:

- Mohammad Falah — Maastricht University
- Anna Wilbik — Maastricht University  
- Marcin Pietrasik — Maastricht University  
- John Mommers — Envalior, Engineering Materials  

The research focuses on applying hierarchical deep learning methods to predict molecular structures directly from Electron Ionization Mass Spectrometry (EI-MS) spectra.
===========================================================================================================================================
This project implements a molecular structure prediction pipeline using multiple training strategies.  
The workflow includes data preprocessing, model training, clustering, and evaluation.

The project is divided into two main stages:

- First and Second Stages  
- Final Stage  

---

# First and Second Stages

This stage focuses on data preparation and model training.  
It contains five main files.

## 1. Data Cleaning and Preprocessing

This script is responsible for:

- Cleaning the dataset  
- Removing invalid samples  
- Performing data preprocessing  
- Defining the chemical subgroup classes using the RDKit library based on the file names  
- Preparing the training, validation, and test sets  

This ensures the dataset is properly structured before training.

---

## 2. Model Training

There are four training scripts in this stage.

### Training Without K-Fold Cross-Validation

- Uses: `Mymodel.py` and `Training_the_models`
- Trains the models using a standard train/test split  
- Tests the models on the test set  
- No cross-validation is applied  

### Training With K-Fold Cross-Validation

- Uses: `mymodel_` and `Training_the_models_with_Kfold`
- Performs training and evaluation using K-Fold cross-validation  
- The number of folds can be specified  
- Provides more robust evaluation results  

---

# Final Stage

The final stage contains the complete implementation of the pipeline and includes two approaches.\
---------------------------------------------------------------------------------------------------
Preprocessing & Multi-Architecture Training

Uses data_cleaning2.ipynb for full dataset preprocessing/cleaning

Uses selfies_featurization_one_hot.ipynb to generate SELFIES one-hot features

Trains and compares multiple multitask model architectures using SELFIES_DC_multitask.py and SELFIES_multitask_tuner.ipynb

## A. Basic Code Without Clustering

- Implements the full training pipeline  
- Trains a single model on the entire dataset  
- No clustering is applied  
- Represents the baseline approach  

---

## B. Basic Code With Clustering

This implementation applies clustering before training.

The dataset is divided into two molecular groups:

- Aromatic  
- Non-Aromatic  

The clustering folder contains two subfolders:

### Aromatic_codes
Contains training scripts for the Aromatic group.

### Non_Aromatic_codes
Contains training scripts for the Non-Aromatic group.

In total, there are 43 training scripts across both folders.  
Each script trains and evaluates a model for a specific cluster or configuration.

---

# Important Notes

## ⚠️ Path Configuration

Before running any scripts, you must update all file paths according to your local environment.

The current paths in the scripts may point to specific directories.  
Please modify them to match the location of your dataset and project folders on your machine.

---

## ⚠️ Data Cleaning and Preprocessing Dependency

The **Data Cleaning and Preprocessing** script must be executed first.

This script generates the processed files required for:

- Training the models in the First and Second Stages  
- Evaluating the models in the First and Second Stages  
If this step is skipped, the remaining scripts will not run correctly because the required processed datasets will be missing.

---

To link all models and stages together, a script called Code_auto is provided. This script integrates and coordinates the execution of all models across the different stages
