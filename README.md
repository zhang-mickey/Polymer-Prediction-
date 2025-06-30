# Polymer-Prediction-

We are presented with data which consists of: SMILES (string representatio of polymer genome), Tg (glass transition temperature), Tc (Thermal conductivity), FFV ( Fractional free volume), Rg (Radius of gyration) and density. 

The ground truth is averaged form multiple runs of molecular dynamics simulation

The goal is to train models which using the SMILES forecast the above 5 metrics: Tg, Tc, FFV, Rg, and density.

# Evaluation 
**weighted Mean Absolute Error(wMAE)**

Submission File
The submission file for this competition must be a csv file. For each id in the test set, you must predict a value for each of the five chemical properties. 

The file should contain a header and have the following format.

```
```

# Visualization tools

RDKit


# mac

```
conda install -c conda-forge rdkit
conda install scikit-learn
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```