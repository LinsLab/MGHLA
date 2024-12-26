# MGHLA

# Overview
    This repository is the source code of our paper "MGHLA: Multidimensional Graph-based Prediction of Class I HLA-Peptide Binding."

# Environment Setting
    This code is based on Pytorch. You can find the specific environment and packages in the requirements.txt file.

# Running the code
    python main.py: Used for training a new model.
    python test.py: Used for testing trained models on various datasets.


# Files and Functions:
    main.py: Main function running file.
    main_ablation.py: Ablation function running file (need to change file storage paths, names, and models).
    Loader.py: Dataset splitting and packaging.
    utils.py: Class-topology graph construction.
    train_test.py: Training, validation, and testing functions.
    feature_extraction.py: Contact graph: Residue physicochemical property extraction.
    data_transformer.py: 3D molecular graph feature calculation.
    performance.py: Performance metric calculation.
    test.py: Testing on various datasets.
    test_len_supertypr.py: Testing performance after dividing the dataset based on different lengths or other angles.
    Loader_only_onehot.py: Ablation experiment (MGHLA_onehot) dataset splitting and packaging functions.
    feature_extraction_only_onthot.py: Ablation experiment (MGHLA_onehot) residue physicochemical property extraction.


# Files in the MGpHLA/model folder:
    main_model.py: Main network architecture.
    ablation_models.py: Ablation experiment network architecture.
    gvp_gnn: Molecular structure feature learning network.
    kan.py: KAN network (used in the classifier).
    pep_encoder: Peptide feature learning network.



# Auxiliary files:
    graph_prepare: Generates contact graphs from HLA sequences.
    pos_cluster: Positive data peptide clustering, generating class-topology node embeddings.
    pytorchtools: Early stopping.


# Data files (data):
    aphlafold2: PDB files of 112 HLA molecules predicted using AlphaFold2.
    concat: HLA sequences and their corresponding keys.
    fold_data: Training and testing datasets.
    hla_hla: Class-topology (supertype graph) related data.
    iedb_subset_new: Subset of data available for baseline methods in External_2.
    Independent_subset_new: Subset of data available for baseline methods in the Independent test set.
    Texternal_subset_new: Subset of data available for baseline methods in Externl_1.
    pre_process: Contact graph files of HLA molecules, with corresponding keys available in the concat file.


# Folders:
    baseline_length_supertype: Results of baseline methods under different lengths and supertypes.
    models: Model file storage.
    results: Storage for performance files during training and testing for each epoch.
    models_ablation: Ablation experiment model storage.
    results_ablation: Ablation experiment training results storage.
