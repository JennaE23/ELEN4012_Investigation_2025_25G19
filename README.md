# IDENTIFICATION OF QUANTUM HARDWARE BASED ON NOISE FINGERPRINT USING MACHINE LEARNING

This repository accompanies the reports written for the 4th year Electical and Information Engineering Laboratory Investigation  Project. 

This file acts as a guide to this repository.

### Code:
All code is in the 'code/' folder. 
In the code folder there are the follwoing sub-folders:

1. The structured_investigation_notebooks folder.

This folder contains notebooks that are written as a walkthrough of the process followed to develop the results of this investigation. It is recommended that the reader starts with these.

2. The investigation_functions folder. 

This folder contains is an importable module that contains all of the functions written and used for the rest of the code.

3. The rough_code_sandbox folder.

This folder should be ignored. It contains the files that were used to test code as it was being developed. These will possibly be formulated into unit tests at a later date.

The backend_vars and the ml_models_vars files contain settings that are common to many functions.

If someone were to repeat this investigation, they should check that the IBM backends listed in backend_vars are still available and that they have the license to access these IBM backends.

If someone were to test other machine learning algorithms, they could add them to the modes classes stored in the ml_models_vars file.

### Results:
All results are stored in the _results folders. 

These are:

1. Hardware_results

This folder contains the raw counts of all of the jobs run on the IBM backends in csv format.

2. Simulated_results

This folder contains the raw counts of all of the jobs run using the 'old' Qiskit Fake_BackendV2 models.

3. Refreshed_Simulated_results

This folder contains the raw counts of all of the jobs run using the refreshed Qiskit Fake_BackendV2 models.

3. Refreshed_Hardware_results

This folder is not yet used at this point in this investigation and should be ignored. 

This folder contains the raw counts of all of the jobs run on the IBM backends roughly a month after the first set of Hardware_results.

4. Transpiled_circuits

This folder contains images of all of the transpiled circuits that the jobs results are from in image format.

5. ML_results

The accuracy and cross-validation accuracy from every different training and testing combination for every different set of hyper-parameters for KNN and SVM are stored in this folder.

6. Fingerprint_results

All of the statistical similarity and distance measures calculated between the different fingerprints are saved in this folder in csv format.