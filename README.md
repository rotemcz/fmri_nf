# fmri_nf

## mock dataset demo

This repo will allow you to perform a mock training of our fMRI neuro-feedback framework. Since the data we used is confidential, we provide code to create a mock dataset - containing all the data one needs to run our network, only with random instead of real measured values.

### steps

1. **create mock dataset**
	```bash
	python3.6 create_mock_data.py --n_subjects <n_subjects>
	```
	This will create a "healthy" subjects dataset, with n_subjects subjects.
	<br>
	<br>
	
2. **run the e2e training + testing**
	```bash
	python3.6 e2e_train.py --lr <f_learning_rate> --batch_size <f_batch_size> --epochs <f_epochs> --classification_epochs <classification_epochs> --classifier_lr <classifier_learning_rate> --classifier_batch_size <classifier_batch_size> --type healthy
	```
