# DeepGRB

<img src="https://user-images.githubusercontent.com/93478548/189541951-83a118d4-0a6f-41f3-bc57-cbf0ab7623c2.png" width="750">

**DeepGRB** is a framework for serching astronomical events. Is composed by a background estimator, performed with a Neural Network, and on top applied an efficent trigger algorithm called FOCuS.

> ### Authors & contributors:
> **Riccardo Crupi**, **Giuseppe Dilillo**, Elisabetta Bissaldi, Fabrizio Fiore, Andrea Vacchi

To know more about this research work, please refer to:
- Searching for long faint astronomical high energy transients: a data driven approach. Riccardo Crupi, Giuseppe Dilillo, Elisabetta Bissaldi, Fabrizio Fiore, Andrea Vacchi. https://arxiv.org/abs/2303.15936
- Poisson-FOCuS: An efficient online method for detecting count bursts with application to gamma ray burst detection. Kester Ward, Giuseppe Dilillo, Idris Eckley, Paul Fearnhead. https://arxiv.org/abs/2208.01494

## Installation
Clone repo and install packages:
```
git clone https://github.com/rcrupi/DeepGRB.git
cd DeepGRB
pip install -r requirements.txt
```

To install the library **Fermi GBM Data Tools** follow this [page](https://fermi.gsfc.nasa.gov/ssc/data/analysis/rmfit/gbm_data_tools/gdt-docs/).

Python version: `3.6.8`

### Configuration

In the config file `connections/utils/config.py` specify the path folder **`PATH_TO_SAVE`** in which you want to download the data and save the results. The other folder names inside it can be specified.

### Quick start
```
python pipeline/pipeline_bkg.py
```

## Workflow
`pipeline/pipeline_bkg.py` is the main script in which you can set the period of interest and run the following steps:
1) download the data
2) preprocess the data
3) train and predict the bkg with an NN
4) perform FOCuS, build the catalog
5) localize the events and update the catalog

### Download
Given the start and end month (e.g. '03-2019' and '07-2019'), `download_spec(start_month, end_month)` run the download of the CSPEC and POSHIST files in the folder **`PATH_TO_SAVE\FOLD_CSPEC_POS`** specified in the config file.

### Preprocess
`build_table` builds the table containing the information\features of the satellites and the detector (e.g. Fermi geographical latitude, longitude, velocity, detectors pointing, ...) and the count rates observed by the detectors (target variables for the Neural Network). The bin time is 4.096s the energy range can be specified in `erange` and the resulting csv file is saved in **`PATH_TO_SAVE\FOLD_BKG`**.

### Train
It is trained a Feed Forward Neural Network with input the features of the satellites and the detector and as output the observed count rates.
`bool_del_trig` is a boolean option to delete in the training phase the events already present in the Fermi GBM catalog. 
Some hyperparameters can be set:
- loss_type: deafult loss function is 'mean', Mean Absolute Error.
- units: number of nodes in the first and second layer, the third is halved.
- epochs: number of epochs of the NN.
- lr: learning rate of the NN during training.
- bs: batch size of the NN during training.
- do: parameters for the dropout between layers.


One the NN class is trained the `predict` method can be used along the parameter `time_to_del` which define how many seconds to exclude before and after entering in the SAA.
In the `plot` method can be selected a time period (`time_r` and/or `time_iso`) and a detector/range (`det_rng`) to plot the count rates observed and estimated by the NN. 
The model is saved in the folder **`PATH_TO_SAVE\FOLD_NN`** and the background estimation as csv in **`PATH_TO_SAVE\FOLD_PRED`**.

### FOCuS
Now it's time for the trigger algorithm to shine. 
Starting from the the observed count rates and the estimated count rates by the NN, FOCuS computes the segments where the excess of count rates is significant more than `threshold` sigma. So the parameters are: 
- mu_min: multiplicative factor of the observed counts in relation to the integral of background values.
- t_max: limits the choice of the best interval.
- threshold: threshold parameter for the significance values exceed.

The catalog table will be stored in the folder **`FOLD_RES`** along with trigger data, significance and plots of the event's lightcurves.

### Localization
This part of the pipeline considers the detectors triggered for each event and localizes the event in the instant of peak energy using simple geometric reasoning.




