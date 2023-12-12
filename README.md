# DeepGRB

<img src="https://user-images.githubusercontent.com/93478548/189541951-83a118d4-0a6f-41f3-bc57-cbf0ab7623c2.png" width="750">

**DeepGRB** is a framework for serching astronomical events. Is composed by a background estimator, performed with a Neural Network, and on top applied an efficent trigger algorithm called FOCuS.

> ### Authors & contributors:
> **Riccardo Crupi**, **Giuseppe Dilillo**, Elisabetta Bissaldi, Kester Ward, Fabrizio Fiore, Andrea Vacchi

To know more about this research work, please refer to:
- **Searching for long faint astronomical high energy transients: a data driven approach**. Riccardo Crupi, Giuseppe Dilillo, Kester Ward, Elisabetta Bissaldi, Fabrizio Fiore, Andrea Vacchi. https://link.springer.com/article/10.1007/s10686-023-09915-7
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

In the config file `connections/utils/config.py` specify the path folder **`PATH_TO_SAVE`** in which you want to download the data and save the results. It is possible to provide the names of the other folders included within it.

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

> <img src=https://user-images.githubusercontent.com/93478548/228681065-9474304f-fb2b-4aa7-a923-97edecafa15e.PNG width="650">
> 
> The background estimation for the n6 detector, in the energy range 1, on three hours of data. The Fermi/GBM count rate observations are represented over time as a black line, whereas the neural network estimation is plotted as a red solid line. The lower panel shows the residuals between the two quantities, with a black solid line denoting the reference of null residual.


### FOCuS
Now it's time for the trigger algorithm to shine. 
Starting from the the observed count rates and the estimated count rates by the NN, FOCuS computes the segments where the excess of count rates is significant more than `threshold` sigma. So the parameters are: 
- mu_min: multiplicative factor of the observed counts in relation to the integral of background values.
- t_max: limits the choice of the best interval.
- threshold: threshold parameter for the significance values exceed.

The catalog table will be stored in the folder **`FOLD_RES`** along with trigger data, significance and plots of the event's lightcurves.

> <img src=https://user-images.githubusercontent.com/93478548/228660136-bf62b826-022b-4c1d-a0d4-4f7823ebe995.JPG width="500">
> 
> Example of a transient event. Photon counts from each triggered detector are plotted with step lines, across three energy bands spanning 28 − 50 keV, 50 − 300 keV and 300 − 500 keV, with a resolution of 4.096 s. The neural network’s prediction of background count rates is represented by solid lines. Different detectors are identified using different colors. A red shaded area limits FOCuS-Poisson’s best guess of the transient duration. Times are expressed in units of seconds according Fermi’s standard mission elapsed time (MET).


### Localization
This part of the pipeline considers the detectors triggered for each event and localizes the event in the instant of peak energy using simple geometric reasoning.

> <img src=https://user-images.githubusercontent.com/93478548/228660171-4377aba6-8f51-43c1-831f-5605195545a7.JPG width="500">
> 
> Estimate of the candidate event’s source localization over the celestial sphere at 2019-04-20 22:32:56 UTC.
> The plot is done thanks to the package **Fermi GBM Data Tools**.


> |trig_ids |start_times      |duration      | catalog_triggers|trig_dets                                |sigma_r0    |sigma_r1    |sigma_r2    |ra     |dec    |
> |---      |---              | ---          |---              |---                                      |---         |---         |---         |---    |---    |
> |  35 |  2019-04-19 09:55:40|  262.532260  |   GRB190419414  |       n0_r0 n0_r1 n0_r2 n1_r0 n1_r1 ... |   16.186566 | 38.933412 | 6.321937  | 116.0  | -46.0 |
> |  36 |2019-04-20 15:08:24  | 69.633413    |          NaN    |       n4_r1 n8_r1                       | 0.000000    | 9.749557  | 0.000000  | 293.0  | -41.0 |
> |  37 |2019-04-20 22:32:56  | 16.384357    |          NaN    |      n6_r0 n6_r1 n7_r0 n7_r1 n8_r0 ...  | 15.158135   | 8.400105  | 0.000000  | 192.0  |  38.0 |
> |  38 |2019-04-20 23:32:27  |  8.382987    | GRB190420981    |            n6_r1                        | 0.000000    | 3.620252  | 0.000000  | 246.0  | -67.0 |
> |  39 |2019-04-22 16:05:09  | 20.480450    | GRB190422670    |         na_r1 nb_r1                     | 0.000000    | 6.935727  | 0.000000  | 193.0  | -41.0 |
> | 40  |2019-04-22 18:58:31  |  4.096085    |          NaN    |     n0_r1 n2_r1 n2_r2 n9_r1             | 0.000000    | 5.668806  | 4.537611  | 134.0  | 11.0  |
> |  41 |2019-04-22 22:56:09  |183.299547    | GRB190422957    |      n6_r0 n6_r1 n7_r0 n7_r1 n8_r0 ...  | 8.826744    | 10.507945 |  0.000000 |  183.0 | -61.0 |
> | 42  |2019-04-28 00:16:26  | 61.441057    |          NaN    |   n0_r0 n0_r1 n1_r0 n1_r1 n2_r1 ...     | 16.546931   | 21.319538 | 0.000000  | 51.0   | 55.0  |
> 
> A portion example of a catalog table.
