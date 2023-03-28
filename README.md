# DeepGRB

<img src="https://user-images.githubusercontent.com/93478548/189541951-83a118d4-0a6f-41f3-bc57-cbf0ab7623c2.png" width="750">

**DeepGRB** is a framework for serching astronomical events. Is composed by a background estimator, performed with a Neural Network, and on top applied an efficent trigger algorithm called FOCuS.

> ### Authors & contributors:
> **Riccardo Crupi**, **Giuseppe Dilillo**, Elisabetta Bissaldi, Fabrizio Fiore, Andrea Vacchi

To know more about this research work, please refer to:
- Searching for long faint astronomical high energy transients: a data driven approach. Riccardo Crupi, Giuseppe Dilillo, Elisabetta Bissaldi, Fabrizio Fiore, Andrea Vacchi. Work in progress
- Poisson-FOCuS: An efficient online method for detecting count bursts with application to gamma ray burst detection. Kester Ward, Giuseppe Dilillo, Idris Eckley, Paul Fearnhead. https://arxiv.org/abs/2208.01494

## Installation
Clone repo and install packages:
```
git clone https://github.com/rcrupi/DeepGRB.git
cd DeepGRB
pip install -r requirements.txt
```

Python version: `3.6.8`

### Configuration

In the config file `connections/utils/config.py` specify the path folder **`PATH_TO_SAVE`** in which you want to download the data and save the results. 

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



