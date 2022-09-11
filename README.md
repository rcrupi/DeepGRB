# DeepGRB

DeepGRB is a framework for serching astronomical events. Is composed by a background estimator, performed with a Neural Network, and on top applied an efficent trigger algorthm called FOCuS.

pipeline_bkg.py is the main script in which you can set the period of interest and:
1) download the data
2) preprocess the data
3) train the NN
4) perform FOCuS
5) localize and build the catalog (work in progress)

This work is based on:
- A data driven framework for searching long faint astronomical high-energy transients. Riccardo Crupi, Giuseppe Dilillo, Fabrizio Fiore, Andrea Vacchi. Work in progress
- Poisson-FOCuS: An efficient online method for detecting count bursts with application to gamma ray burst detection. Kester Ward, Giuseppe Dilillo, Idris Eckley, Paul Fearnhead. https://arxiv.org/abs/2208.01494

