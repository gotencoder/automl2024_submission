# Create a conda environment
conda create -n myenv python=3.9.1

#activate environment
conda activate myenv

#install packages
pip install -r requirments.txt

#Run the experiments
#AutoML_PC.py is the script that runs the experiments for the paper - see Table 2, 4, 5, 6.
#gat_ks.py, gcn_ks.py, graphsage_ks.py, deep_walk_ks.py are the scripts deep learning based models
#models.py is all of te deep learning models within a single script
#classical.py contains the code for the classical models of the paper
