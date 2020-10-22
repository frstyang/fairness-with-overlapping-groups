For some (dataset) in {communities, adult, german, lawschool}, run 

python3 train_script.py --expfile ind_exp --dataset (dataset) --expname (dataset)_ind --ntrials 3

to run the independent group fairness experiment comparing WERM, Plugin, and Regularizer.
To run equal opportunity experiments, replace ind_exp with eo_exp, and (dataset)_ind with (dataset)_eo.
