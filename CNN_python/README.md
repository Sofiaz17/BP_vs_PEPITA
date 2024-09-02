# Pytorch code for MNIST, CIFAR10 and CIFAR100 
Code to run from terminal with command 
`python main_pytorch.py --exp_name Experiment2 --learn_type ERIN --n_runs 1 --train_epochs 100 --eta 0.01 --dropout 0.9 --Bstd 0.05 --eta_decay --dataset mn --batch_size 50 --update_type mom --w_init he_uniform --model Net1conv1fcXL_cif --is-pool`

Follow instructions on the code on measures to be implemented according to the dataset used. 

In `models.py` CNN network models are defined, both for cif and mn.
