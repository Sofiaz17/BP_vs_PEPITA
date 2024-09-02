# Backpropagation implementation in C
`main.c`: file that executes training 
`get_data.c`: retrieves data from .csv file
`activation.c`: definition of activation functions

## How to compile
`gcc main.c activation.c get_data.c -o main -lm`

## Using Valgrind
Compile adding `-g` flag to compilation command.
Run executable with `valgrind --tool=massif --time-unit=B --stacks=yes --xtree-memory=full ./main`.
Visualize xtree map with KCachegrind with `KCachegrind &`, then open the xtree file.

## Getting MNIST dataset
Download train and test set in .csv format from [text](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)