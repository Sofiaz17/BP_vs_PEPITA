#ifndef __GET_DATA_H__
#define __GET_DATA_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define N_SAMPLES 60000
#define N_DIMS 784
#define N_CLASSES 10
#define N_TEST_SAMPLES 10000
#define N_TEST_SAMPLES_1 13
#define N_SAMPLES_1 202



// Function to read training and test data and store them appropriately
void read_csv_file(float** data, float* y_temp, float** y, char* dataset);

// Function to scale the dataset
void scale_data(float** data, char* dataset);

// Function to normalize the dataset
void normalize_data(float** X_train, float** X_test);

#endif
