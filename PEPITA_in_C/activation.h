#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// Sigmoid activation function
float sigmoid(float x);

// Derivative of Sigmoid activation function
float sigmoid_d(float x);

// ReLU activation function
float relu(float x);

// Derivative of ReLU activation function
float relu_d(float x);

// Derivative of tanh activation function
float tanh_d(float x);

#endif
