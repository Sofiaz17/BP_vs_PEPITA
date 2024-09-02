#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// Sigmoid activation function
float sigmoid(float x){
    float ans = (float)1/(float)(1 + exp(-x));
    return ans;
}


// Derivative of Sigmoid activation function
float sigmoid_d(float x){
    return sigmoid(x)*(1-sigmoid(x));
}


// ReLU activation function
float relu(float x){
    if(x < 0.0){
        return 0.0;
    }
    return x;
}


// Derivative of ReLU activation function
float relu_d(float x){
    if(x < 0.0){
        return 0.0;
    }
    return 1.0;
}

// Derivative of tanh activation function
float tanh_d(float x){
    float res = 1.0 - tanh(x)*tanh(x);
    return res;
}