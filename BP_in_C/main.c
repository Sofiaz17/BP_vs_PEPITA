#include "activation.h"
#include "get_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define NUM_LAYERS 3
#define NUM_NEU_L1 784
#define NUM_NEU_L2 128
#define NUM_NEU_L3 10
#define BATCH_SIZE 128
#define EPOCHS 100


int neu_per_lay[] = {NUM_NEU_L1, NUM_NEU_L2, NUM_NEU_L3};
float deltas[2];

const float mom_gamma = 0.9;
int batch_index = 0;


typedef struct NeuralNet{
    int n_layers;
    int* n_neurons_per_layer;
    float*** w;
    float** b;
    float*** momentum_w;
    float** momentum_b;
    float** error;
    float** actv_in;
    float** actv_out;
    float* targets;
} __attribute__((packed))NeuralNet;

void initialize_net(NeuralNet* nn){
    int i,j,k;

    if(nn->n_layers == 0){
        printf("No layers in Neural Network...\n");
        return;
    }

    printf("Initializing weights...\n");

    for(i=0;i<nn->n_layers-1;i++){
        float nin = nn->n_neurons_per_layer[i];
        float stddev = sqrtf(6.0f / nin);

        for(j=0;j<nn->n_neurons_per_layer[i];j++){
            for(k=0;k<nn->n_neurons_per_layer[i+1];k++){
                float scale = rand() / (float) RAND_MAX;
                nn->w[i][j][k] = ((float) rand() / RAND_MAX) * (stddev - (-stddev)) - stddev;
                nn->momentum_w[i][j][k] = 0.0;
                nn->error[i][k] = 0.0;
                nn->b[i][k] = 0.0;
                nn->momentum_b[i][k] = 0.0;
            }
        }
    }
    printf("weights initialized\n");
}

void free_NN(NeuralNet* nn);


// Function to create a neural network and allocate memory
NeuralNet* newNet(){
    printf("in newNet\n");
    int i,j;
    //space for nn
    NeuralNet* nn = malloc(sizeof(struct NeuralNet));
    nn->n_layers = NUM_LAYERS;
    //space for layers
    nn->n_neurons_per_layer = (int*)malloc(nn->n_layers * sizeof(int));

    //initialize layer with num neurons
    for(i=0; i<nn->n_layers; i++){
        nn->n_neurons_per_layer[i] = neu_per_lay[i];
    }

    //space for weight matrix and weight momentum (first dimension)->layer
    nn->w = (float***)malloc((nn->n_layers-1)*sizeof(float**));
    nn->momentum_w = (float***)malloc((nn->n_layers-1)*sizeof(float**));
    //space for bias matrix and bias momentum (first dimension)->layer
    nn->b = (float**)malloc((nn->n_layers-1)*sizeof(float*));
    nn->momentum_b = (float**)malloc((nn->n_layers-1)*sizeof(float*));
    

    for(int i=0;i<nn->n_layers-1;i++){
        //weight matrix and momentum (second dimension)->neurons of curr layer
        nn->w[i] = (float**)malloc((nn->n_neurons_per_layer[i])*sizeof(float*));
        nn->momentum_w[i] = (float**)malloc((nn->n_neurons_per_layer[i])*sizeof(float*));
        //bias matrix and mometum (second dimension)->neurons of curr layer
    
        nn->b[i] = (float*)malloc((nn->n_neurons_per_layer[i+1])*sizeof(float));
        nn->momentum_b[i] = (float*)malloc((nn->n_neurons_per_layer[i+1])*sizeof(float));

        //space for weight matrix and weight momentum (third dimension)->neurond of next layer
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){
            nn->w[i][j] = malloc((nn->n_neurons_per_layer[i+1])*sizeof(float));
            nn->momentum_w[i][j] = malloc((nn->n_neurons_per_layer[i+1])*sizeof(float));
        }
    }
    
    //space for error matrix for each neuron in each layer(layer dimension) 
    nn->error = (float**)malloc((nn->n_layers)*sizeof(float*));
    //space for input and output to activation functions (layer dimension)
    nn->actv_in = (float**)malloc((nn->n_layers)*sizeof(float*));
    nn->actv_out = (float**)malloc((nn->n_layers)*sizeof(float*));
    
    for(int i=0;i<nn->n_layers;i++){
        //space for error matrix for each neuron in each layer(neuron dimension) 
        nn->error[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float));
        //space for input and output to activation functions (neuron dimension)
        nn->actv_in[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float));
        nn->actv_out[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float));
    }
    //space for desired outputs (one hot vector)
    nn->targets = malloc((nn->n_neurons_per_layer[nn->n_layers-1])*sizeof(float));
    
    // Initialize the weights
    initialize_net(nn);

    printf("end newNet\n");
    return nn;
}

// Function to free the dynamically allocated memory
void free_NN(struct NeuralNet* nn){
    printf("in free_NN\n");
    if(!nn) return;
    for(int i=0;i<nn->n_layers-1;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){
            free(nn->w[i][j]);
            free(nn->momentum_w[i][j]);
        }
        free(nn->w[i]);
        free(nn->momentum_w[i]);
        free(nn->b[i]);
        free(nn->momentum_b[i]);
    }
    free(nn->w);
    free(nn->momentum_w);
    free(nn->b);
    free(nn->momentum_b);
    for(int i=0;i<nn->n_layers;i++){
        free(nn->actv_in[i]);
        free(nn->actv_out[i]);
        free(nn->error[i]);
    }
    free(nn->actv_in);
    free(nn->actv_out);
    free(nn->error);
    free(nn->targets);
    free(nn->n_neurons_per_layer);
    free(nn);
}

// Function for forward propagation step
void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss){
    // printf("in forward prop\n");
    //initialize input to actv for every layer
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){
            nn->actv_in[i][j] = 0.0;
        }
    }
    for(int i=1;i<nn->n_layers;i++){
        // Compute the weighted sum -> add bias to every input
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){
            nn->actv_in[i][j] += 1.0 * nn->b[i-1][j];
            //printf("actv_in + bias\n");
        }

        //add previous weighted output
        for(int k=0;k<nn->n_neurons_per_layer[i-1];k++){
            for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                nn->actv_in[i][j] += nn->actv_out[i-1][k] * nn->w[i-1][k][j];
            }
        }
        // Apply non-linear activation function to the weighted sums
        //if last layer, apply softmax
        if(i == nn->n_layers-1){
            if(strcmp(loss, "mse") == 0){
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                    nn->actv_out[i][j] = sigmoid(nn->actv_in[i][j]);
                }
            }
            else if(strcmp(loss, "ce") == 0){
                float max_input_to_softmax = (float)INT_MIN;
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                    if(fabs(nn->actv_in[i][j]) > max_input_to_softmax){
                        max_input_to_softmax = fabs(nn->actv_in[i][j]);
                    }
                }
                float deno = 0.0;
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                    nn->actv_in[i][j] -= max_input_to_softmax;
                    deno += exp(nn->actv_in[i][j]);
                }
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                    nn->actv_out[i][j] = (float)exp(nn->actv_in[i][j])/(float)deno;
                    
                }
            }
        } //if other layers, apply something else
        else{
            for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                if(strcmp(activation_fun, "sigmoid") == 0){
                    nn->actv_out[i][j] = sigmoid(nn->actv_in[i][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0){
                    nn->actv_out[i][j] = tanh(nn->actv_in[i][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0){
                    nn->actv_out[i][j] = relu(nn->actv_in[i][j]);
                }
                else{
                    nn->actv_out[i][j] = relu(nn->actv_in[i][j]);
                }
            }
        }
    }
}


// Function to calculate loss
float calc_loss(struct NeuralNet* nn, char* loss){
    // printf("in calc loss\n");
    int i;
    float running_loss = 0.0;
    int last_layer = nn->n_layers-1;
    float epsilon = 1e-10;
    for(i=0;i<nn->n_neurons_per_layer[last_layer];i++){
        if(strcmp(loss, "mse") == 0){
            running_loss += (0.5)*(nn->actv_out[last_layer][i] - nn->targets[i]) * (nn->actv_out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            running_loss -= nn->targets[i]*(log(nn->actv_out[last_layer][i]+epsilon));
        }
	}
    return running_loss;
}


// Function for back propagation step
void back_propagation(struct NeuralNet* nn, char* activation_fun, float learning_rate, char* loss, char* opt){
    // printf("in backprop\n");
    int last_layer = nn->n_layers-1;
    // Calculate the error in the output layer
    for(int i=0;i<nn->n_neurons_per_layer[last_layer];i++){
        if(strcmp(loss, "mse") == 0){
            float grad = sigmoid_d(nn->actv_out[last_layer][i]);
            nn->error[last_layer][i] = grad*(nn->actv_out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            nn->error[last_layer][i] = nn->actv_out[last_layer][i] - nn->targets[i];
        }
    }
    // Backpropagate the error from the last layer to the first layer
    for(int k=nn->n_layers-2;k>0;k--){
        
        for(int i=0;i<nn->n_neurons_per_layer[k];i++){     
            float sum = 0.0;
             for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){
                sum += nn->b[k][j] * nn->error[k+1][j];
                sum += nn->w[k][i][j] * nn->error[k+1][j];
            }
            float grad;
            if(strcmp(activation_fun, "sigmoid") == 0){
                grad = sigmoid_d(nn->actv_out[k][i]);
            }
            else if(strcmp(activation_fun, "tanh") == 0){
                grad = tanh_d(nn->actv_out[k][i]);
            }
            else if(strcmp(activation_fun, "relu") == 0){
                grad = relu_d(nn->actv_out[k][i]);
            }
            else{
                grad = sigmoid_d(nn->actv_out[k][i]);
            }
            nn->error[k][i] = grad * sum;
        }
    }
}


void shuffle(int* arr, size_t n) {
    if (n > 1) {

        for (size_t i = n - 1; i > 0; i--) {
            // Generate a random index j such that 0 <= j <= i
            size_t j = rand() % (i + 1);

            // Swap arr[i] with arr[j]
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }
}

// Function to train the model
void model_train(struct NeuralNet* nn, float** X_train, float** y_train, float* y_train_temp, float** X_test, float** y_test, float* y_test_temp,
                    char* activation_fun, char* loss_fun, char* opt, float learning_rate){
    printf("in model train\n");

    float test_accs[EPOCHS];
    float train_losses[EPOCHS];
    float train_accs[EPOCHS];
    float curr_loss = 0.0;

    float** delta_w_all = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int l=0; l<NUM_LAYERS-1;l++){
        delta_w_all[l] = (float*)malloc(nn->n_neurons_per_layer[l+1]*nn->n_neurons_per_layer[l]*sizeof(float));
    }
    float** delta_b_all = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int l=0; l<NUM_LAYERS-1;l++){
        delta_b_all[l] = (float*)malloc(nn->n_neurons_per_layer[l]*sizeof(float));
    }

    for(int epoch=0;epoch<EPOCHS;epoch++){
        float avg_loss = 0.0;

        if(epoch == 50){
            learning_rate *= 0.1;
        }
        int shuffler_train[N_SAMPLES];
        for(int i=0;i<N_SAMPLES;i++){
            shuffler_train[i] = i;
        }

        float running_loss = 0.0;
        int batch_count = 0;
        float total = 0.0;
        float correct = 0.0;
        
        int idx = -1;
        float max_val = (float)INT_MIN;
        int in_cnt_train = 0;
        int in_cnt_train_2 = 0;
        

        for(int batch_num=0;batch_num<floor(N_SAMPLES/BATCH_SIZE);batch_num++){
     
            //printf("[%d]TRAIN BATCH %d\n", epoch, batch_num);

            for(int k=0;k<nn->n_layers-1;k++){
                for(int i=0;i<nn->n_neurons_per_layer[k];i++){
                    for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){
                        delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j] = 0.0;
                    }
                    delta_b_all[k][i] = 0.0;
                }
            }

            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
             
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){
                    nn->actv_out[0][j] = X_train[shuffler_train[in_cnt_train]][j];  
                }
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){
                    nn->targets[j] = y_train[shuffler_train[in_cnt_train]][j];        //assign target labels (one hot)
                }
                forward_propagation(nn, activation_fun, loss_fun);
                back_propagation(nn, activation_fun, learning_rate, loss_fun, opt);
                running_loss += calc_loss(nn, loss_fun);

                for(int k=0;k<nn->n_layers-1;k++){
                    for(int i=0;i<nn->n_neurons_per_layer[k];i++){
                        for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){
                            delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j] += nn->error[k+1][j] * nn->actv_out[k][i];   
                        }
                    }
                }
                for(int k=0;k<nn->n_layers-1;k++){
                    for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){  
                        delta_b_all[k][j] += nn->error[k+1][j] * 1.0;
                    }
                }
              

                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){      //trova neurone con output maggiore
                        max_val = nn->actv_out[nn->n_layers-1][j];
                        idx = j;
                    }
                }
                if(idx == (int)y_train_temp[shuffler_train[in_cnt_train]]){   //checks train prediction
                    correct++;
                }
                ++in_cnt_train;

            }

            for(int k=0;k<nn->n_layers-1;k++){
                for(int i=0;i<nn->n_neurons_per_layer[k];i++){
                    for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){
                        delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j] /= BATCH_SIZE;
                    }
                }
            }
            for(int k=0;k<nn->n_layers-1;k++){
                for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){
                    delta_b_all[k][j] /= BATCH_SIZE;
                }
            }
        
            total += BATCH_SIZE;
            avg_loss = running_loss/BATCH_SIZE;
            
            for(int k=0;k<nn->n_layers-1;k++){
                for(int i=0;i<nn->n_neurons_per_layer[k];i++){
                    for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){

                        if(strcmp(opt, "sgd") == 0){
                            nn->w[k][i][j] -= learning_rate * delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j];
                        }
                        else if(strcmp(opt, "momentum") == 0){
                            nn->momentum_w[k][i][j] = mom_gamma * nn->momentum_w[k][i][j] + (1.0-mom_gamma) * delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j] * learning_rate;
                            nn->w[k][i][j] -= nn->momentum_w[k][i][j];
                        }
                        else if(strcmp(opt, "rmsprop") == 0){
                            nn->momentum_w[k][i][j] = mom_gamma * nn->momentum_w[k][i][j] + (1.0-mom_gamma) * delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j] * delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j];
                            nn->w[k][i][j] -= (learning_rate * delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j])/(sqrt(nn->momentum_w[k][i][j]) + 1e-6);
                        }
                   
                    }
                }
                // Update the bias weights
                for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){

                    if(strcmp(opt, "sgd") == 0){
                        nn->b[k][j] -= learning_rate * delta_b_all[k][j];
                    }
                    else if(strcmp(opt, "momentum") == 0){
                        nn->momentum_b[k][j] = mom_gamma * nn->momentum_b[k][j] + (1.0-mom_gamma) * delta_b_all[k][j] * learning_rate;
                        nn->b[k][j] -= nn->momentum_b[k][j];
                    }
                    else if(strcmp(opt, "rmsprop") == 0){
                        nn->momentum_b[k][j] = mom_gamma * nn->momentum_b[k][j] + (1.0-mom_gamma) * delta_b_all[k][j] * delta_b_all[k][j];
                        nn->b[k][j] -= (learning_rate * delta_b_all[k][j])/(sqrt(nn->momentum_b[k][j]) + 1e-6);
                    }
                }
            }
            batch_count += 1;
        }

        curr_loss = running_loss / (float)batch_count;
        printf("[%d, %5d] loss: %.3f\n", epoch, batch_count, curr_loss);
        train_losses[epoch] = curr_loss;
        printf("Train correct: %.2f\n", correct);
        printf("Train total: %.2f\n", total);
        printf("Train accuracy epoch [%d]: %.4f %%\n", epoch, 100 * correct / total);
        train_accs[epoch] = 100 * correct / total;

        printf("TESTING...\n");
        correct = 0.0;
        total = 0.0;
        batch_count = 0;

        int shuffler_test[N_TEST_SAMPLES];
        for(int i=0;i<N_TEST_SAMPLES;i++){
            shuffler_test[i] = i;
        }

        shuffle(shuffler_test, N_TEST_SAMPLES);

        int in_cnt_test = 0;

        for(int batch_num=0;batch_num<floor(N_TEST_SAMPLES/BATCH_SIZE);batch_num++){
        
            // printf("[%d]TEST BATCH %d\n", epoch,batch_num);

            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){

                max_val = (float)INT_MIN;
                idx = -1;
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){ 
                    nn->actv_out[0][j] = X_test[shuffler_test[in_cnt_test]][j]; 
                }
                
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){
                    nn->targets[j] = y_test[shuffler_test[in_cnt_test]][j];        //assign target labels (one hot)
                }
                
                forward_propagation(nn, activation_fun, loss_fun);
            
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){
                        max_val =nn->actv_out[nn->n_layers-1][j];
                        idx = j;
                    }
                }
                
                if(idx == (int)y_test_temp[shuffler_test[in_cnt_test]]){
                    correct++;
                }
                ++in_cnt_test;
            }
            total += BATCH_SIZE;
            batch_count += 1;
        }
        printf("Test accuracy epoch [%d]: %f %%\n",epoch, 100 * correct / total);
        test_accs[epoch] = 100 * correct / total;
    }

    printf("FINISHED TRAINING\n");
    for(int l=0; l<NUM_LAYERS-1;l++){
        free(delta_w_all[l]);
    }
    free(delta_w_all);
    for(int l=0; l<NUM_LAYERS-1;l++){
        free(delta_b_all[l]);
    }
    free(delta_b_all);  
}

int main(){

    // Used for setting a random seed
    srand(time(NULL));
    int seed = rand();

    // Create and initialize the neural network
    struct NeuralNet* nn = newNet();
    //init_nn(nn);
 
    // Initialize the learning rate, optimizer, loss, and other hyper-parameters
    float learning_rate = 0.1;
    float init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "momentum";
    
    float** img_train;
    float** lbl_train;
    float* lbl_train_temp;
    float** img_test;
    float** lbl_test;
    float* lbl_test_temp;

   img_train = (float**) malloc(N_SAMPLES*sizeof(float*));
    for(int i=0;i<N_SAMPLES;i++){
        img_train[i] = (float*)malloc(N_DIMS*sizeof(float));
    }
    lbl_train = malloc(N_SAMPLES * sizeof(float*));
    for(int i=0;i<N_SAMPLES;i++){
        lbl_train[i] = malloc(N_CLASSES * sizeof(float));
    }
    lbl_train_temp = malloc(N_SAMPLES*sizeof(float));
    read_csv_file(img_train, lbl_train_temp, lbl_train, "train");
    scale_data(img_train, "train");

    img_test = malloc(N_TEST_SAMPLES*sizeof(float*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        img_test[i] = malloc(N_DIMS*sizeof(float));
    }
    lbl_test = malloc(N_TEST_SAMPLES * sizeof(float*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        lbl_test[i] = malloc(N_CLASSES * sizeof(float));
    }
    lbl_test_temp = malloc(N_TEST_SAMPLES*sizeof(float));
    read_csv_file(img_test, lbl_test_temp, lbl_test, "test");
    printf("heading to scale_data\n");
    scale_data(img_test, "test");

    model_train(nn,img_train,lbl_train,lbl_train_temp,img_test,lbl_test,lbl_test_temp,activation_fun,loss,opt,learning_rate);

    return 0;
}