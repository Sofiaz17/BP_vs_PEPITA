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
#define BATCH_SIZE 64
#define EPOCHS 2

int neu_per_lay[] = {NUM_NEU_L1, NUM_NEU_L2, NUM_NEU_L3};

const float mom_gamma = 0.9;
int batch_index = 0;
int exp_number = 0;

void matrix_multiply(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
// Initialize the result matrix C to zero
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i * B_cols + j] = 0.0;
        }
    }

    // Perform the matrix multiplication
    for (int i = 0; i < A_rows; i++) {
        for (int k = 0; k < A_cols; k++) {
            for (int j = 0; j < B_cols; j++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

void matrix_transpose(float* A, float* B, int rows, int cols) {
    // A is rows x cols
    // B will be cols x rows (transpose of A)

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void matrix_subtract(float* A, float* B, float* C, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i * cols + j] = -A[i * cols + j] + B[i * cols + j];
        }
    }
}

int find_max(int array[], int size) {
    if (size <= 0) {
        // Handle the case where the array size is non-positive
        fprintf(stderr, "Array size must be greater than 0\n");
        return 1;
    }

    int max_value = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
        }
    }
    return max_value;
}

typedef struct NeuralNet{
    int n_layers;
    int* n_neurons_per_layer;
    float*** w;
    float** b;
    float*** momentum_w;
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

    printf("Initializing net...\n");

    //weight initialization with "he" method
    for(i=0;i<nn->n_layers-1;i++){
        float nin = nn->n_neurons_per_layer[i];
        float stddev = sqrtf(6.0f / nin);

        for(j=0;j<nn->n_neurons_per_layer[i+1];j++){
            for(k=0;k<nn->n_neurons_per_layer[i];k++){  
                nn->w[i][k][j] = ((float) rand() / RAND_MAX) * (stddev - (-stddev)) - stddev;
                nn->momentum_w[i][k][j] = 0.0;
            }
            nn->b[i][j] = 0.0;
        }
    }
    printf("net initialized\n");
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
    //space for bias matrix (first dimension)->layer
    nn->b = (float**)malloc((nn->n_layers-1)*sizeof(float*));
    

    for(int i=0;i<nn->n_layers-1;i++){
        //weight matrix and momentum (second dimension)->neurons of curr layer
        nn->w[i] = (float**)malloc((nn->n_neurons_per_layer[i])*sizeof(float*));  
        nn->momentum_w[i] = (float**)malloc((nn->n_neurons_per_layer[i])*sizeof(float*));
        //bias matrix (second dimension)->neurons of curr layer
        nn->b[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float)); 
        
        //space for weight matrix and weight momentum (third dimension)->neuron of next layer
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
    }
    free(nn->w);
    free(nn->momentum_w);
    free(nn->b);
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
void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss, int epoch, int batch_count){
    // printf("in forward prop\n");
    //initialize input to actv for every layer
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){ 
            nn->actv_in[i][j] = 0.0;
        }
    }

    for(int i=1;i<nn->n_layers;i++){
        // Compute the weighted sum -> add bias to every input
        // for(int j=0;j<nn->n_neurons_per_layer[i];j++){
        //     nn->actv_in[i][j] += 1.0 * nn->b[0][j];
        // }
        //add previous weighted output
        for(int k=0;k<nn->n_neurons_per_layer[i-1];k++){
            for(int j=0;j<nn->n_neurons_per_layer[i];j++){  
                nn->actv_in[i][j] += nn->actv_out[i-1][k] * nn->w[i-1][k][j];  
            }
        }
        //activation for output layer
        if(i == nn->n_layers-1){
            if(strcmp(loss, "mse") == 0){
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                    nn->actv_out[i][j] = sigmoid(nn->actv_in[i][j]);
                }
            }
            else if(strcmp(loss, "ce") == 0){  
                float max_input_to_softmax = (float)INT_MIN;
                    for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                        if(nn->actv_in[i][j] > max_input_to_softmax){
                            max_input_to_softmax = nn->actv_in[i][j];
                        }
                    }
                    float deno = 0.0;
                    for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                        nn->actv_in[i][j] -= max_input_to_softmax;
                        deno += exp(nn->actv_in[i][j]);
                    }
                    for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                        nn->actv_out[i][j] = exp(nn->actv_in[i][j]) / deno;
                    }
                    for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                        nn->actv_in[i][j] += max_input_to_softmax;
                    }
            }
        } //activation for hidden layers
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
        //mean standard error
        if(strcmp(loss, "mse") == 0){
            running_loss += (nn->actv_out[last_layer][i] - nn->targets[i]) * (nn->actv_out[last_layer][i] - nn->targets[i]);
        }
        //cross-entropy loss
        else if(strcmp(loss, "ce") == 0){
            running_loss -= nn->targets[i]*(log(nn->actv_out[last_layer][i] ));
        }
	}
    if(strcmp(loss, "mse") == 0){
        running_loss /= BATCH_SIZE;
    }
    return running_loss;
}

//function to shuffle array
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
void model_train(struct NeuralNet* nn, float** X_train, float** y_train, float* y_train_num, float** X_test, float** y_test, float* y_test_num,
                 char* activation_fun, char* loss_fun, char* opt, float learning_rate){
    printf("in model train\n");
    
    float test_accs[EPOCHS];
    float train_losses[EPOCHS];
    float train_accs[EPOCHS];
    float curr_loss = 0.0;

    //matrix to modulate input
    float B_T[N_CLASSES*N_DIMS];
    //"he" initilization for B_T
    int nin = 28*28;
    float sd = sqrtf(6.0f/(float)nin);

    for(int i=0;i<N_CLASSES;i++){
        for(int j=0;j<N_DIMS;j++){
            float rand_num = ((float)rand() / RAND_MAX);
            //B_T[i * N_DIMS + j] = 1.0;
            B_T[i * N_CLASSES + j] = (rand_num * 2 * sd - sd) * 0.05;
        }
    }
    int max_num_neu = find_max(neu_per_lay, NUM_LAYERS);

    float outputs[BATCH_SIZE*N_CLASSES];
    float targets[BATCH_SIZE*N_CLASSES];
    float inputs[BATCH_SIZE*N_DIMS];
    float** layers_act = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        layers_act[i] = (float*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(float));
    }
    float error[BATCH_SIZE*N_CLASSES];
    float error_input[BATCH_SIZE*N_DIMS];
    float mod_inputs[BATCH_SIZE*N_DIMS];
    float mod_outputs[BATCH_SIZE*N_CLASSES];
    float** mod_layers_act = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        mod_layers_act[i] = (float*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(float));
    }
    float mod_error[BATCH_SIZE*N_CLASSES];
    float** delta_w_all = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int l=0; l<NUM_LAYERS-1;l++){
        delta_w_all[l] = (float*)malloc(nn->n_neurons_per_layer[l+1]*nn->n_neurons_per_layer[l]*sizeof(float));
    }
    float mod_error_T[N_CLASSES*BATCH_SIZE];
    float** delta_lay_act = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        delta_lay_act[i] = (float*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(float));
    }
    float** delta_lay_act_T = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        delta_lay_act_T[i] = (float*)malloc(nn->n_neurons_per_layer[i+1]*BATCH_SIZE*sizeof(float));
    }
    printf("memory allocated\n");
    
  
    for(int epoch=0;epoch<EPOCHS;epoch++){
        float avg_loss = 0.0;
        
        //scheduler
        if(epoch == 50){
            learning_rate *= 0.1;
        }

        //shuffle input
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

        //loop over batches
        for(int batch_num=0;batch_num<floor(N_SAMPLES/BATCH_SIZE);batch_num++){
            //assegna batch
            printf("[%d]TRAIN BATCH %d\n", epoch, batch_num);
            
            //loop over elements in batches
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                //TO DO: do masks
            
                //assign input sample
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){
                    nn->actv_out[0][j] = X_train[shuffler_train[in_cnt_train]][j];
                }
                //assign target of input sample
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){      
                    nn->targets[j] = y_train[shuffler_train[in_cnt_train]][j];
                }
                //save all batch inputs
                for(int in_neu=0;in_neu<N_DIMS;in_neu++){
                    inputs[batch_elem*N_DIMS+in_neu] = nn->actv_out[0][in_neu];
                }

                forward_propagation(nn, activation_fun, loss_fun, epoch, batch_count);
                running_loss += calc_loss(nn, loss_fun);

                //compute and store error for every sample
                for(int out_neu=0;out_neu<N_CLASSES;out_neu++){
                    error[batch_elem*N_CLASSES+out_neu] = nn->actv_out[nn->n_layers-1][out_neu] - nn->targets[out_neu];
                            // outputs[batch_elem*N_CLASSES+out_neu] = nn->actv_out[nn->n_layers-1][out_neu]; 
                            // targets[batch_elem*N_CLASSES+out_neu] = nn->targets[out_neu];
                }

                //check train accuracy
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){
                    //find maximum output value
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){
                        max_val = nn->actv_out[nn->n_layers-1][j];
                        idx = j;  
                    }
                }
                             
                if(idx == (int)y_train_num[shuffler_train[in_cnt_train]]){
                    correct++;
                }

                //save activations
                for(int lay=0;lay<NUM_LAYERS-1;lay++){
                    for(int neu=0;neu<nn->n_neurons_per_layer[lay+1];neu++){
                        layers_act[lay][batch_elem*nn->n_neurons_per_layer[lay+1]+neu] = nn->actv_out[lay+1][neu];
                    }
                }
                ++in_cnt_train;
            }
            total += BATCH_SIZE;
            avg_loss =  running_loss/BATCH_SIZE;
      
            matrix_multiply(error, B_T, error_input, BATCH_SIZE, N_CLASSES, N_DIMS);

            //perturb input
            for(int i = 0; i < BATCH_SIZE; i++) {
                for(int j = 0; j < N_DIMS; j++) {
                    mod_inputs[i*N_DIMS+j] = inputs[i*N_DIMS+j] + error_input[i*N_DIMS+j];
                }
            }

            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                //TO DO: do masks
                max_val = (float)INT_MIN;
                idx = -1;

                //assign modulated input
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){        
                    nn->actv_out[0][j] = mod_inputs[batch_elem*nn->n_neurons_per_layer[0] + j];
                }
                forward_propagation(nn, activation_fun, loss_fun, epoch, batch_count);

                //compute modulated error for each sample
                for(int out_neu=0;out_neu<N_CLASSES;out_neu++){
                    mod_error[batch_elem*N_CLASSES+out_neu] = nn->actv_out[nn->n_layers-1][out_neu] - nn->targets[out_neu];
                }
                if(idx == (int)y_train_num[shuffler_train[in_cnt_train_2]]){
                    correct++;
                }

                //compute modulated activations
                for(int lay=0;lay<NUM_LAYERS-1;lay++){
                    for(int neu=0;neu<nn->n_neurons_per_layer[lay+1];neu++){
                        mod_layers_act[lay][batch_elem*nn->n_neurons_per_layer[lay+1]+neu] = nn->actv_out[lay+1][neu];
                    }
                }
                ++in_cnt_train_2;
            }

            matrix_transpose(mod_error,mod_error_T,BATCH_SIZE,N_CLASSES);
           
            for(int l=0;l<NUM_LAYERS-1;l++){
                matrix_subtract(mod_layers_act[l], layers_act[l], delta_lay_act[l], BATCH_SIZE, nn->n_neurons_per_layer[l+1]);
                matrix_transpose(delta_lay_act[l], delta_lay_act_T[l], BATCH_SIZE, nn->n_neurons_per_layer[l+1]);
                
                //compute delta for last layer as in BP
                if(l == NUM_LAYERS-2){
                    if((NUM_LAYERS-1) > 1){
                        matrix_multiply(mod_error_T, mod_layers_act[l-1], delta_w_all[l], N_CLASSES, BATCH_SIZE, nn->n_neurons_per_layer[l]);
                    }
                    else {
                        matrix_multiply(mod_error_T, mod_inputs, delta_w_all[l], N_CLASSES, BATCH_SIZE, N_DIMS);
                    }
                } //compute delta in first layer 
                else if(l==0){
                    matrix_multiply(delta_lay_act_T[l], mod_inputs, delta_w_all[l], nn->n_neurons_per_layer[l+1], BATCH_SIZE, N_DIMS);
                } //compute delta in hidden layers
                else if(l>0 && l<NUM_LAYERS-2){
                    matrix_multiply(delta_lay_act_T[l], mod_layers_act[l-1], delta_w_all[l], nn->n_neurons_per_layer[l+1], BATCH_SIZE, nn->n_neurons_per_layer[l]);
                }
            }

            for(int k=0;k<nn->n_layers-1;k++){
                for(int i=0;i<nn->n_neurons_per_layer[k];i++){
                    for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){    

                        if(strcmp(opt, "sgd") == 0){
                            nn->w[k][i][j] -= learning_rate * (delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j]/BATCH_SIZE);
                        }
                        else if(strcmp(opt, "momentum") == 0){
                            nn->momentum_w[k][i][j] = mom_gamma * nn->momentum_w[k][i][j] + (delta_w_all[k][i*nn->n_neurons_per_layer[k+1]+j]/BATCH_SIZE) * learning_rate;
                            nn->w[k][i][j] += nn->momentum_w[k][i][j];
                        }
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

            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){

                max_val = (float)INT_MIN;
                idx = -1;
            
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){
                    nn->actv_out[0][j] = X_test[shuffler_test[in_cnt_test]][j]; 
                }
                
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){      
                    nn->targets[j] = y_test[shuffler_test[in_cnt_test]][j];
                }
                
                forward_propagation(nn, activation_fun, loss_fun, epoch, batch_count);
            
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){
                        max_val =nn->actv_out[nn->n_layers-1][j];
                        idx = j;
                    }
                }

                if(idx == (int)y_test_num[shuffler_test[in_cnt_test]]){
                    correct++;
                }
                ++in_cnt_test;
            }
            total += BATCH_SIZE;
            batch_count += 1;
        }
        printf("test total: %f \n",total);
        printf("test correct: %f\n",correct);
        printf("Test accuracy epoch [%d]: %f %%\n",epoch, 100 * correct / total);
        test_accs[epoch] = 100 * correct / total;
  
    }
    
    printf("FINISHED TRAINING\n");

    FILE *file = fopen("PEPITA_C_implem.txt", "w");
    printf("open file\n");

    for(int epoch=0;epoch<EPOCHS;epoch++){
        fprintf(file, "EPOCH %d\n", epoch);
        fprintf(file, "Train loss epoch [%d]: %lf\n", epoch, train_losses[epoch]);
        fprintf(file, "Train accuracy epoch [%d]: %lf\n", epoch, train_accs[epoch]);
        fprintf(file, "Test accuracy epoch [%d]: %lf\n", epoch, test_accs[epoch]);
    }
    
    fclose(file);
    printf("close file\n");

    printf("Freeing memory...\n");

    for(int i=0; i<NUM_LAYERS-1;i++){
        free(layers_act[i]);
    }
    free(layers_act);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(mod_layers_act[i]);
    }
    free(mod_layers_act);
   
    for(int l=0; l<NUM_LAYERS-1;l++){
        free(delta_w_all[l]);
    }
    free(delta_w_all);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(delta_lay_act[i]);
    }
    free(delta_lay_act);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(delta_lay_act_T[i]);
    }
    free(delta_lay_act_T);  
}


int main(){

    // Used for setting a random seed
    srand(time(NULL));

    // Create and initialize the neural network
    struct NeuralNet* nn = newNet();

    // Initialize the learning rate, optimizer, loss, and other hyper-parameters
    float learning_rate = 0.1;
    float init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "momentum";
   
    float** img_train;
    float** lbl_train;
    float* lbl_train_num;
    float** img_test;
    float** lbl_test;
    float* lbl_test_num;

    //structure for all train data
    img_train = (float**) malloc(N_SAMPLES*sizeof(float*));
    for(int i=0;i<N_SAMPLES;i++){
        img_train[i] = (float*)malloc(N_DIMS*sizeof(float));
    }
    //structure for train labels (one_hot)
    lbl_train = malloc(N_SAMPLES * sizeof(float*));
    for(int i=0;i<N_SAMPLES;i++){
        lbl_train[i] = malloc(N_CLASSES * sizeof(float));
    }
    //structure for train labels (number)
    lbl_train_num = malloc(N_SAMPLES*sizeof(float));
    read_csv_file(img_train, lbl_train_num, lbl_train, "train");
    scale_data(img_train, "train");

    //structure for all test data
    img_test = malloc(N_TEST_SAMPLES*sizeof(float*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        img_test[i] = malloc(N_DIMS*sizeof(float));
    }
    //structure for test labels (one_hot)
    lbl_test = malloc(N_TEST_SAMPLES * sizeof(float*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        lbl_test[i] = malloc(N_CLASSES * sizeof(float));
    }
    //structure for test labels (number)
    lbl_test_num = malloc(N_TEST_SAMPLES*sizeof(float));
    read_csv_file(img_test, lbl_test_num, lbl_test, "test");
    printf("heading to scale_data\n");
    scale_data(img_test, "test");


    model_train(nn,img_train,lbl_train,lbl_train_num,img_test,lbl_test,lbl_test_num,activation_fun,loss,opt,learning_rate);


    // Free the dynamically allocated memory
    free_NN(nn);
   
    for(int i=0;i<N_SAMPLES;i++){
        free(img_train[i]);
        free(lbl_train[i]);
    }
    free(img_train);
    free(lbl_train);
    free(lbl_train_num);
    for(int i=0;i<N_TEST_SAMPLES;i++){
        free(img_test[i]);
        free(lbl_test[i]);
    }
    free(img_test);
    free(lbl_test);
    free(lbl_test_num);
  
    return 0;
}