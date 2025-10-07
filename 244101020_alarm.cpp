// Alarm.cpp : Defines the entry point for the console application.
//training the models

//0->set , 1-> snooze , 2-> stop


#include "stdafx.h"
#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include <time.h>
#include <iostream>
using namespace std;
#include <fstream>
#include<sstream>
#include <Windows.h>
#pragma comment(lib, "winmm.lib")
#define THRESHOLD 0.001
#define EPSILON 0.03
#define K 32
#define ll double
#define NUM_FILES 30
#define M_PI 3.14159265358979323846
#define NUM_TYPES 3
#define P 12
#define N 320
#define NUM_FRAMES 150
#define M 21500

#define S 5   // states
#define OS 32 // observation symbols
#define T_MAX 200
long double alpha[T_MAX][S]; // Forward probabilities
long double beta[T_MAX][S];  // Backward probabilities
long double gamma[T_MAX][S]; // State probabilities
long double xi[T_MAX][S][S]; // Joint probabilities

// Declare matrices with higher precision
long double A[S][OS];  // State transition matrix
long double B[S][OS];  // Observation probability matrix
long double Pi[S];    // Initial state probabilities

short int waveIn[16025 * 3];


// #define TOKHURA_WEIGHTS {1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0}
ll universe[M][P];
int observation_sequence[NUM_TYPES][NUM_FILES][NUM_FRAMES];
int observation_sequence_test[NUM_TYPES][10][NUM_FRAMES];

long double test_prob[10][10][10];

double tokhura_weights[P] = {1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};


int obs[NUM_FRAMES]; // Observation sequence for a file
long double A_sum[S][S] = {0}, B_sum[S][OS] = {0}, Pi_sum[S] = {0};
long double avgA[S][S], avgB[S][OS], avgPi[OS];



// Function to compute Tokhura distance between two vectors
double tokhura_distance(double *vec2, double *vec1)
{
    double distance = 0.0;
    for (int i = 0; i < P; i++)
    {
        distance += tokhura_weights[i] * (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return distance;
}

// Function to initialize the codebook with the centroid of the universe
void initialize_codebook_with_centroid(double **codebook, double universe[M][P])
{
    for (int j = 0; j < P; j++)
    {
        double sum = 0.0;
        for (int i = 0; i < M; i++)
        {
            sum += universe[i][j];
        }
        codebook[0][j] = sum / M;
    }
}

// Function to split the codebook by epsilon
void split_codebook(double **new_codebook, double **old_codebook, int current_size)
{
    for (int i = 0; i < current_size; i++)
    {
        for (int j = 0; j < P; j++)
        {
            new_codebook[2 * i][j] = old_codebook[i][j] * (1 + EPSILON);
            new_codebook[2 * i + 1][j] = old_codebook[i][j] * (1 - EPSILON);
        }
    }
}

// Function to assign each vector in the universe to the nearest codebook vector
void assign_to_regions(double universe[M][P], double **codebook, int region[M], int current_size)
{
    for (int i = 0; i < M; i++)
    {
        double min_distance = tokhura_distance(universe[i], codebook[0]);
        region[i] = 0;
        for (int j = 1; j < current_size; j++)
        {
            double distance = tokhura_distance(universe[i], codebook[j]);
            if (distance < min_distance)
            {
                min_distance = distance;
                region[i] = j;
            }
        }
    }
}

// Function to calculate the total distortion
double calculate_distortion(double universe[M][P], double **codebook, int region[M], int current_size)
{
    double total_distortion = 0.0;
    for (int i = 0; i < M; i++)
    {
        total_distortion += tokhura_distance(universe[i], codebook[region[i]]);
    }
    return total_distortion / M;
}

// Function to update centroids of the regions
void update_centroids(double universe[M][P], double **codebook, int region[M], int current_size)
{
    int count[K] = {0};
    double new_centroids[K][P] = {0};

    for (int i = 0; i < M; i++)
    {
        int r = region[i];
        for (int j = 0; j < P; j++)
        {
            new_centroids[r][j] += universe[i][j];
        }
        count[r]++;
    }

    for (int i = 0; i < current_size; i++)
    {
        if (count[i] == 0)
            continue;
        for (int j = 0; j < P; j++)
        {
            codebook[i][j] = new_centroids[i][j] / count[i];
        }
    }
}

void normalization(ll *arr, ll count)
{
    ll maxi = 0;
    for (int i = 0; i < count; i++)
    {
        arr[i] = fabs(arr[i]);
        if (arr[i] > maxi)
            maxi = arr[i];
    }

    if (maxi == 0)
    {
        printf("Warning: Maximum value in the array is 0. Normalization skipped.\n");
        return;
    }

    for (int i = 0; i < count; i++)
    {
        arr[i] *= (5000 / maxi);
    }
}

void Dc_Shift(ll *arr, ll count)
{
    ll sum = 0;
    for (int i = 0; i < count; i++)
    {
        sum += arr[i];
    }
    sum /= count;
    for (int i = 0; i < count; i++)
    {
        arr[i] -= sum;
    }
}

ll autocorrelation(ll input[], int k, int startind)
{
    ll sum = 0;
    int end = startind + N - k;
    for (int i = startind; i < end; i++)
    {
        sum += input[i] * input[i + k];
    }
    return sum;
}

void calculateErrorsAndAlphas(ll *R, ll *E, ll alpha[13][13], int p)
{
    E[0] = R[0];
    for (int i = 1; i <= p; i++)
    {
        ll summation = 0;
        for (int j = 1; j <= i - 1; j++)
        {
            summation += R[i - j] * alpha[i - 1][j];
        }
        ll k = (R[i] - summation) / E[i - 1];
        alpha[i][i] = k;
        for (int j = 1; j <= i - 1; j++)
        {
            alpha[i][j] = alpha[i - 1][j] - (k * alpha[i - 1][i - j]);
        }
        E[i] = (1 - (k * k)) * E[i - 1];
        if (fabs(E[i]) < 1e-10)
        {
            E[i] = 1e-10;
        }
    }
}

void calculateCepstralCoefficients(ll alpha[13][13], ll *cepstralCoefficients, ll R_0)
{
    cepstralCoefficients[0] = (R_0 > 0) ? log(R_0) : 0;
    for (int m = 1; m <= P; m++)
    {
        cepstralCoefficients[m] = alpha[P][m];
        for (int k = 1; k < m; k++)
        {
            cepstralCoefficients[m] += (k * cepstralCoefficients[k] * alpha[P][m - k]) / m;
        }
    }
}

void applyHammingWindowToFrame(ll *frame, int frame_size)
{
    for (int n = 0; n < frame_size; n++)
    {
        ll hammingValue = 0.54 - 0.46 * cos((2 * M_PI * n) / (frame_size - 1));
        frame[n] *= hammingValue;
    }
}

void raisedSineWindow(ll *cepstralCoefficients, int Q)
{
    for (int m = 1; m < Q; m++)
    {
        ll raisedValue = 1 + (Q - 1) * 0.5 * sin(M_PI * m / (Q - 1));
        cepstralCoefficients[m] *= raisedValue;
    }
}

void saveCiValues(ll ci[NUM_FRAMES][P + 1], char type, int file_num)
{
    char filename[50];
    sprintf(filename, "ci_test/244101020_%c_%d_ci.txt", type, file_num);

    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file %s for writing\n", filename);
        return;
    }

    for (int frame = 0; frame < NUM_FRAMES; frame++)
    {
        for (int i = 1; i <= P; i++)
        {
            fprintf(file, "%f ", ci[frame][i]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void generate_observation_sequences(double **codebook)
{
	int rowcount=0;
    for (int t = 0; t < NUM_TYPES; t++)
    {
		char filename[100];
		sprintf(filename, "observations_digit_%d.txt", t);
		std::ofstream file(filename);
        for (int f = 0; f < NUM_FILES; f++)
        {
            for (int n = 0; n < NUM_FRAMES; n++)
            {
                int observation = 0;
				double mini_distance = 99999;

                for (int k = 0; k < K; k++)
                {
                    double distance = tokhura_distance(universe[rowcount], codebook[k]);
                    if (distance < mini_distance)
                    {
                        mini_distance = distance;
                        observation = k;
                    }
                }
					rowcount++;


                observation_sequence[t][f][n] = observation;
				file << observation_sequence[t][f][n] << " "; 
            }
			file << "\n";
        }
		file.close();
    }
}

void generate_observation_sequences_test(double **codebook) 
{
	 char types[NUM_TYPES] = {'0', '1', '2'};
    // Loop through each digit
    for (int digit = 0; digit < NUM_TYPES; digit++) 
    {
        char output_filename[100];
        sprintf(output_filename, "ci_test/observations_digit_%c.txt", types[digit]);
        std::ofstream output_file(output_filename);

        // Loop through test files 31 to 40 for each digit
        for (int file_idx = 31; file_idx <= 40; file_idx++) 
        {
            // Construct the file path for each test file in the ci_test folder
            char test_filename[150];
			

           sprintf(test_filename, "ci_test/244101020_%c_%d_ci.txt", types[digit], file_idx);

           printf("Trying to open file: %s\n", test_filename);


            std::ifstream test_file(test_filename);
            if (!test_file.is_open()) {
                std::cerr << "Error opening file: " << test_filename << std::endl;
                continue;
            }

            // Loop through each frame (row of Ci values) in the test file
            int frame = 0;
            std::string line;
            while (std::getline(test_file, line) && frame < NUM_FRAMES) 
            {
                // Parse the Ci values for each frame
                std::istringstream line_stream(line);
                double ci_values[P];
                for (int p = 0; p < P; p++) {
                    line_stream >> ci_values[p];
                }

                // Find the nearest codebook vector for the current Ci values
                int observation = 0;
                double min_distance = 1e10; // Initialize with a high value

                for (int k = 0; k < K; k++) 
                {
                    double distance = tokhura_distance(ci_values, codebook[k]);
                    if (distance < min_distance) 
                    {
                        min_distance = distance;
                        observation = k;
                    }
                }

                // Store the observation in the sequence array and write to file
                observation_sequence_test[digit][file_idx - 31][frame] = observation;
				 output_file << observation<<" ";
				printf("%d ",observation);
                frame++;
            }
			printf("\n");
            output_file << "\n"; // Newline after each file's observations
            test_file.close();
        }

        output_file.close();
        //printf("Observation sequence generated for digit %d\n", digit);
    }
}


void read_hmm_data(){
	 FILE *file = fopen("A_matrix.txt", "r");
    if (file == NULL) {
        printf("Error opening file: \n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            if (fscanf(file, "%lf", &A[i][j]) != 1) {
                printf("Error reading matrix from file: \n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    fclose(file);

	 FILE *file1 = fopen("B_matrix.txt", "r");
    if (file1 == NULL) {
        printf("Error opening file: \n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < OS; j++) {
            if (fscanf(file1, "%lf", &B[i][j]) != 1) {
                printf("Error reading matrix from file: \n" );
                fclose(file1);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    fclose(file1);

	 FILE *file2 = fopen("Pi_matrix.txt", "r");
    if (file2 == NULL) {
        printf("Error opening file: \n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < S; i++) {
        if (fscanf(file2, "%lf", &Pi[i]) != 1) {
            printf("Error reading matrix from file: \n");
            fclose(file2);
            exit(EXIT_FAILURE);
        }
    }
    
    fclose(file2);
}

void forward(){
	for (int i = 0; i < S; ++i) {
            alpha[0][i] = Pi[i] * B[i][obs[0]];
        }

        for (int t = 1; t <NUM_FRAMES ; ++t) {
            for (int j = 0; j < S; ++j) {
                alpha[t][j] = 0;
                for (int i = 0; i < S; ++i) {
                    alpha[t][j] += alpha[t - 1][i] * A[i][j];
                }
                alpha[t][j] *= B[j][obs[t]];
            }
        }
}

void backward(){
	for (int i = 0; i < S; ++i) {
            beta[NUM_FRAMES - 1][i] = 1.0;
        }

        for (int t = NUM_FRAMES - 2; t >= 0; --t) {
            for (int i = 0; i < S; ++i) {
                beta[t][i] = 0;
                for (int j = 0; j < S; ++j) {
                    beta[t][i] += A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j];
                }
            }
        }
}

void avg_intilization(){
	for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
			avgA[i][j] =0;
        }
    }

	for (int i = 0; i < S; i++) {
        for (int j = 0; j < OS; j++) {
            avgB[i][j] =0;
        }
    }

	for (int i = 0; i < S; i++) {
        avgPi[i] =0;
    }

	for(int i=0; i<T_MAX; i++){
		for(int j=0; j<S; j++){
			alpha[i][j]=0;
			beta[i][j]=0;
			gamma[i][j]=0;
			for(int k=0; k<S; k++){
				xi[i][j][k]=0;
			}
		}
	}
}

void read_observation_hmm(int t, int f){
	for(int i=0; i<NUM_FRAMES; i++){
		obs[i] = observation_sequence[t][f][i];
	}
}

void store_final_model(int m){
	char file1[100];
    sprintf(file1, "digit_%d.txt", m);
    FILE* file = fopen(file1, "w");

	for (int i = 0; i < S; i++) {
		fprintf(file, "%e ", Pi[i]);
    }

    fprintf(file, "\n");  // Add a newline to separate matrices

	 for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            fprintf(file, "%e ", avgA[i][j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");  // Add a newline to separate matrices

	for (int i = 0; i < S; i++) {
        for (int j = 0; j < OS; j++) {
            fprintf(file, "%e ", avgB[i][j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");  // Add a newline to separate matrices

	fclose(file);
}


void implement_hmm3(int m){

	cout <<"avg_initli" << endl;
	avg_intilization();
	for(int i=0; i<NUM_FILES; i++){
		cout <<"read_observation_hmm" << endl;
		read_observation_hmm(m,i);
		cout <<"read_hmm" << endl;
		read_hmm_data();

		for(int x=0; x<1000; x++){
			forward();
			backward();
			// * Compute Gamma and Xi *
			for (int t = 0; t < NUM_FRAMES - 1; ++t) {
				long double denom = 0;
				for (int i = 0; i < S; ++i) {
					for (int j = 0; j < S; ++j) {
						denom += alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j];
					}
				}

				for (int i = 0; i < S; ++i) {
					gamma[t][i] = 0;
					for (int j = 0; j < S; ++j) {
						xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]) / denom;
						gamma[t][i] += xi[t][i][j];
					}
				}
			}

			// Special case for gamma at time T-1
			long double denom = 0;
			for (int i = 0; i < S; ++i) {
				denom += alpha[NUM_FRAMES - 1][i];
			}
			for (int i = 0; i < S; ++i) {
				gamma[NUM_FRAMES - 1][i] = alpha[NUM_FRAMES - 1][i] / denom;
			}

			
			for (int i = 0; i < S; ++i) {
				for (int j = 0; j < S; ++j) {
					long double numer = 0, denom = 0;
					for (int t = 0; t < NUM_FRAMES - 1; ++t) {
						numer += xi[t][i][j];
						denom += gamma[t][i];
					}
					A[i][j] = numer / denom;
				}
			}

			
			for (int i = 0; i < S; ++i) {
				int sum=0, maxindex =0;
				for (int k = 0; k < OS; ++k) {
					long double numer = 0, denom = 0;
					for (int t = 0; t < NUM_FRAMES; ++t) {
						if (obs[t] == k) {
							numer += gamma[t][i];
						}
						denom += gamma[t][i];
					}
					B[i][k] = numer / denom;
					if(B[i][k] <1e-15)  B[i][k] = 1e-15;
					sum+=B[i][k];
					if(B[i][k] > B[i][maxindex]) maxindex = k;
				}
				if(sum>1) B[i][maxindex] -= (1-sum);
			}

		}

		
		
		for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                avgA[i][j] +=A[i][j];
            }
            for (int m = 0; m < OS; m++) {
                avgB[i][m] += B[i][m];
            }
        }
	
	}
	cout <<"Avg"<<endl;
	for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            avgA[i][j] /= 30.0;
        }
        for (int m = 0; m < OS; m++) {
            avgB[i][m] /=30.0;
        }
    }
	store_final_model(m);
}

void read_hmm_model(int digit) 
{
    char filename[100];
    sprintf(filename, "digit_%d.txt", digit);

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read Pi (initial state probabilities, 1 line with 5 values)
    for (int i = 0; i < S; i++) {
        file >> Pi[i];
    }

    // Read A (state transition matrix, 5 lines with 5 values each)
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            file >> A[i][j];
        }
    }

    // Skip the blank line
    std::string blank_line;
    std::getline(file, blank_line); // Complete current line
    std::getline(file, blank_line); // Skip empty line

    // Read B (observation probability matrix, 5 lines with 32 values each)
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < OS; j++) {
            file >> B[i][j];
        }
    }

    file.close();
   // std::cout << "HMM model for digit " << digit << " loaded successfully.\n";
}



long double forward_algorithm(int obs_sequence[NUM_FRAMES]) 
{
    // Initialization
    for (int i = 0; i < S; i++) {
        alpha[0][i] = Pi[i] * B[i][obs_sequence[0]];
    }

    // Induction
    for (int t = 1; t < NUM_FRAMES; t++) {
        for (int j = 0; j < S; j++) {
            alpha[t][j] = 0;
            for (int i = 0; i < S; i++) {
                alpha[t][j] += alpha[t - 1][i] * A[i][j];
            }
            alpha[t][j] *= B[j][obs_sequence[t]];
        }
    }

    // Termination
    long double prob = 0;
    for (int i = 0; i < S; i++) {
        prob += alpha[NUM_FRAMES - 1][i];
    }
    return prob;
}

void test_hmm_on_observation_sequences() 
{
    char types[NUM_TYPES] = {'0', '1', '2'};
    int correct_predictions = 0; // Counter for correctly recognized digits

    for (int digit = 0; digit < NUM_TYPES; digit++) 
    {
        // Open the observation sequence file for the current digit
        char obs_filename[100];
        sprintf(obs_filename, "ci_test/observations_digit_%c.txt", types[digit]);
        std::ifstream obs_file(obs_filename);
        if (!obs_file.is_open()) {
            std::cerr << "Error opening observation file: " << obs_filename << std::endl;
            continue;
        }

        int obs_sequence[NUM_FRAMES];
        int file_idx = 31;

        // Process each observation sequence in the file
        std::string line;
        while (std::getline(obs_file, line)) 
        {
            std::istringstream line_stream(line);
            for (int t = 0; t < NUM_FRAMES; t++) {
                line_stream >> obs_sequence[t];
            }

            // Calculate probabilities for all digit models
            for (int other_digit = 0; other_digit < NUM_TYPES; other_digit++) {
                read_hmm_model(other_digit);
                long double prob = forward_algorithm(obs_sequence);
                test_prob[digit][file_idx - 31][other_digit] = prob;
            }

            // Find the digit model with the maximum probability
            int recognized_digit = 0;
            long double max_prob = test_prob[digit][file_idx - 31][0];
            for (int other_digit = 1; other_digit < NUM_TYPES; other_digit++) {
                if (test_prob[digit][file_idx - 31][other_digit] > max_prob) {
                    max_prob = test_prob[digit][file_idx - 31][other_digit];
                    recognized_digit = other_digit;
                }
            }

            // Check if the recognized digit is correct
            if (recognized_digit == digit) {
                correct_predictions++;
            }

            std::cout << "Test file " << file_idx << " for digit " << digit 
                      << " recognized as " << recognized_digit 
                      << " with probability " << max_prob << std::endl;

            file_idx++;
        }

        obs_file.close();
    }

    
    double accuracy = (double)correct_predictions;
    std::cout << "Total Correct Predictions: " << correct_predictions << std::endl;
    std::cout << "Recognition Accuracy: " << accuracy << "%" << std::endl;
}





void saveCodebookToFile(double **codebook, int codebookSize, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s for writing\n", filename);
        return;
    }

    for (int i = 0; i < codebookSize; i++) {
        for (int j = 0; j < P; j++) {
            fprintf(file, "%lf ", codebook[i][j]);
        }
        fprintf(file, "\n"); // New line for each codeword
    }

    fclose(file);
    printf("Codebook saved to %s\n", filename);
}

void readCodebookFromFile(const char *filename, double **codebook) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < P; j++) {
            if (fscanf(file, "%lf", &codebook[i][j]) != 1) { // Read into 2D array
                fprintf(stderr, "Error reading codebook data at index [%d][%d]\n", i, j);
                fclose(file);
                return;
            }
        }
    }

    fclose(file);
    //printf("Codebook successfully read from %s\n", filename);
}


int _tmain(int argc, _TCHAR *argv[])
{
    char filename[100];
    char types[NUM_TYPES] = {'0', '1', '2'};
    int universe_index = 0;

    for (int type_index = 0; type_index < NUM_TYPES; type_index++)
    {
        for (int file_index = 1; file_index <= NUM_FILES; file_index++)
        {
            sprintf(filename, "244101020_dataset/English/txt/244101020_E_%c_%d.txt", types[type_index], file_index);

            FILE *file = fopen(filename, "r");
            if (!file)
            {
                printf("Error opening test file %s\n", filename);
                continue;
            }

            int file_size = 0;
            ll temp;
            while (fscanf(file, "%lf", &temp) != EOF)
            {
                file_size++;
            }
            fseek(file, 0, SEEK_SET);

            ll *array = (ll *)malloc(file_size * sizeof(ll));
            if (array == NULL)
            {
                printf("Memory allocation error for file %s\n", filename);
                fclose(file);
                continue;
            }

            for (int i = 0; i < file_size; i++)
            {
                fscanf(file, "%lf", &array[i]);
            }

            Dc_Shift(array, file_size);
            normalization(array, file_size);

            ll test_ci[NUM_FRAMES][P + 1] = {0};
            for (int frame = 0; frame < NUM_FRAMES; frame++)
            {
                int frame_start = frame * N;
                if (frame_start + N > file_size)
                    break;

                ll frame_data[N];
                for (int i = 0; i < N; i++)
                {
                    frame_data[i] = array[frame_start + i];
                }

                // Apply Hamming window to the frame
                applyHammingWindowToFrame(frame_data, N);

                // Compute autocorrelations after applying the Hamming window
                ll R_temp[P + 1] = {0};
                for (int j = 0; j <= P; j++)
                {
                    R_temp[j] = autocorrelation(frame_data, j, 0);
                }

                ll E_temp[P + 1] = {0};
                ll alpha_temp[13][13] = {0};
                calculateErrorsAndAlphas(R_temp, E_temp, alpha_temp, P);

                calculateCepstralCoefficients(alpha_temp, test_ci[frame], R_temp[0]);
                // applyHammingWindow(test_ci[frame], P + 1);
                raisedSineWindow(test_ci[frame], P + 1);
            }

            saveCiValues(test_ci, types[type_index], file_index);

            if (universe_index >= M)
                continue;
            for (int i = 0; i < NUM_FRAMES; i++)
            {
                for (int j = 1; j <= P; j++)
                {
                    universe[universe_index][j - 1] = test_ci[i][j];
                    // printf("%f ", universe[universe_index][j - 1]);
                }
                // printf("\n");
                universe_index++;
            }

            free(array);
            fclose(file);
        }
    }

    //Initialize codebooks
	
    double** codebook = (double**)malloc(K * sizeof(double*));
double** temp_codebook = (double**)malloc((K / 2) * sizeof(double*));
for (int i = 0; i < K; i++) codebook[i] = (double*)malloc(P * sizeof(double));
for (int i = 0; i < K / 2; i++) temp_codebook[i] = (double*)malloc(P * sizeof(double));

    int region[M];

    initialize_codebook_with_centroid(temp_codebook, universe);

    int current_codebook_size = 1;
    int max_iterations = 10; // Define the number of iterations you want for each codebook size.

    // No cycle-based termination; ensure every iteration runs
    while (current_codebook_size < K)
    {
        split_codebook(codebook, temp_codebook, current_codebook_size);
        current_codebook_size *= 2;

        //printf("Training with codebook size: %d\n", current_codebook_size);

        double prev_distortion = 0.0, current_distortion = 0.0;
        int iteration = 0;

        // Run until distortion change is below the threshold
        do
        {
            iteration++;
           // printf("Iteration %d:\n", iteration);

            // Assign vectors to their nearest codebook regions
            assign_to_regions(universe, codebook, region, current_codebook_size);

            // Calculate the new total distortion
            current_distortion = calculate_distortion(universe, codebook, region, current_codebook_size);
           // printf("Distortion after iteration %d: %lf\n", iteration, current_distortion);

            // Update the codebook with new centroids
            update_centroids(universe, codebook, region, current_codebook_size);

            // Check for convergence
            if (fabs(prev_distortion - current_distortion) < THRESHOLD)
            {
                //printf("Convergence reached at iteration %d with distortion: %lf\n", iteration, current_distortion);
                break;
            }

            prev_distortion = current_distortion;

        } while (1); // Infinite loop until convergence is met
        if (current_codebook_size == K)
            break;
        // Copy the updated codebook to temp_codebook for the next phase
        for (int i = 0; i < current_codebook_size; i++)
        {
            memcpy(temp_codebook[i], codebook[i], sizeof(double) * P);
        }
        //printf("current Codebook Size: %d\n", current_codebook_size);
    }

    printf("Final Codebook Size: %d\n", K);
saveCodebookToFile(codebook, K, "final_codebook.txt");
     //Generating the observation sequences using the finalized codebook
	generate_observation_sequences(codebook);
	for(int i=0;i<3; i++){
	cout <<"HMM: "<< i;
		implement_hmm3(i);
	}
generate_observation_sequences_test(codebook);
	
    return 0;
}

// Free memory
