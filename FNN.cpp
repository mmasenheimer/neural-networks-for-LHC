#include <iostream>
#include <vector>
#include <cmath>
#include <random>
// Generate random numbers to initialize the weights
#include <stdexcept>
#include <chrono>
// Help measuring the training time
// Activation function in the AIEs
// Efficient activation of calulation functions

// TODO look into vitis ai development kit

#include <fstream>
#include <sstream>
#include <string>
using namespace std;

namespace Activation{
    inline double relu(double x) { return max(0.0,x); }
    // Return x if it's positive and 0 if x < 0

    inline double reluDerivative(double x) { return (x > 0) ? 1.0 : 0.0; }
    // If input x > 0, the derivative is 1, this is to adjust the 
    // weights and improve learning

    inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    // Help decide if a point is in the circle (1.0) or outside (0.0)

    inline double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
        // Use the sigmoid derivative during back prop 
        // to find out how much adjustment of weights to minimize error
    }
}

class Matrix{
private:
    vector<vector<double>> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c):rows(r), cols(c) {
        data.resize(rows, vector<double>(c, 0.0));
        // Each matrix starts with 0 values per row column entry
    }

    double &operator()(size_t i , size_t j){return data[i][j]; }
    const double &operator()(size_t i, size_t j) const {return data[i][j];}
    // When we access the matrix element using const matrix 
    // object, this generates read-only access

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};

class NeuralNetwork {
    // Stores number of layers, neurons per layer, and weights connecting them

private:

    vector<int> layerSizes;
    Matrix weights1, weights2, weights3;
    vector<double> bias1, bias2, bias3;

    mt19937 gen;
    // Initialize the weights with random values

    void initializeWeights() {
        // Randomly assign weight values to matrices
        normal_distribution<> dist(0.0, 1.0);

        // He initializing to prevent vanishing gradients or slow convergence
        double scale1 = sqrt(2.0 / layerSizes[0]);
        double scale2 = sqrt(2.0 / layerSizes[1]);
        double scale3 = sqrt(2.0 / layerSizes[2]);
        // Divide 2 by the # of neurons per layer 
        // and take sqrt to ensure weights are not too large nor too small

        for (int i = 0; i < weights1.getRows(); i++) {
            for (int j = 0; j < weights1.getCols(); j++) {
                weights1(i, j) = dist(gen) * scale1;
            }
        }

        for (int i = 0; i < weights2.getRows(); i++) {
            for (int j = 0; j < weights2.getCols(); j++) {
                weights2(i, j) = dist(gen) * scale2;
            }
        }

        for (int i = 0; i < weights3.getRows(); i++) {
            for (int j = 0; j < weights3.getCols(); j++) {
                weights3(i, j) = dist(gen) * scale3;
            }
        }

        // The model starts training with no bias
        fill(bias1.begin(), bias1.end(), 0.0);
        fill(bias2.begin(), bias2.end(), 0.0);
        fill(bias3.begin(), bias3.end(), 0.0);
    }

    // Customizable structure by specifying the number of neurons in each layer

public:

    // Access weights
    const Matrix& getWeights1() const { return weights1; }
    const Matrix& getWeights2() const { return weights2; }
    const Matrix& getWeights3() const { return weights3; }

    // Access biases
    const vector<double>& getBias1() const { return bias1; }
    const vector<double>& getBias2() const { return bias2; }
    const vector<double>& getBias3() const { return bias3; }
    
    NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize)
    : layerSizes{inputSize, hidden1Size, hidden2Size, outputSize},
        weights1(inputSize, hidden1Size),
        weights2(hidden1Size, hidden2Size),
        weights3(hidden2Size, outputSize),
        // Initializes weight matrices for connections between layers
        
        bias1(hidden1Size), bias2(hidden2Size), bias3(outputSize),
        gen(random_device{}())
    
    {
        if (inputSize <= 0 || hidden1Size <= 0 || 
            hidden2Size <= 0 || outputSize <= 0) {
            throw invalid_argument("Layer sizes must be positive");
        }

        initializeWeights();
    }

    vector<double> forward (const vector<double> &input) {
        if (input.size() != layerSizes[0]) {
            throw runtime_error("input size mismatch");
        }

        vector<double> hidden1(layerSizes[1]);
        // Create a vector to store the activations of the first hidden layer

        for (int j = 0; j < layerSizes[1]; j++) {
            double sum = bias1[j];
            for (int i = 0; i < layerSizes[0]; i++) {
                // Look through all neurons in the previous layer
                sum += input[i] * weights1(i, j);
                // Multiply each input to its corresponding weight
            }

            hidden1[j] = Activation::relu(sum);
        }

        vector<double> hidden2(layerSizes[2]);
        for (int j = 0; j < layerSizes[2]; j++) {
            double sum = bias2[j];
            for (int i = 0; i < layerSizes[1]; i++) {
                sum += hidden1[i] * weights2(i, j);
            }

            hidden2[j] = Activation::relu(sum);
        }

        vector<double> output(layerSizes[3]);
        for (int j = 0; j < layerSizes[3]; j++) {
            double sum = bias3[j];
            for (int i = 0; i < layerSizes[2]; i++) {
                sum += hidden2[i] * weights3(i, j);
        
            }

            output[j] = Activation::sigmoid(sum);
        }

        return output;
    }

    // Handles the learning process, takes a set of inputs and 
    // corresponding target outputs, with a learning rate to control weight 
    // with a learning rate to control weight adjustments, and number 
    // of epochs (num times the network goes through the dataset)

    void train(const vector<vector<double>> &inputs,
        const vector<vector<double>> & targets, double learningRate, int epochs) {
        
        if (inputs.size() != targets.size()) {
            throw runtime_error("Input and target sizes don't match!");

        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;
            // Track the total error across all training samples

            for(size_t k = 0; k < inputs.size(); k++) {
                vector<double> hidden1(layerSizes[1]);

                vector<double> hidden1Pre(layerSizes[1]);
                // Store the pre activation values of the first 
                // hidden layer (Used in gradient descent calculations)

                for (int j = 0; j < layerSizes[1]; j++) {
                    double sum = bias1[j];
                    for (int i = 0; i < layerSizes[0]; i++) {
                        sum += inputs[k][i] * weights1(i, j);

                    }

                    hidden1Pre[j] = sum;

                    hidden1[j] = Activation::relu(sum);
                }
               
                // Repeat for second layer
                vector<double> hidden2(layerSizes[2]);
                vector<double> hidden2Pre(layerSizes[2]);

                for (int j = 0; j < layerSizes[2]; j++) {
                    double sum = bias2[j];
                    for (int i = 0; i < layerSizes[1];i++) {
                        sum += hidden1[i] * weights2(i, j);
                    }

                    hidden2Pre[j] = sum;
                    hidden2[j] = Activation::relu(sum);
                }

                vector<double> output(layerSizes[3]);
                vector<double> outputPre(layerSizes[3]);

                for (int j = 0; j < layerSizes[3]; j++) {
                    double sum = bias3[j];
                    for (int i = 0; i < layerSizes[2]; i++) {
                        sum += hidden2[i] * weights3(i, j);
                    }

                    outputPre[j] = sum;
                    output[j] = Activation::sigmoid(sum);
                }

                // Calculate the error for reporting, helps track the training perf.

                for (int j = 0; j < layerSizes[3]; j++) {
                    double error = targets[k][j] - output[j];

                    // Mean squared error for calculating predicted vs actual values
                    totalError += error * error;
                }

                vector<double> outputGradients(layerSizes[3]);

                for (int j = 0; j< layerSizes[3]; j++) {
                    // Compute the values for output gradients
                    outputGradients[j] = (output[j] - targets[k][j]) * 
                    Activation:: sigmoidDerivative(outputPre[j]);

                }

                // Compute the gradients for the hidden 2 layers
                vector<double> hidden2Gradients(layerSizes[2]);
                // Determine how much error is propogated back from the output layer
                // Sum the contributions of the output 
                // gradients scaled by the corresponding weights

                for (int i = 0; i < layerSizes[2]; i++) {
                    double error = 0;
                    for (int j = 0; j < layerSizes[3]; j++) {
                       
                        error += outputGradients[j] * weights3(i, j);
                        // Represents how much each neuron contributes 
                        // to the total error in the output layer
                    }

                    hidden2Gradients[i] = error * Activation::reluDerivative(hidden2Pre[i]);
                }

                // Calculate the gradients for the first hidden layer

                vector<double> hidden1Gradients(layerSizes[1]);
                for (int i = 0; i < layerSizes[1]; i++) {
                    double error = 0;
                    for (int j = 0; j < layerSizes[2]; j++) {
                        error += hidden2Gradients[j] * weights2(i, j);
                        // Once we have the error, we multiply by the derivative of the
                        // Activation function apply to the pre activation 
                        // value for each neuron in hidden 1

                    }

                    hidden1Gradients[i] = error * Activation:: reluDerivative(hidden1Pre[i]);
                }

                // Updating the weights with gradient descent
                for (int i = 0; i < layerSizes[2]; i++) {
                    for (int j = 0; j < layerSizes[3]; j++) {
                        weights3(i, j) -= learningRate * outputGradients[j] * hidden2[i];
                    }
                }

                for (int j = 0; j < layerSizes[3]; j++) {
                    bias3[j] -= learningRate * outputGradients[j];
                }

                for (int i = 0; i < layerSizes[1]; i++) {
                    for (int j = 0; j < layerSizes[2]; j++) {
                        weights2(i, j) -= learningRate * hidden2Gradients[j] *  hidden1[i];
                    }
                }

                for (int j = 0; j < layerSizes[2]; j++) {
                    bias2[j] -= learningRate * hidden2Gradients[j];
                }

                for (int i = 0; i < layerSizes[0]; i++) {
                    for (int j = 0; j < layerSizes[1]; j++) {
                        weights1(i, j) -= learningRate * hidden1Gradients[j] * inputs[k][i];
                    }
                }

                for (int j = 0; j < layerSizes[1]; j++) {
                    bias1[j] -= learningRate * hidden1Gradients[j];
                }

                // Print out the MSE every 100 epochs, helps 
                // manage how well the network learns over time

                if (epoch % 100 == 0) {
                    cout << "EPOCH : " << epoch << "  MSE: " <<
                    totalError / inputs.size() << "\n";

                }
            }
        }
    }
};


// FOR GRAPH PROGRAMMING: ADJUST THIS TO READ IN INPUT DATA WITH KERNELS, NOT IFSTREAM


pair<vector<vector<double>>, vector<vector<double>>> loadCSVData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    vector<vector<double>> inputs;
    vector<vector<double>> targets;
    string line;
    
    while (getline(file, line)) {
        // Skip header line
        if (line.empty() || line[0] == '#') continue;
        
        stringstream ss(line);
        string cell;
        vector<double> row;
        
        while (getline(ss, cell, ',')) {
            // Parse CSV line
            cell.erase(0, cell.find_first_not_of(" \t"));
            cell.erase(cell.find_last_not_of(" \t") + 1);
            
            if (!cell.empty()) {
                row.push_back(stod(cell));
            }
        }
        
        if (row.size() >= 3) {
            inputs.push_back({row[0], row[1]});
            // x, y coordinates
            targets.push_back({row[2]});
            // target value
        }
    }
    
    file.close();
    cout << "Loaded " << inputs.size() << " samples from " << filename << endl;
}

// **RUNNING AND TESTING/TRAINING THE NETWORK**
int main() {
    try{
        NeuralNetwork nn(2, 8, 4, 1);
        // 2 Input neurons, 1 hidden layer with 8 neurons, another 
        // hidden with 4 neurons, one output neuron
        mt19937 gen(random_device{}());
        // Shuffle training data to improve learning

        uniform_real_distribution<> dist(-2.0, 2.0);
        // Chnce of a point being selected is the same across the range

        const int numSamples = 1000;

        vector<vector<double>> inputs(numSamples);
        // Store the input features for each training example
        vector<vector<double>> targets(numSamples);
        // Store the corresponding outputs

        for (int i = 0; i < numSamples; i++) {
            // For each iteration, we generate a random 
            // input and its corresponding target value
            double x = dist(gen);
            double y = dist(gen);
            // Between -2.0 and 2.0 for each

            inputs[i] = {x, y};

            double distance = sqrt(x * x + y * y );
            // Euclidean algorithm to check distance from origin

            targets[i] = {distance < 1.0 ? 1.0 : 0.0};
        }

        auto start = chrono::high_resolution_clock::now();

        // Start the training process
        nn.train(inputs, targets, 0.01, 1000);
        auto end = chrono::high_resolution_clock::now();
        // Display the duration of the training time in seconds

        cout<< "Training time : " 
            <<chrono::duration_cast<chrono::milliseconds>(end - start).count() 
            << " ms\n";
    
        vector<vector<double>> testPoints = {
        // Represent different x and y coordinates to 
        // see if the nn can correctly classify
            {0.0, 0.0},
            {1.0, 1.0},
            {0.5, 0.5},
            {2.0, 0.0}
        };

        cout << "\n Test Results (1 means the point is inside, 0 outside) : \n";

        for (const auto &point : testPoints) {
            auto output = nn.forward(point);
            double actual = sqrt(point[0] * point[0] + point[1] * point[1]) < 1.0 ? 1.0 : 0.0;
            // Compute the label for each point

            cout << "Point (" << point[0] << ", " << point[1] << ") → " 
                << output[0] << " (actual: " << actual 
                << ", error: " << abs(output[0] - actual) << ")\n";
            // Print the prediced value along side the actual value
        }

    auto &w1 = nn.getWeights1();
    auto &w2 = nn.getWeights2();
    auto &w3 = nn.getWeights3();

    auto &b1 = nn.getBias1();
    auto &b2 = nn.getBias2();
    auto &b3 = nn.getBias3();
    // Write weights and biases to file
ofstream out("nn_weights.txt");
if (out.is_open()) {
    out << "# Weights1\n";
    for (int i = 0; i < w1.getRows(); i++) {
        for (int j = 0; j < w1.getCols(); j++) {
            out << w1(i, j) << " ";
        }
        out << "\n";
    }

    out << "# Weights2\n";
    for (int i = 0; i < w2.getRows(); i++) {
        for (int j = 0; j < w2.getCols(); j++) {
            out << w2(i, j) << " ";
        }
        out << "\n";
    }

    out << "# Weights3\n";
    for (int i = 0; i < w3.getRows(); i++) {
        for (int j = 0; j < w3.getCols(); j++) {
            out << w3(i, j) << " ";
        }
        out << "\n";
    }

    out << "# Bias1\n";
    for (double b : b1) out << b << " ";
    out << "\n# Bias2\n";
    for (double b : b2) out << b << " ";
    out << "\n# Bias3\n";
    for (double b : b3) out << b << " ";
    out << "\n";

    out.close();

    cout << "Weights and biases written to nn_weights.txt\n";
} else {
    cerr << "ERROR: Could not open output file\n";
}

}catch(const exception &e) {
        cerr << " ERROR : " << e.what() << endl;
        return 1;
    }

    return 0;
}