#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include "Matrix.hpp"
#include "Layer.hpp"
#include "json.hpp"

using namespace std;

#define COST_MSE 1

/*enum ANN_COST {
  COST_MSE
};*/

/*enum ANN_ACTIVATION {
  A_TANH,
  A_RELU,
  A_SIGM
};*/

/*struct ANNConfig {
  vector<int> topology;
  double bias;
  double learningRate;
  double momentum;
  int epoch;
  ANN_ACTIVATION hActivation;
  ANN_ACTIVATION oActivation;
  ANN_COST cost;
  string trainingFile;
  string labelsFile;
  string weightsFile;
};*/

class NeuralNetwork
{
public:
  //NeuralNetwork(ANNConfig config);

  // Constructor
  NeuralNetwork(
                vector<int> topology,
                int hiddenActivationType = 2,
                int outputActivationType = 3,
                int costFunctionType = 1,
                double bias = 1,
                double learningRate = 0.05,
                double momentum = 1
                 );

  void train(
        vector<double> input,
        vector<double> target,
        double bias,
        double learningRate,
        double momentum
      );

  void setCurrentInput(vector<double> input , bool ad = true);
  void setCurrentTarget(vector<double> target) { this->target = target; };

  void feedForward();
  void backPropagation();
  void setErrors();
  void printtoconsole();

  vector<double> getActivatedVals(int index) { return this->layers.at(index)->getActivatedVals(); }

  Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); }
  Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
  Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); }
  Matrix *getWeightMatrix(int index) { return new Matrix(*this->weightMatrices.at(index)); };
  Matrix *matrixifyError();
  Matrix *set_errs_for_nxt_itr(Matrix * , Matrix *);

  void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); }
  void lr_inclusion(Matrix *);

  void saveWeights(string file);
  void loadWeights(string file);

  int topologySize;
  int hiddenActivationType  = RELU;
  int outputActivationType  = TANH;
  int costFunctionType      = COST_MSE;

  vector<int> topology;
  vector<Layer *> layers;
  vector<Matrix *> weightMatrices;
  vector<Matrix *> gradientMatrices;

  vector<double> input;
  vector<double> target;
  vector<double> errors;
  vector<double> derivedErrors;
  vector<double> testerror;

  double error              = 0;
  double bias               = 1;
  double momentum           = 1;
  double learningRate       = 0.05;

  //ANNConfig config;

private:
  void setErrorMSE();
};

#endif