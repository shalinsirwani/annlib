#include "NeuralNetwork.hpp"

// Constructor
NeuralNetwork::NeuralNetwork(
                            vector<int> topology,
                            int hiddenActivationType,
                            int outputActivationType,
                            int costFunctionType,
                            double bias,
                            double learningRate,
                            double momentum
                            )
{
  //this->config        = config;
  this->topology      = topology;
  this->topologySize  = topology.size();
  this->learningRate  = learningRate;
  this->momentum      = momentum;
  this->bias          = bias;
  this->costFunctionType = costFunctionType;
  this->hiddenActivationType = hiddenActivationType;
  this->outputActivationType = outputActivationType;

  //this->hiddenActivationType  = config.hActivation;
  //this->outputActivationType  = config.oActivation;
  //this->costFunctionType      = config.cost;

  for(int i = 0; i < topologySize; i++) {
    if(i > 0 && i < (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->hiddenActivationType);
      this->layers.push_back(l);
    } else if(i == (topologySize - 1)) {
      Layer *l  = new Layer(topology.at(i), this->outputActivationType);
      this->layers.push_back(l);
    } else {
      Layer *l  = new Layer(topology.at(i) , false);
      this->layers.push_back(l);
    }
  }

  for(int i = 0; i < (topologySize - 1); i++) {
    Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);

    this->weightMatrices.push_back(m);
  }

  // Initialize empty errors
  for(int i = 0; i < topology.at(topology.size() - 1); i++) {
    errors.push_back(0.00);
    derivedErrors.push_back(0.00);
    testerror.push_back(0.00);
  }

  this->error = 0.00;
}