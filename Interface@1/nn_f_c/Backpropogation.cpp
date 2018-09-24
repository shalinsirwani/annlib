#include "NeuralNetwork.hpp"
#include "Math.hpp"

void NeuralNetwork::backPropagation() {

  vector<Matrix *> saveweights;
  Matrix *errors;
  Matrix *derivedVals;
  Matrix *gradient;
  Matrix *previousLayerT;
  Matrix *deltaweight;
  Matrix *orignalweights;
  Matrix *previousLayer;
  Matrix *store_error;

  store_error = this->matrixifyError();

  for(int i = topologySize - 1; i > 0; i--)
  {
    errors = new Matrix(*store_error);

    delete store_error;
    
    derivedVals = 
    this->layers[i]->matrixifyDerivedVals();

    gradient = 
    new Matrix(1 , topology[i] , false);

    for(int k = 0; k < errors->getNumCols(); k++)
    {
      gradient->setValue(
                0 ,
                k , 
                (errors->getValue(0 , k) * derivedVals->getValue(0 , k))
                        );
    }
    
    this->lr_inclusion(gradient);


    previousLayer = 
    this->layers[i-1]->matrixifyActivatedVals();

    if( i - 1 == 0)
    {
      previousLayer = this->layers[i-1]->matrixifyVals();      
    }


    previousLayerT = previousLayer->transpose();

    deltaweight = 
    new Matrix(
              previousLayerT->getNumRows() , 
              gradient->getNumCols() , 
              false
              );

    utils::Math::multiplyMatrix(
                                previousLayerT , 
                                gradient , 
                                deltaweight
                                );

    if(i - 1 != 0){    
    store_error = this->set_errs_for_nxt_itr(
                              this->weightMatrices[i-1] ,
                               errors
                               );
            }

    orignalweights = new Matrix (*(this->weightMatrices[i-1]));

    utils::Math::subMatrix(orignalweights , deltaweight);

    saveweights.push_back(new Matrix(*orignalweights));

    delete errors;
    delete derivedVals;
    delete gradient;
    delete previousLayerT;
    delete deltaweight;
    delete orignalweights;
    delete previousLayer;
  }
  for(int i = 0; i < this->weightMatrices.size(); i++) {
    delete this->weightMatrices[i];
  }

  this->weightMatrices.clear();

  reverse(saveweights.begin(), saveweights.end());

  for(int i = 0; i < saveweights.size(); i++) {
    this->weightMatrices.push_back(new Matrix(*saveweights[i]));
    delete saveweights[i];
  }

}