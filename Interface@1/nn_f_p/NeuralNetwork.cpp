#include "NeuralNetwork.hpp"
#include "Math.hpp"

void NeuralNetwork::setCurrentInput(vector<double> input , bool ad) {
  this->input = input;

  for(int i = 0; i < input.size(); i++) {
    this->layers.at(0)->setVal(i, input.at(i) , false);
  }
}

void NeuralNetwork::printtoconsole()
{
    cout<<"--------PRINTING LAYERS---------"<<endl;

    for(int i = 0; i < this->topologySize; i++)
    {
        cout<<"Layer "<<i<<" :";
        if(i == 0)
            this->layers[i]->matrixifyVals()->printToConsole();
        else{
            this->layers[i]->matrixifyActivatedVals()->printToConsole();
            }
    }
    cout<<endl<<endl<<endl;
    cout<<"---------PRINTING WEIGHTS----------"<<endl;

    for(int i = 0; i < (this->topologySize - 1); i++)
    {
        cout<<"Weight matrixes "<<i<<" :\n";
        this->getWeightMatrix(i)->printToConsole();
        cout<<endl;
    }
}

Matrix *NeuralNetwork::matrixifyError()
{
  Matrix *m = new Matrix(1, this->derivedErrors.size(), false);

  for(int i = 0; i < this->derivedErrors.size(); i++) {
    m->setValue(0, i, this->derivedErrors.at(i));
  }

  return m;
}


void NeuralNetwork::lr_inclusion(Matrix *gradient)
{
  for(int i = 0; i < gradient->getNumCols(); i++) {
    gradient->setValue(
                    0, 
                    i, 
                    gradient->getValue(0 , i) * this->learningRate
                      );
  }
}

Matrix *NeuralNetwork::
set_errs_for_nxt_itr(Matrix *weight , Matrix *erorrs)
{
  Matrix *weight_transpose = weight->transpose();

  Matrix *temp_store = new Matrix(
                                  erorrs->getNumRows() , 
                                  weight_transpose->getNumCols() , 
                                  false
                                 );

  utils::Math::multiplyMatrix(
                            erorrs , 
                            weight_transpose , 
                            temp_store
                              );


  delete weight_transpose;
  return temp_store;
}