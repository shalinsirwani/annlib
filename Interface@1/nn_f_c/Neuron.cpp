#include "Neuron.hpp"

void Neuron::setVal(double val , bool ad) {
  this->val = val;
  if(ad){
  activate();
  derive();
  }
}

void Neuron::activate() {

  if(this->activationType == TANH) {
    this->activatedVal = tanh(this->val);
  } else if(this->activationType == RELU) {
    if(this->val > 0) {
      this->activatedVal = this->val;
    } else {
      this->activatedVal = 0;
    }
  } else if(this->activationType == 3) {
        this->activatedVal = (1 / (1 + exp(-this->val)));

  } else {
        this->activatedVal = (1 / (1 + exp(-this->val)));
  }
}

void Neuron::derive() {
  if(activationType == TANH) {
    this->derivedVal = (1.0 - (this->activatedVal * this->activatedVal));
  } else if(activatedVal == RELU) {
    if(this->val > 0) {
      this->derivedVal = 1;
    } else {
      this->derivedVal = 0;
    }
  } else if(activatedVal == SIGM) {
    this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
  } else {
    this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
  }
}

// Constructor
Neuron::Neuron(double val) {
  this->setVal(val);
}

Neuron::Neuron(double val, int activationType , bool ad) {
  this->activationType = activationType;
  this->setVal(val , ad);
}


