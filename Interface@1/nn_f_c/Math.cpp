#include "Math.hpp"

void utils::Math::multiplyMatrix(Matrix *a, Matrix *b, Matrix *c) {
  
  assert(a->getNumCols() == b->getNumRows());

  for(int i = 0; i < a->getNumRows(); i++) {
    for(int j = 0; j < b->getNumCols(); j++) {
      for(int k = 0; k < b->getNumRows(); k++) {
        double p      = a->getValue(i, k) * b->getValue(k, j);
        double newVal = c->getValue(i, j) + p;
        c->setValue(i, j, newVal);
      }

      c->setValue(i, j, c->getValue(i, j));
    }
  }
}

void utils::Math::subMatrix(Matrix *a , Matrix *b)
{
    assert(a->getNumRows() == b->getNumRows() && a->getNumCols() == b->getNumCols());

    for(int i = 0; i < a->getNumRows(); i++)
    {
        for(int j = 0; j < a->getNumCols(); j++)
        {
            a->setValue(i , j , (a->getValue(i , j) - b->getValue(i , j)));
        }
    }
}