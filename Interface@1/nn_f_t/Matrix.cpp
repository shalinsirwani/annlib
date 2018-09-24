#include "Matrix.hpp"

Matrix *Matrix::transpose() {
  Matrix *m = new Matrix(this->numCols, this->numRows, false);

  for(int i = 0; i < this->numRows; i++) {
    for(int j = 0; j < this->numCols; j++) {
      m->setValue(j, i, this->getValue(i, j));
    }
  }

  return m;
}

Matrix *Matrix::copy() {
  Matrix *m = new Matrix(this->numRows, this->numCols, false);

  for(int i = 0; i < this->numRows; i++) {
    for(int j = 0; j < this->numCols; j++) {
      m->setValue(i, j, this->getValue(i, j));
    }
  }

  return m;
}

Matrix::Matrix(int numRows, int numCols, bool isRandom) {
  this->numRows = numRows;
  this->numCols = numCols;

  for(int i = 0; i < numRows; i++) {
    vector<double> colValues;

    for(int j = 0; j < numCols; j++) {
      double r = isRandom == true ? this->generateRandomNumber() : 0.00;
      colValues.push_back(r);
    }

    this->values.push_back(colValues);
  }
}

double Matrix::generateRandomNumber() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.01, 0.01);

  return dis(gen);
}

void Matrix::printToConsole() {
  for(int i = 0; i < this->numRows; i++) {
    for(int j = 0; j < this->numCols; j++) {
      cout << this->values.at(i).at(j) << "\t\t";
    }
    cout << endl;
  }
}


double Matrix::getmax()
{
  int max_id = -1;
  double max = 0.0;
  for(int j = 0; j < this->numCols; j++) {
    if(max < values[0][j])
    {
      max_id = j;
      max = values[0][j];
    }
  }
  return max_id;
}