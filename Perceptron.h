#include <cmath>
#include "Matrix.h"

double sigmoid(double x)
{
  return (double) 1 / (1 + exp(-x));
}

void sigmoidify(Matrix<double>& mat)
{
  mat.forEach([](double& x) {x = sigmoid(x);});
}

class Perceptron
{
  private:
    int input_nodes{};
    int hidden_nodes{};
    int output_nodes{};
    double learning_rate{};
    
    Matrix<double> weights_ih;
    Matrix<double> weights_ho;

    Matrix<double> bias_h;
    Matrix<double> bias_o;
    

  public:
    Perceptron(int _input, int _hidden, int _output, double _lr = 0.2)
    {
      srand(time(NULL));
      input_nodes = _input;
      hidden_nodes = _hidden;
      output_nodes = _output;
      learning_rate = _lr;

      Matrix<double> w_ih(hidden_nodes, input_nodes);
      Matrix<double> w_ho(output_nodes, hidden_nodes);

      weights_ih = w_ih;
      weights_ho = w_ho;
      weights_ih.randomize();
      weights_ho.randomize();

      Matrix<double> b_h (hidden_nodes, 1);
      Matrix<double> b_o (output_nodes, 1);

      bias_h = b_h;
      bias_o = b_o;
      bias_h.randomize();
      bias_o.randomize();

    }

    Matrix<double> feedForward(Matrix<double> inp)
    {
      Matrix<double> hidden = weights_ih * (inp.transpose());
      hidden = hidden + bias_h;
      sigmoidify(hidden);

      Matrix<double> output = weights_ho * hidden;
      output = output + bias_o;
      sigmoidify(output);
      return output;
    }
    
};