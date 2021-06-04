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

double sigmoid_prime(double x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}

void sigmoid_primeify(Matrix<double>& mat)
{
  mat.forEach([](double& x) {x = sigmoid_prime(x);});
}

class SingleLayerPerceptron
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
    SingleLayerPerceptron(int _input, int _hidden, int _output, double _lr = 0.2)
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
    
    void train(Matrix<double> inputs, Matrix<double> target)
    {
      Matrix<double> hidden = weights_ih * (inputs.transpose());
      hidden = hidden + bias_h;
      sigmoidify(hidden);

      Matrix<double> output = weights_ho * hidden;
      output = output + bias_o;
      sigmoidify(output);

      target = target.transpose();
      Matrix<double> output_errors = target - output;
      Matrix<double> hidden_errors = weights_ho.transpose() * output_errors;

      Matrix<double> grad = output;
      grad.forEach([] (double& x){
        x = x * (1-x);
      });
      grad = multiplyElementWise(grad, output_errors);
      grad = grad * learning_rate;
      
      bias_o = bias_o + grad;
      
      Matrix<double> delta_weights_ho = grad * hidden.transpose();

      weights_ho = weights_ho + delta_weights_ho;

      Matrix<double> grad_hidden = hidden;
      grad_hidden.forEach([] (double& x) {
        x = x * (1 - x);
      });
      grad_hidden = multiplyElementWise(grad_hidden, hidden_errors);
      grad_hidden = grad_hidden * learning_rate;

      bias_h = bias_h + grad_hidden;
      Matrix<double> delta_weights_ih = grad_hidden * inputs;
      weights_ih = weights_ih + delta_weights_ih;
    }

};

class MultiLayerPerceptron
{
  private:
    int num_layers;
    vector<int> sizes;
    vector<Matrix<double>> biases;
    vector<Matrix<double>> weights;
    
    // Constructor
  public:
    MultiLayerPerceptron(int _layers, vector<int> _sizes)
    {
      srand(time(NULL));
      num_layers = _layers;
      sizes = _sizes;
           
      if(num_layers != _sizes.size())
      {
        throw std::invalid_argument("Number of layers must be equal to the length of the size vector.");
      }

      if(_sizes.size() < 2)
      {
        throw std::invalid_argument("Multilayer perceptron must have at least two layers.");
      }

      for(int i = 1; i < num_layers; i++)
      {
        Matrix<double> bias{sizes[i], 1};
        Matrix<double> weight{sizes[i], sizes[i - 1]};

        bias.randomize();
        weight.randomize();

        biases.push_back(bias);
        weights.push_back(weight);
      }
    }

    Matrix<double> feedForward(Matrix<double> input)
    {
      input = input.transpose();
      for(int i = 0; i < biases.size(); i++)
      {
        input = (weights[i] * input) + biases[i];
        sigmoidify(input);
      }
      return input;
    }

    pair<vector<Matrix<double>>, vector<Matrix<double>>> 
    backprop(Matrix<double> x, Matrix<double> y)
    {

      x = x.transpose();

      vector<Matrix<double>> grad_b;
      vector<Matrix<double>> grad_w;

      for(const auto& b : biases)
      {
        Matrix<double> x{b.rows, b.cols};
        grad_b.push_back(x);
      }
    
      for(const auto& w : weights)
      {
        Matrix<double> x{w.rows, w.cols};
        grad_w.push_back(x);
      }
      
      // feedForward
            
      Matrix<double> activation = x;
      vector<Matrix<double>> activations = {x};
      vector<Matrix<double>> zs;
      Matrix<double> z;
      for(int i = 0; i < biases.size(); i++)
      {
        z = (weights[i] * activation) + biases[i];
        zs.push_back(z);
        activation = z;
        sigmoidify(activation);
        activations.push_back(activation);
      }
   

      // backpropagation

      // calculate output errors and gradient
      Matrix<double> delta = activations.back() - y;

      Matrix<double> temp = zs.back();
      sigmoid_primeify(temp);
      delta = delta * temp;
      

      grad_b.back() = delta;

      grad_w.back() = delta * activations[activations.size() - 2].transpose();

      // backpropagate to previous layers
      for(int l = 2; l < num_layers; l++)
      {
        z = zs[zs.size() - l]; 
        Matrix<double> sp = z;
        sigmoid_primeify(sp);
        delta = weights[weights.size() -l + 1].transpose() * delta;
        delta = multiplyElementWise(delta, sp);
        grad_b[grad_b.size() - l] = delta;
        grad_w[grad_w.size() - l] = delta * activations[activations.size() - l - 1].transpose(); 
      }
      return {grad_b, grad_w};
    }
};