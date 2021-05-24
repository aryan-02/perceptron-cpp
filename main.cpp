#include <iostream>
#include <algorithm>
#include "Perceptron.h"

using namespace std;

#define ITERATIONS 100000

int main()
{
  Perceptron nn(2, 5, 1);
  Matrix<double> input({{1, 2}});
  vector<tuple<double, double, double>> training_data = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
  };
  for(int i = 0; i < ITERATIONS; i++)
  {
    int x = rand() % 4;
    Matrix<double> input(
      {{get<0>(training_data[x]), get<1>(training_data[x])}}
    );
    Matrix<double> output({{get<2>(training_data[x])}});
    nn.train(input, output);
  }
  Matrix<double> a({{0, 0}});
  Matrix<double> b({{0, 1}});
  Matrix<double> c({{1, 0}});
  Matrix<double> d({{1, 1}});

  cout << "Predictions:\n";
  cout << "0 xor 0: " << nn.feedForward(a);
  cout << "0 xor 1: " << nn.feedForward(b);
  cout << "1 xor 0: " << nn.feedForward(c);
  cout << "1 xor 1: " << nn.feedForward(d);
}

/*
int main()
{
  Matrix<int> m ({{3, 2, 0}, {5, 2, 1}});
  Matrix<int> a ({{2, 4}, {3, 2}});
  Matrix<int> c = a * m;

  cout << a << '\n';
  cout << m << '\n';
  cout << c << '\n';
  
  cout << m + c << '\n';
  cout << 4 * Matrix<int>::identity(6) << '\n';

  cout << a[0][0] << '\n' << '\n';
  cout << Matrix<int>::identity(4) << '\n';
  cout << Matrix<int>::identity(4).subMatRemove(1, 0) << '\n';

  Matrix<double> test ({{1, 2, 3}, {3, 2, 7}, {1, 2, 5}});
  cout << determinant(test) << "\n\n";

  Matrix<int> hello(
    {
      {1, 2, 3, 4}, 
      {5, 6, 7, 8}
    }
  );
  hello.forEach([](int& x){x += 1;});
  cout << hello.transpose() << '\n';

}
*/