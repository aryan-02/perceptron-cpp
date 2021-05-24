#include <iostream>
#include <algorithm>
#include "Perceptron.h"

using namespace std;

int main()
{
  Perceptron nn(2, 5, 3);
  Matrix<double> input({{1, 2}});
  cout << nn.feedForward(input) << '\n'; 
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