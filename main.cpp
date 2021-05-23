#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
  Matrix<int> m ({{3, 2, 0}, {5, 2, 1}});
  Matrix<int> a ({{2, 4}, {3, 2}});
  Matrix<int> c = a * m;

  cout << a << '\n';
  cout << m << '\n';
  cout << c << '\n';
  cout << m + c << '\n';
  cout << Matrix<int>::identity(6) << '\n';

}