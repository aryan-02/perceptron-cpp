#include <vector>
#include <string>
using namespace std;

struct invalidMatrixShapeMult : public exception
{
   const char * what () const throw ()
   {
      return "Invalid Shapes for Matrix Multiplication";
   }
};

struct invalidMatrixShapeAdd : public exception
{
   const char * what () const throw ()
   {
      return "Invalid Shapes for Matrix Addition";
   }
};


template <typename T>
class Matrix
{
  public:
    int rows = 1;
    int cols = 1;
    vector<vector<T>> data {0};
    Matrix(int _r, int _c)
    {
      rows = _r;
      cols = _c;
      vector<vector<T>> v(_r, vector<T>(_c, 0));
      data = v;
    }

    Matrix(vector<vector<T>> v)
    {
      rows = v.size();
      cols = (rows != 0) ? v[0].size() : 0;
      data = v;
    }

    static Matrix<T> identity(int n)
    {
      Matrix<T> mat{n, n};
      for(int i = 0; i < n; i++)
      {
        mat.data[i][i] = 1;
      }
      return mat;      
    }

    std::string toString() const
    {
      std::string res = "";
      for(vector<T> i : data)
      {
        for(T j : i)
        {
          res += to_string(j) + " ";
        }
        res += "\n";
      }
      return res;
    }

    void scale(T factor)
    {
      for(int r = 0; r < rows; r++)
      {
        for(int c = 0; c < cols; c++)
        {
          data[r][c] *= factor;
        }
      }
    }
};

template<typename T>
Matrix<T> multiply(Matrix<T> a, Matrix<T> b)
{
    if(a.cols != b.rows)
    {
      throw invalidMatrixShapeMult();
    }
    else
    {
      int rows = a.rows;
      int cols = b.cols;
      Matrix<T> res{rows, cols};
      for(int r = 0; r < rows; r++)
      {
        for(int c = 0; c < cols; c++)
        {
          T sum = 0;
          for(int i = 0; i < a.cols; i++)
          {
            sum += a.data[r][i] * b.data[i][c];
          }
          res.data[r][c] = sum;
        }
      }
      return res;
    }
}

template<typename T>
Matrix<T> add(Matrix<T> a, Matrix<T> b)
{
  if(a.rows != b.rows and a.cols != b.cols)
  {
    throw invalidMatrixShapeAdd();
  }
  else
  {
    Matrix<T> res{a.rows, a.cols};
    for(int i = 0; i < a.rows; i++)
    {
      for(int j = 0; j < a.cols; j++)
      {
        res.data[i][j] = a.data[i][j] + b.data[i][j];
      }
    }
    return res;
  }
}

template<typename T>
Matrix<T> operator* (const Matrix<T>& a, const Matrix<T>& b)
{
  return multiply(a, b);
}

template<typename T>
Matrix<T> operator+ (const Matrix<T>& a, const Matrix<T>& b)
{
  return add(a, b);
}

template<typename T>
ostream& operator<< (ostream& os, const Matrix<T>& mat)
{
  os << mat.toString();
  return os;
}