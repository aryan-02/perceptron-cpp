#include <vector>
#include <string>
#include <functional>

using namespace std;

double randomDouble()
{
  return (double)rand() / RAND_MAX;
}

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
    Matrix()
    {
      rows = 0;
      cols = 0;
      data = {};
    }
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

    Matrix<T> transpose()
    {
      Matrix res{cols, rows};
      for(int r = 0; r < rows; r++)
      {
        for(int c = 0; c < cols; c++)
        {
          res.data[c][r] = data[r][c];
        }
      }
      return res;
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

    vector<T>& operator[] (int idx)
    {
      if(idx >= data.size())
      {
        throw std::out_of_range("Matrix index out of range.");
      }
      else{
        return data[idx];
      }
    }

    // Return a matrix where row R and column C are removed
    // Useful for determinants
    Matrix<T> subMatRemove(int R, int C)
    {
      if(R >= rows or C >= cols)
      {
        throw std::out_of_range("Submatrix remove index out of range.");
      }
      else
      {
        vector<vector<T>> res;
        for(int r = 0; r < rows; r++)
        {
          if(r != R)
          {
            vector<T> curr_row;
            for(int c = 0; c < cols; c++)
            {
              if(c != C)
              {
                curr_row.push_back(data[r][c]);
              }
            }
            res.push_back(curr_row);
          }
        }
        return Matrix(res);
      }
    }

    void randomize()
    {
      for(int r = 0; r < rows; r++)
      {
        for(int c = 0; c < cols; c++)
        {
          data[r][c] = 2 * randomDouble() - 1;
        }
      }
    }

    void forEach(std::function<void(T&)> operation)
    {
      for(vector<T>& row : data)
      {
        for_each(row.begin(), row.end(), operation);
      }
    }
};

template<typename T>
Matrix<T> scale(Matrix<T> mat, T factor)
{
  Matrix<T> res{mat.rows, mat.cols};
  for(int r = 0; r < mat.rows; r++)
  {
    for(int c = 0; c < mat.cols; c++)
    {
      res.data[r][c] = mat.data[r][c] * factor;
    }
  }
  return res;
}

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
  if(a.rows != b.rows or a.cols != b.cols)
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
Matrix<T> operator- (const Matrix<T>& a, const Matrix<T>& b)
{
  return add(a, scale(b, (T) -1));
}

template<typename T>
Matrix<T> operator* (const Matrix<T>& mat, const T& factor)
{
  return scale(mat, factor);
}

template<typename T>
Matrix<T> operator* (const T& factor, const Matrix<T>& mat)
{
  return scale(mat, factor);
}

template<typename T>
ostream& operator<< (ostream& os, const Matrix<T>& mat)
{
  os << mat.toString();
  return os;
}

template<typename T>
Matrix<T> multiplyElementWise(Matrix<T> a, Matrix<T> b)
{
  if(a.rows != b.rows or a.cols != b.cols)
  {
    throw std::invalid_argument("Matrix shape mismatch for element-wise multiplication");
  }
  else
  {
    Matrix<T> res(a.rows, a.cols);
    for(int r = 0; r < a.rows; r++)
    {
      for(int c = 0; c < a.cols; c++)
      {
        res[r][c] = a[r][c] * b[r][c];
      }
    }
    return res;
  }
}

template<typename T>
double determinant(Matrix<T> mat)
{
  if(mat.rows != mat.cols)
  {
    throw std::invalid_argument("Non-square matrix provided to the determinant function.");
  }
  else
  {
    int size = mat.rows;
    if(size == 2)
    {
      return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
    }
    else
    {
      double res = 0;
      for(int i = 0; i < size; i++)
      {
        res += mat[0][i] * ((i % 2 == 0) ? 1 : -1) * determinant(mat.subMatRemove(0, i));
      }
      return res;
    }
  }
}

template<typename T>
T& get(vector<T> arr, int idx) 
{
  
    if(idx >= 0)
    {
      return arr.at(idx);
    }
    else
    {
      return arr.at(arr.size() + idx);
    }
}