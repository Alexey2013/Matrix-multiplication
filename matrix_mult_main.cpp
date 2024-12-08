#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <cstring>
#include <cstddef>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <stddef.h>


// #define MKL
// #ifdef MKL
// #include "mkl.h"
// #endif

using namespace std;

void generation(double * mat, size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
	for (size_t i = 0; i < size * size; i++)
		mat[i] = uniform_distance(gen);
}

void matrix_mult(double * a, double * b, double * res, size_t size)
{
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
	    {
			for (int k = 0; k < size; k++)
		    {
				res[i*size + j] += a[i*size + k] * b[k*size + j];
			}
		}
	}
}

void matrix_mult2(double *a, double *b, double *res, size_t size) {
    int block_size = 32;
    #pragma omp parallel for
    for (int i = 0; i < size; i += block_size) {
        for (int j = 0; j < size; j += block_size) {
            for (int k = 0; k < size; k += block_size) {
                for (int ii = i; ii < i + block_size && ii < size; ++ii) {
                    for (int jj = j; jj < j + block_size && jj < size; ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < k + block_size && kk < size; ++kk) {
                            sum += a[ii * size + kk] * b[kk * size + jj];
                        }
                        res[ii * size + jj] += sum;
                    }
                }
            }
        }
    }
}

// void matrix_mult2(double *a, double *b, double *res, size_t size) {
//     const int block_size = 32;
//     #pragma omp parallel for
//     for (int i = 0; i < size; i += block_size) {
//         for (int j = 0; j < size; j += block_size) {
//             for (int k = 0; k < size; k += block_size) {
//                 for (int ii = i; ii < i + block_size && ii < size; ++ii) {
//                     for (int jj = j; jj < j + block_size && jj < size; ++jj) {
//                         __m256d rres = _mm256_set1_pd(0.0); // Регистр для накопления суммы
//                         for (int kk = k; kk < k + block_size && kk < size; kk += 4) {
//                             // Загружаем 4 элемента из строк матрицы a и b
//                             __m256d ra = _mm256_loadu_pd(a + ii * size + kk);
//                             __m256d rb = _mm256_loadu_pd(b + kk * size + jj);
//                             // Умножаем и аккумулируем
//                             rres = _mm256_fmadd_pd(ra, rb, rres);
//                         }

//                         // Суммируем элементы из регистра rres
//                         double sum[4];
//                         _mm256_storeu_pd(sum, rres);
//                         double scalar_sum = sum[0] + sum[1] + sum[2] + sum[3];

//                         // Учитываем остаток, если блок не кратен 4
//                         for (int kk = k + (block_size / 4) * 4; kk < k + block_size && kk < size; ++kk) {
//                             scalar_sum += a[ii * size + kk] * b[kk * size + jj];
//                         }

//                         res[ii * size + jj] += scalar_sum;
//                     }
//                 }
//             }
//         }
//     }
// }


// void matrix_mult_avx(double *a, double *b, double *res, size_t size) {
//     #pragma omp parallel for
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; j < size; j++) {
//             __m256d sum = _mm256_setzero_pd(); // Vectorized sum for each element in result

//             for (int k = 0; k < size; k++) {
//                 // Load a row from matrix 'a' and a column from matrix 'b'
//                 __m256d a_vals = _mm256_set1_pd(a[i * size + k]); // Replicate a[i, k] across the vector
//                 __m256d b_vals = _mm256_loadu_pd(&b[k * size + j]); // Load the j-th column of b

//                 // Multiply the elements (dot product of row i from a and column j from b)
//                 __m256d product = _mm256_mul_pd(a_vals, b_vals);

//                 // Add to the sum
//                 sum = _mm256_add_pd(sum, product);
//             }

//             // Horizontal add to get the final sum for res[i*size + j]
//             double tmp[4];
//             _mm256_storeu_pd(tmp, sum);
//             res[i * size + j] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
//         }
//     }
// }

// void matrix_mult2(double *a, double *b, double *res, size_t size) {
//     const size_t block_size = 64; // Подберите оптимальное значение для вашей архитектуры.

//     #pragma omp parallel for collapse(2) schedule(dynamic)
//     for (size_t ii = 0; ii < size; ii += block_size) {
//         for (size_t jj = 0; jj < size; jj += block_size) {
//             for (size_t kk = 0; kk < size; kk += block_size) {
//                 for (size_t i = ii; i < ii + block_size && i < size; i++) {
//                     for (size_t j = jj; j < jj + block_size && j < size; j++) {
//                         __m256d sum_vec = _mm256_setzero_pd(); // Инициализация суммы.
//                         size_t k;
//                         for (k = kk; k + 4 <= kk + block_size && k + 4 <= size; k += 4) {
//                             // Загрузка данных из A и B.
//                             __m256d a_vec = _mm256_loadu_pd(&a[i * size + k]);
//                             __m256d b_vec = _mm256_set_pd(
//                                 b[(k + 3) * size + j],
//                                 b[(k + 2) * size + j],
//                                 b[(k + 1) * size + j],
//                                 b[k * size + j]
//                             );
//                             // Умножение и аккумулирование.
//                             sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
//                         }
//                         // Сохранение сумм в скалярную переменную.
//                         double sum_array[4];
//                         _mm256_storeu_pd(sum_array, sum_vec);
//                         double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
//                         // Обработка оставшихся элементов.
//                         for (; k < kk + block_size && k < size; k++) {
//                             sum += a[i * size + k] * b[k * size + j];
//                         }
//                         res[i * size + j] += sum;
//                     }
//                 }
//             }
//         }
//     }
// }

int main()
{
	double *mat, *mat_mkl, *a, *b, *a_mkl, *b_mkl;
    double *mat2, *a2, *b2;
	size_t size = 1000;
	chrono::time_point<chrono::system_clock> start, end;

	mat = new double[size * size];
	a = new double[size * size];
	b = new double[size * size];
	generation(a, size);
	generation(b, size);
    mat2 = new double[size * size];
	a2 = new double[size * size];
	b2 = new double[size * size];
	memcpy(a2, a, sizeof(double)*size*size);
    memcpy(b2, b, sizeof(double)*size*size);
	memset(mat, 0, size*size * sizeof(double));
    memset(mat2, 0, size*size * sizeof(double));

#ifdef MKL     
    mat_mkl = new double[size * size];
	a_mkl = new double[size * size];
	b_mkl = new double[size * size];
	memcpy(a_mkl, a, sizeof(double)*size*size);
    memcpy(b_mkl, b, sizeof(double)*size*size);
	memset(mat_mkl, 0, size*size * sizeof(double));
#endif

	start = chrono::system_clock::now();
	matrix_mult(a, b, mat, size);
	end = chrono::system_clock::now();
    
   
	int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds/1000.0 << " sec" << endl;

		start = chrono::system_clock::now();
	matrix_mult2(a2, b2, mat2, size);
	end = chrono::system_clock::now();
    
   
	 elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds/1000.0 << " sec" << endl;


        int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if(abs(mat[i] - mat2[i]) > size*1e-14){
		    flag = 1;
			std::cout<<mat[i]<<" "<<mat2[i]<<std::endl;
        }
    if (flag)
        cout << "fail" << endl;
    else
        cout << "correct" << endl; 
    

#ifdef MKL 
	start = chrono::system_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
    end = chrono::system_clock::now();
    
    elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time mkl: " << elapsed_seconds/1000.0 << " sec" << endl;
     
    int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if(abs(mat[i] - mat_mkl[i]) > size*1e-14){
		    flag = 1;
        }
    if (flag)
        cout << "fail" << endl;
    else
        cout << "correct" << endl; 
    
    delete (a_mkl);
    delete (b_mkl);
    delete (mat_mkl);
#endif

    delete (a);
    delete (b);
    delete (mat);

	delete (a2);
    delete (b2);
    delete (mat2);

	system("pause");
	
	return 0;
}
