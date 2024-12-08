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