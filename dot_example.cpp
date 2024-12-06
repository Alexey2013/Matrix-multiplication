#include <immintrin.h>
#include <stdio.h>

double dot(double * a, double * b, int num) {
    double res = 0.0;
    int i = 0;
    __m256 ra, rb, rres;
    rres = _mm256_set1_pd(0.0); 
    for(i = 0; i < num - 3; i+=4)
    {
        ra = _mm256_loadu_pd(a + i);
        rb = _mm256_loadu_pd(b + i);
        rres = _mm256_fmadd_pd(ra, rb, rres);
    }
    for(; i < num; i++){
        res += a[i]*b[i];
    }
    res += rres[0];
    res += rres[1];
    res += rres[2];
    res += rres[3];
    
    return res;
}