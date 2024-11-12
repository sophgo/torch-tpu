#include <stdlib.h>
#include <math.h>
#include "nodechip_utils.h"
#include "tpu_kernel.h"

// 函数用于比较两个整数，用于qsort排序
static int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

int findAllDivisors(unsigned long n , int *factors, unsigned int sizes)
{
    unsigned int count = 0;

    // 遍历到平方根
    for (int i = 1; i <= sqrt(n); i++) {
        if (n % i == 0) {
            factors[count++] = i;
            if ((unsigned int)i != n / i) {  // 避免平方根被重复添加
                factors[count++] = n / i;
            }
        }
    }
    TPUKERNEL_ASSERT(count < sizes );
    // 对结果进行排序
    qsort(factors, count, sizeof(int), compare);
    return count;
}

int findFactorsIndex(int factor, int *factors)
{
    int idx = 0;
    while(factors[idx] <= factor) {
        idx++;
    }
    --idx;
    return idx;
}