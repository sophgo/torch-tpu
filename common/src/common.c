#ifdef _WIN32
#include "..\include_win\common_win.h"
#else
#include "common.h"
#endif

#ifdef __x86_64__
#include <execinfo.h>
void print_trace() {
  void* array[10];
  char** strings = NULL;

  int size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  printf("Obtained %d stack frames.\n", size);

  for (int i = 0; i < size; i++) {
    printf("%s\n", strings[i]);
  }
  if (strings) free(strings);
}
#else
void print_trace() {}
#endif