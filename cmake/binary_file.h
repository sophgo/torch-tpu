
extern "C"
{
    extern const char _binary_${name}_start;
    extern const char _binary_${name}_end;
}

#define ${name} (&_binary_${name}_start)
#define ${name}_length (&_binary_${name}_end - &_binary_${name}_start)
