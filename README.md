# "cubit" - A CUDA Header-Only, In-Place, Non-Power-of-Two cudaBitonic sorter

For some CUDA large-BVH-construction project I recently needed a
reasonably fast sorter that would be able to sort "in place", without
requiring extra temp memory or extra copies of the keys and/or values.

Since bitonic sort fulfills those requirements I first looked for an
existing bitonic sorter, but couldn't find any that worked on
arbitrary data and, in particular, arbitrary (i.e., non-power of two)
data sizes, so eventually wrote my own... and since I thought this
might be useful for others, too, I decided to provide it here on
github.

# Usage

Project should build with cmake as usual; sort code itself is a
header-only and only requires including `cubit/cubit.h`, then calling
`cubit::sort(d_keys,numKeys)` for key-only, and
`cubit::sort(d_keys,d_values,numKeys)` for key-value sort. In both
cases pointers are device-pointers. 

Both key-only and key-value variants are templated over the key and/or
value type; all that is requires is that the key type has a
`operator<` defined over them, and that there is a function
`cubit::max_value<KeyT>()` that returns a type that is never `<` than
any of the inputs values.

For an example usage, look to `testing/test.cu`, which is also used
for verifying correctness.

# Performance

As can be expected, bitonic sort will create more memory traffic than
a radix sorted; and not surprisingly, though this code is quite
competitive for moderately sized arrays (say, up to 10 million
elements), for large arrays the `cub` radix sorter that comes with
CUDA will be significantly faster than this code (around 5-15x for
really large arrays). However, unlike cub this codebase will sort
entirely 'in place', and not require temp mem or a second copy for the
output array.






