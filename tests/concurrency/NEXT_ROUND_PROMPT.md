Use interlace to find bugs in the LIBRARY CODE ITSELF for [library name].
Do not write intentionally buggy application code.

Approach:
1. Read the library's source code to understand its internals and identify
   areas where thread-safety or ordering might be fragile.
2. Write tests that stress those specific internal code paths - concurrent
   calls to the library's own APIs that should be safe according to its
   contract.
3. Run through interlace with many schedules to see if the library's own
   synchronization has gaps.
4. Any failure is a real bug in the library worth reporting upstream.

Focus on internal data structures (registries, pools, caches), lifecycle
operations (init/shutdown races), and state transitions that the library
manages on behalf of the user.
