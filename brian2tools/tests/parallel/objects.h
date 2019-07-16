
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>


namespace brian {

// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;
extern Network network;

//////////////// dynamic arrays ///////////
extern std::vector<double> _dynamic_array_monitor_t;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_monitor__indices;
extern const int _num__array_monitor__indices;
extern int32_t *_array_monitor_N;
extern const int _num__array_monitor_N;
extern double *_array_monitor_v;
extern const int _num__array_monitor_v;
extern int32_t *_array_neurons__spikespace;
extern const int _num__array_neurons__spikespace;
extern double *_array_neurons_C;
extern const int _num__array_neurons_C;
extern double *_array_neurons_gL;
extern const int _num__array_neurons_gL;
extern int32_t *_array_neurons_i;
extern const int _num__array_neurons_i;
extern double *_array_neurons_v;
extern const int _num__array_neurons_v;

//////////////// dynamic arrays 2d /////////
extern DynamicArray2D<double> _dynamic_array_monitor_v;

/////////////// static arrays /////////////
extern int32_t *_static_array__array_monitor__indices;
extern const int _num__static_array__array_monitor__indices;
extern double *_static_array__array_neurons_C;
extern const int _num__static_array__array_neurons_C;
extern double *_static_array__array_neurons_gL;
extern const int _num__static_array__array_neurons_gL;
extern double *_timedarray_values;
extern const int _num__timedarray_values;

//////////////// synapses /////////////////

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


