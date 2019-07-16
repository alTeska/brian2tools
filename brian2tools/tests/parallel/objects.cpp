
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
Network magicnetwork;
Network network;

//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_monitor__indices;
const int _num__array_monitor__indices = 2;
int32_t * _array_monitor_N;
const int _num__array_monitor_N = 1;
double * _array_monitor_v;
const int _num__array_monitor_v = (0, 2);
int32_t * _array_neurons__spikespace;
const int _num__array_neurons__spikespace = 3;
double * _array_neurons_C;
const int _num__array_neurons_C = 2;
double * _array_neurons_gL;
const int _num__array_neurons_gL = 2;
int32_t * _array_neurons_i;
const int _num__array_neurons_i = 2;
double * _array_neurons_v;
const int _num__array_neurons_v = 2;

//////////////// dynamic arrays 1d /////////
std::vector<double> _dynamic_array_monitor_t;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> _dynamic_array_monitor_v;

/////////////// static arrays /////////////
int32_t * _static_array__array_monitor__indices;
const int _num__static_array__array_monitor__indices = 2;
double * _static_array__array_neurons_C;
const int _num__static_array__array_neurons_C = 1;
double * _static_array__array_neurons_gL;
const int _num__static_array__array_neurons_gL = 1;
double * _timedarray_values;
const int _num__timedarray_values = 2994;

//////////////// synapses /////////////////

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_monitor__indices = new int32_t[2];
    
	for(int i=0; i<2; i++) _array_monitor__indices[i] = 0;

	_array_monitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_monitor_N[i] = 0;

	_array_neurons__spikespace = new int32_t[3];
    
	for(int i=0; i<3; i++) _array_neurons__spikespace[i] = 0;

	_array_neurons_C = new double[2];
    
	for(int i=0; i<2; i++) _array_neurons_C[i] = 0;

	_array_neurons_gL = new double[2];
    
	for(int i=0; i<2; i++) _array_neurons_gL[i] = 0;

	_array_neurons_i = new int32_t[2];
    
	for(int i=0; i<2; i++) _array_neurons_i[i] = 0;

	_array_neurons_v = new double[2];
    
	for(int i=0; i<2; i++) _array_neurons_v[i] = 0;


	// Arrays initialized to an "arange"
	_array_neurons_i = new int32_t[2];
    
	for(int i=0; i<2; i++) _array_neurons_i[i] = 0 + i;


	// static arrays
	_static_array__array_monitor__indices = new int32_t[2];
	_static_array__array_neurons_C = new double[1];
	_static_array__array_neurons_gL = new double[1];
	_timedarray_values = new double[2994];

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_monitor__indices;
	f_static_array__array_monitor__indices.open("static_arrays/_static_array__array_monitor__indices", ios::in | ios::binary);
	if(f_static_array__array_monitor__indices.is_open())
	{
		f_static_array__array_monitor__indices.read(reinterpret_cast<char*>(_static_array__array_monitor__indices), 2*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_monitor__indices." << endl;
	}
	ifstream f_static_array__array_neurons_C;
	f_static_array__array_neurons_C.open("static_arrays/_static_array__array_neurons_C", ios::in | ios::binary);
	if(f_static_array__array_neurons_C.is_open())
	{
		f_static_array__array_neurons_C.read(reinterpret_cast<char*>(_static_array__array_neurons_C), 1*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurons_C." << endl;
	}
	ifstream f_static_array__array_neurons_gL;
	f_static_array__array_neurons_gL.open("static_arrays/_static_array__array_neurons_gL", ios::in | ios::binary);
	if(f_static_array__array_neurons_gL.is_open())
	{
		f_static_array__array_neurons_gL.read(reinterpret_cast<char*>(_static_array__array_neurons_gL), 1*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurons_gL." << endl;
	}
	ifstream f_timedarray_values;
	f_timedarray_values.open("static_arrays/_timedarray_values", ios::in | ios::binary);
	if(f_timedarray_values.is_open())
	{
		f_timedarray_values.read(reinterpret_cast<char*>(_timedarray_values), 2994*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _timedarray_values." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_1771757190631946187", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_-7133266770088703665", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-5890343775314378696", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_monitor__indices;
	outfile__array_monitor__indices.open("results/_array_monitor__indices_473105479306919410", ios::binary | ios::out);
	if(outfile__array_monitor__indices.is_open())
	{
		outfile__array_monitor__indices.write(reinterpret_cast<char*>(_array_monitor__indices), 2*sizeof(_array_monitor__indices[0]));
		outfile__array_monitor__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_monitor__indices." << endl;
	}
	ofstream outfile__array_monitor_N;
	outfile__array_monitor_N.open("results/_array_monitor_N_2935745448265660719", ios::binary | ios::out);
	if(outfile__array_monitor_N.is_open())
	{
		outfile__array_monitor_N.write(reinterpret_cast<char*>(_array_monitor_N), 1*sizeof(_array_monitor_N[0]));
		outfile__array_monitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_monitor_N." << endl;
	}
	ofstream outfile__array_neurons__spikespace;
	outfile__array_neurons__spikespace.open("results/_array_neurons__spikespace_-9189510784890083110", ios::binary | ios::out);
	if(outfile__array_neurons__spikespace.is_open())
	{
		outfile__array_neurons__spikespace.write(reinterpret_cast<char*>(_array_neurons__spikespace), 3*sizeof(_array_neurons__spikespace[0]));
		outfile__array_neurons__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurons__spikespace." << endl;
	}
	ofstream outfile__array_neurons_C;
	outfile__array_neurons_C.open("results/_array_neurons_C_-7442981317025764055", ios::binary | ios::out);
	if(outfile__array_neurons_C.is_open())
	{
		outfile__array_neurons_C.write(reinterpret_cast<char*>(_array_neurons_C), 2*sizeof(_array_neurons_C[0]));
		outfile__array_neurons_C.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurons_C." << endl;
	}
	ofstream outfile__array_neurons_gL;
	outfile__array_neurons_gL.open("results/_array_neurons_gL_-6799974216313613454", ios::binary | ios::out);
	if(outfile__array_neurons_gL.is_open())
	{
		outfile__array_neurons_gL.write(reinterpret_cast<char*>(_array_neurons_gL), 2*sizeof(_array_neurons_gL[0]));
		outfile__array_neurons_gL.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurons_gL." << endl;
	}
	ofstream outfile__array_neurons_i;
	outfile__array_neurons_i.open("results/_array_neurons_i_-1487134892286366806", ios::binary | ios::out);
	if(outfile__array_neurons_i.is_open())
	{
		outfile__array_neurons_i.write(reinterpret_cast<char*>(_array_neurons_i), 2*sizeof(_array_neurons_i[0]));
		outfile__array_neurons_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurons_i." << endl;
	}
	ofstream outfile__array_neurons_v;
	outfile__array_neurons_v.open("results/_array_neurons_v_8914526473608568583", ios::binary | ios::out);
	if(outfile__array_neurons_v.is_open())
	{
		outfile__array_neurons_v.write(reinterpret_cast<char*>(_array_neurons_v), 2*sizeof(_array_neurons_v[0]));
		outfile__array_neurons_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurons_v." << endl;
	}

	ofstream outfile__dynamic_array_monitor_t;
	outfile__dynamic_array_monitor_t.open("results/_dynamic_array_monitor_t_-6479526473913045", ios::binary | ios::out);
	if(outfile__dynamic_array_monitor_t.is_open())
	{
        if (! _dynamic_array_monitor_t.empty() )
        {
			outfile__dynamic_array_monitor_t.write(reinterpret_cast<char*>(&_dynamic_array_monitor_t[0]), _dynamic_array_monitor_t.size()*sizeof(_dynamic_array_monitor_t[0]));
		    outfile__dynamic_array_monitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_monitor_t." << endl;
	}

	ofstream outfile__dynamic_array_monitor_v;
	outfile__dynamic_array_monitor_v.open("results/_dynamic_array_monitor_v_1929057591490184532", ios::binary | ios::out);
	if(outfile__dynamic_array_monitor_v.is_open())
	{
        for (int n=0; n<_dynamic_array_monitor_v.n; n++)
        {
            if (! _dynamic_array_monitor_v(n).empty())
            {
                outfile__dynamic_array_monitor_v.write(reinterpret_cast<char*>(&_dynamic_array_monitor_v(n, 0)), _dynamic_array_monitor_v.m*sizeof(_dynamic_array_monitor_v(0, 0)));
            }
        }
        outfile__dynamic_array_monitor_v.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_monitor_v." << endl;
	}
	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
}

void _dealloc_arrays()
{
	using namespace brian;


	// static arrays
	if(_static_array__array_monitor__indices!=0)
	{
		delete [] _static_array__array_monitor__indices;
		_static_array__array_monitor__indices = 0;
	}
	if(_static_array__array_neurons_C!=0)
	{
		delete [] _static_array__array_neurons_C;
		_static_array__array_neurons_C = 0;
	}
	if(_static_array__array_neurons_gL!=0)
	{
		delete [] _static_array__array_neurons_gL;
		_static_array__array_neurons_gL = 0;
	}
	if(_timedarray_values!=0)
	{
		delete [] _timedarray_values;
		_timedarray_values = 0;
	}
}

