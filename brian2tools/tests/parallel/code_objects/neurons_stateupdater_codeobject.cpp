#include "objects.h"
#include "code_objects/neurons_stateupdater_codeobject.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace {
 	
 static double* _namespace_timedarray_values;
 static inline double _timedarray(const double t, const int i)
 {
     const double epsilon = 0.000010000000000000 / 8;
     if (i < 0 || i >= 2)
         return NAN;
     int timestep = (int)((t/epsilon + 0.5)/8);
     if(timestep < 0)
        timestep = 0;
     else if(timestep >= 1497)
         timestep = 1497-1;
     return _namespace_timedarray_values[timestep*2 + i];
 }
 template < typename T1, typename T2 > struct _higher_type;
 template < > struct _higher_type<int,int> { typedef int type; };
 template < > struct _higher_type<int,long> { typedef long type; };
 template < > struct _higher_type<int,long long> { typedef long long type; };
 template < > struct _higher_type<int,float> { typedef float type; };
 template < > struct _higher_type<int,double> { typedef double type; };
 template < > struct _higher_type<int,long double> { typedef long double type; };
 template < > struct _higher_type<long,int> { typedef long type; };
 template < > struct _higher_type<long,long> { typedef long type; };
 template < > struct _higher_type<long,long long> { typedef long long type; };
 template < > struct _higher_type<long,float> { typedef float type; };
 template < > struct _higher_type<long,double> { typedef double type; };
 template < > struct _higher_type<long,long double> { typedef long double type; };
 template < > struct _higher_type<long long,int> { typedef long long type; };
 template < > struct _higher_type<long long,long> { typedef long long type; };
 template < > struct _higher_type<long long,long long> { typedef long long type; };
 template < > struct _higher_type<long long,float> { typedef float type; };
 template < > struct _higher_type<long long,double> { typedef double type; };
 template < > struct _higher_type<long long,long double> { typedef long double type; };
 template < > struct _higher_type<float,int> { typedef float type; };
 template < > struct _higher_type<float,long> { typedef float type; };
 template < > struct _higher_type<float,long long> { typedef float type; };
 template < > struct _higher_type<float,float> { typedef float type; };
 template < > struct _higher_type<float,double> { typedef double type; };
 template < > struct _higher_type<float,long double> { typedef long double type; };
 template < > struct _higher_type<double,int> { typedef double type; };
 template < > struct _higher_type<double,long> { typedef double type; };
 template < > struct _higher_type<double,long long> { typedef double type; };
 template < > struct _higher_type<double,float> { typedef double type; };
 template < > struct _higher_type<double,double> { typedef double type; };
 template < > struct _higher_type<double,long double> { typedef long double type; };
 template < > struct _higher_type<long double,int> { typedef long double type; };
 template < > struct _higher_type<long double,long> { typedef long double type; };
 template < > struct _higher_type<long double,long long> { typedef long double type; };
 template < > struct _higher_type<long double,float> { typedef long double type; };
 template < > struct _higher_type<long double,double> { typedef long double type; };
 template < > struct _higher_type<long double,long double> { typedef long double type; };
 template < typename T1, typename T2 >
 static inline typename _higher_type<T1,T2>::type
 _brian_mod(T1 x, T2 y)
 {{
     return x-y*floor(1.0*x/y);
 }}
 template < typename T1, typename T2 >
 static inline typename _higher_type<T1,T2>::type
 _brian_floordiv(T1 x, T2 y)
 {{
     return floor(1.0*x/y);
 }}
 #ifdef _MSC_VER
 #define _brian_pow(x, y) (pow((double)(x), (y)))
 #else
 #define _brian_pow(x, y) (pow((x), (y)))
 #endif

}

////// HASH DEFINES ///////



void _run_neurons_stateupdater_codeobject()
{
	using namespace brian;


	///// CONSTANTS ///////////
	const int _numt = 1;
const int _numi = 2;
const int _numC = 2;
const int _numv = 2;
const int _numgL = 2;
const int _numdt = 1;
	///// POINTERS ////////////
 	
 double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
 int32_t* __restrict  _ptr_array_neurons_i = _array_neurons_i;
 double* __restrict  _ptr_array_neurons_C = _array_neurons_C;
 double* __restrict  _ptr_array_neurons_v = _array_neurons_v;
 double* __restrict  _ptr_array_neurons_gL = _array_neurons_gL;
 double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
 _namespace_timedarray_values = _timedarray_values;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const double t = _ptr_array_defaultclock_t[0];
 const double dt = _ptr_array_defaultclock_dt[0];
 const double _lio_1 = 144009798674.772 * 0.001;
 const double _lio_2 = 1.0f*0.5/0.001;
 const double _lio_3 = 70.0 * 0.001;
 const double _lio_4 = - dt;


	const int _N = 2;
	
	for(int _idx=0; _idx<_N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
                
        const int32_t i = _ptr_array_neurons_i[_idx];
        const double C = _ptr_array_neurons_C[_idx];
        double v = _ptr_array_neurons_v[_idx];
        const double gL = _ptr_array_neurons_gL[_idx];
        const double _BA_v = 1.0f*((- C) * (((1.0f*(_lio_1 * (gL * exp(_lio_2 * v)))/C) + (1.0f*_timedarray(t, _brian_mod(i, 2))/C)) - (1.0f*(_lio_3 * gL)/C)))/gL;
        const double _v = (- _BA_v) + ((_BA_v + v) * exp(1.0f*(_lio_4 * gL)/C));
        v = _v;
        _ptr_array_neurons_v[_idx] = v;

	}

}


