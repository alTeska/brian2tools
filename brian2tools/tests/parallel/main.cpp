#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

#include "code_objects/monitor_codeobject.h"
#include "code_objects/neurons_resetter_codeobject.h"
#include "code_objects/neurons_stateupdater_codeobject.h"
#include "code_objects/neurons_thresholder_codeobject.h"


#include <iostream>
#include <fstream>




int main(int argc, char **argv)
{
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 1e-05;
        
                        
                        for(int i=0; i<_num__array_neurons_v; i++)
                        {
                            _array_neurons_v[i] = - 0.07;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_monitor__indices; i++)
                        {
                            _array_monitor__indices[i] = _static_array__array_monitor__indices[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurons_C; i++)
                        {
                            _array_neurons_C[i] = _static_array__array_neurons_C[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurons_gL; i++)
                        {
                            _array_neurons_gL[i] = _static_array__array_neurons_gL[i];
                        }
                        
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        network.clear();
        network.add(&defaultclock, _run_monitor_codeobject);
        network.add(&defaultclock, _run_neurons_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurons_thresholder_codeobject);
        network.add(&defaultclock, _run_neurons_resetter_codeobject);
        network.run(0.01497, NULL, 10.0);

	}
        

	brian_end();
        

	return 0;
}