PROGRAM = main

SRCS = objects.cpp main.cpp code_objects/monitor_codeobject.cpp code_objects/neurons_stateupdater_codeobject.cpp code_objects/neurons_thresholder_codeobject.cpp code_objects/neurons_resetter_codeobject.cpp network.cpp synapses_classes.cpp run.cpp brianlib/randomkit/randomkit.c
H_SRCS = objects.h code_objects/monitor_codeobject.h code_objects/neurons_stateupdater_codeobject.h code_objects/neurons_thresholder_codeobject.h code_objects/neurons_resetter_codeobject.h network.h synapses_classes.h run.h brianlib/common_math.h brianlib/clocks.h brianlib/dynamic_array.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
CC = @g++
OPTIMISATIONS = -Ibrianlib/randomkit -I/home/alteska/anaconda3/envs/brian_opt/include -w -O3 -ffast-math -fno-finite-math-only -march=native
CFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -Lbrianlib/randomkit -L/home/alteska/anaconda3/envs/brian_opt/lib -Wl,-R/home/alteska/anaconda3/envs/brian_opt/lib 
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CC) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CC) $(CFLAGS) $< -o $@