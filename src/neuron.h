#pragma once 
#include "stdh.h"

typedef float T;

class Neuron{
public:
	Neuron();
	~Neuron();
	T getFunc(T x);
	T getDeriv(T x);
};

