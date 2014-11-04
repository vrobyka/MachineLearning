#include "neuron.h"

//todo cache exp values to avoid calculating

 T Neuron::getFunc(T x){
	return 1.f / (1.f + exp(-x));
}

 T Neuron::getDeriv(T x){
	T expx = exp(-x);
	return expx / sqr(1.f + expx);
}

Neuron::Neuron(){
}

Neuron::~Neuron(){
}