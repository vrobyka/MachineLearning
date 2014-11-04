#pragma once
#include "neuron.h"
#include "classifier.h"
#include "util.h"

class Neural : public Classifier{
public:
	int nrlayers;
	int* neurons_in_layers;
	Neuron** ns; //neurons for each position enable the usage of different actication functions
	T** X; //inputs for each neuron
	T** Y; //outputs for each neuron
	T** G; //gradient values for each neuron
	Mat<T>* W; //weight/transition matrix from adjecent layers
	Neural();
	Neural(int nrlayers, int* neurons_in_layers); // first layer is the number of inputs, last layer is the nr of outputs
	~Neural();

	T predict(T* Xin) override; //feedforward
	void backpropagate(T truelabel);

	T train() override;
	void saveModel(const char* fname) override;
	void loadModel(const char* fname) override;
};