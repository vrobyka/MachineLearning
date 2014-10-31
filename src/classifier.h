#pragma once
#include "data.h"

class Classifier{
protected:
	Data* data;	
public:
	Classifier();
	~Classifier();

	virtual void saveModel(const char* fname) = 0;
	virtual void loadModel(const char* fname) = 0;

	void setData(Data* td);	
	virtual T train() = 0; /** Training procedure with data already loaded.*/	
	virtual T predict(T* sample) = 0; /** Predict using an existing model.*/

	T calcError(); /** Calculate the classification error on the currently loaded dataset.*/	
	T crossvalidation(int nfold, T* errs); /** Perform nfold training and evaluation on the currently loaded dataset.*/
	T randsplit(float percentTraining); /** Split the currently loaded dataset into training and testing and perform test.*/
	void speedTest(); /** Evaluate the speed of the classifier on the currently loaded dataset.*/
private:
	
};