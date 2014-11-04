#include "neural.h"

Neural::Neural(){
	ns = 0; W = 0; X = 0; Y = 0; G = 0; neurons_in_layers = 0;
}

Neural::Neural(int nrlayers_, int* neurons_in_layers_){
	nrlayers = nrlayers_;
	neurons_in_layers = new int[nrlayers];
	memcpy(neurons_in_layers, neurons_in_layers_, nrlayers*sizeof(T));

	ns = new Neuron*[nrlayers];	
	for (int i = 0; i < nrlayers; i++)
		ns[i] = new Neuron[neurons_in_layers[i]];
	
	W = new Mat<T>[nrlayers - 1];
	for (int i = 0; i < nrlayers-1; i++)
	{
		W[i].width = neurons_in_layers[i]+1; //additional for constant term
		W[i].height = neurons_in_layers[i+1];
		W[i].data = new T[W[i].width*W[i].height];
	}

	X = new T*[nrlayers];
	for (int i = 0; i < nrlayers; i++)
	{
		X[i] = new T[neurons_in_layers[i]+1]; 	//additional for constant term
		X[i][0] = 1;//a dummy to keep notation consistent(it only adds nrlayer unused Ts)
	}

	Y = new T*[nrlayers];
	for (int i = 0; i < nrlayers; i++)
	{
		Y[i] = new T[neurons_in_layers[i] + 1]; //additional for constant term
		Y[i][0] = 1;
	}

	G = new T*[nrlayers];
	for (int i = 0; i < nrlayers; i++)
	{
		G[i] = new T[neurons_in_layers[i] + 1]; //additional for constant term
		G[i][0] = 0;
	}
}

template <class U>
void deleteptr(U* X, int nrlayers){
	if (X!=0)
	{
		for (int i = 0; i < nrlayers; i++)
			delete[] X[i];
		delete X;
	}
}

Neural::~Neural(){
	deleteptr(ns, nrlayers); 
	deleteptr(X, nrlayers);
	deleteptr(Y, nrlayers);
	deleteptr(G, nrlayers);
	if (W) delete[] W;
	if (neurons_in_layers) delete[] neurons_in_layers;
}

T Neural::predict(T* Xin){
	memcpy(Y[0] + 1, Xin, neurons_in_layers[0] * sizeof(T));
	for (int l = 0; l < nrlayers - 1; l++)
	{
		memset(X[l+1] + 1, 0, neurons_in_layers[l + 1] * sizeof(T));
		const int ww = neurons_in_layers[l] + 1;
		//X^(l+1) = W*Y^(l) - can be optimized with better matrix multiplication algorithm
		for (int j = 0; j < neurons_in_layers[l+1]; j++){
			for (int i = 0; i < neurons_in_layers[l]+1; i++){			
				X[l + 1][j+1] += W[l].data[j*ww + i] * Y[l][i];
			}
			Y[l + 1][j+1] = ns[l + 1][j].getFunc(X[l + 1][j+1]);
		}
	}
	return Y[nrlayers - 1][1];
}

void Neural::backpropagate(T truelabel){
	//last layer
	T err = Y[nrlayers - 1][1] - truelabel; 	
	G[nrlayers - 1][1] = err* ns[nrlayers - 1][1].getDeriv(X[nrlayers - 1][1]);

	//inner layers
	for (int l = nrlayers - 2; l >= 0; l--)
	{
		for (int i = 0; i < neurons_in_layers[l]+1; i++){
			G[l][i] = 0;
			const int ww = neurons_in_layers[l] + 1;
			for (int j = 0; j < neurons_in_layers[l+1]; j++){
				G[l][i] += G[l+1][j]*W[l].data[i*ww+j];									
			}
			G[l][i] *= ns[l][i].getDeriv(X[l][i]);
		}		
	}

	//gradient descent step
	const float alpha = 0.1f;
	for (int l = nrlayers - 1; l >= 0; l--){
		for (int i = 0; i < neurons_in_layers[l]; i++){
			const int ww = neurons_in_layers[l] + 1;
			for (int j = 0; j < neurons_in_layers[l + 1]; j++){
				W[l].data[i*ww + j] -= X[l][i] * alpha* G[l][j];
			}
		}
	}
}

T Neural::train(){
	//init - randomize weights
	srand(0);
	for (int l = 0; l < nrlayers - 1; l++)
	{
		for (int i = 0; i < (neurons_in_layers[l]+1) * neurons_in_layers[l + 1]; i++){
			W[l].data[i] = -1.f+rand()*2.f / RAND_MAX;
		}
	}

	int nrtrials = 1;
	for (int t = 0; t < nrtrials; t++)
	{
		T* descr = data->d;
		for (int i = 0; i < data->N; i++){
			T dd[3] = { 3, -1, 1 };
			predict(dd);
			backpropagate(7);
			/*
			predict(descr);
			backpropagate(data->l[i]);
			*/
			descr += data->M;
		}
	}

	return 1;
}

void Neural::saveModel(const char* fname){

}

void Neural::loadModel(const char* fname){

}