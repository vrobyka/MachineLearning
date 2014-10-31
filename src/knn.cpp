#include "knn.h"

Knn::Knn():Knn(3){
}

Knn::Knn(int KK){
	K = KK;
}

Knn::~Knn(){
}

void Knn::saveModel(const char* fname){
}

void Knn::loadModel(const char* fname){
}

T Knn::train(){
	trainingdata = data;
	return 0;
}

T distL2(int n, T* x, T* y){
	T res = 0;
	for (int i = 0; i < n; i++)
		res += sqr(x[i] - y[i]);
	return res;
}

T distL1(int n, T* x, T* y){
	T res = 0;
	for (int i = 0; i < n; i++)
		res += abs(x[i] - y[i]);
	return res;
}

T Knn::predict(T* sample){	

	int nrlabels = 2;
	T* histvotes = new float[nrlabels];
	T labels[2] = { -1, 1 };

	T (*distX)(int,T*,T*) = &distL2;
	vector<Pair> dists;	
	for (int i = 0, i2 = 0; i < trainingdata->N; i++, i2 += trainingdata->M)
	{		
		if (trainingdata->w[i] != 0) //only use training samples
		{
			T dist = (*distX)(trainingdata->M, sample, trainingdata->d + i2);
			dists.push_back(Pair(dist, i));
		}
	}

	sort(dists.begin(), dists.end(), dists[0]);
	memset(histvotes, 0, nrlabels*sizeof(T));
	for (int k = 0; k < K; k++){
		T lbl = data->l[dists[k].i];
		//
		int index = 0;
		if (lbl == 1)
			index = 1;
		//
		histvotes[index]++;
	}

	int maxi = 0;
	for (int i = 1; i < nrlabels; i++)
		if (histvotes[i]>histvotes[maxi])
			maxi = i;
	delete[] histvotes;

	return labels[maxi];
}