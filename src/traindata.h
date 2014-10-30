#pragma once
#include "stdh.h"

class Traindata{
public:
	int N;		/**< number of instances */
	int M;		/**< feature vector dimension */
	T *d;			/**< data pointer to all the feature vectors NxM */
	T *l;			/**< labels for every instance*/
	T2 *w;		/**< weights for every instance */	

	Traindata();
	~Traindata();
	void loadData(const char* fname);
	void loadDataBinary(const char* fname, int dim);
	void normalizeData(Traindata* tdata);
};