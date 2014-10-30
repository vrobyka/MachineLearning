#include "traindata.h"
#include "util.h"

#define RED 1
#define MAXNRSAMPLES 10000000//10000000

Traindata::Traindata(){
	d = 0; l = 0; w = 0;
}

Traindata::~Traindata(){
	if (d) delete[] d;
	if (l) delete[] l;
	if (w) delete[] w;
}

void Traindata::loadData(const char* fname){
	FILE* f = fopen2(fname, "r");

	//read training data from a csv file where the first column is the binary class label 1/-1
	printf("Loading training data\n");

	//find nr of features by reading the first instance
	int nrfeatures = 0, lbl; T tmp;
	fscanf(f, "%d", &lbl);
	while (fscanf(f, ",%f", &tmp) == 1) 	nrfeatures++;
	fseek(f, 0, SEEK_END);
	int file_size = ftell(f);
	rewind(f);

	w = new T2[MAXNRSAMPLES];
	l = new T[MAXNRSAMPLES];
	d = new T[MAXNRSAMPLES*nrfeatures];
	T* descr = d;

	int nrsamples = 0;


	const T2 POSW = 1.0;
	//read the training data while there are training samples
	while (nrsamples<MAXNRSAMPLES)
	{
		if (fscanf(f, "%d", &lbl) < 1)
			break;

		if (lbl == 1){
			w[nrsamples] = POSW;
			l[nrsamples] = 1;
		}
		else{
			w[nrsamples] = 1;
			l[nrsamples] = -1;
		}

		for (int j = 0; j<nrfeatures; j++)
			fscanf(f, ",%f", descr++);
		nrsamples++;
		if (nrsamples % 100)
			printf("\r %.2f %%", ftell(f)*100.f / file_size);
	}
	fclose(f);
	printf("\nDone.\n"); if (nrsamples == MAXNRSAMPLES) printf("Stopping prematurely (%d)\n", nrsamples);

	if (nrsamples > 65535 && sizeof(T3) == 2){
		printf("Error, nrsamples=%d is larger than T3 type can handle (65535)\n", nrsamples);
		getch();
		exit(-2);
	}

	//set fields and normalize weights
	N = nrsamples;
	M = nrfeatures;
	// nrfeatures == (descr-tdata.d)/nrsamples; //debug assert
	T iN = 1.f / N;
	for (int i = 0; i<N; i++)
		w[i] *= iN;
}

void Traindata::loadDataBinary(const char* fname, int dim){
	int nrsamples = 0;
	int tmplbl;
	ifstream in(fname, ios::in | ios::binary);
	if (!in)
	{
		printf("File not found %s\n", fname);
		return;
	}

	T2 sumw = 0;

	w = new T2[MAXNRSAMPLES];
	l = new T[MAXNRSAMPLES];
	d = new T[MAXNRSAMPLES*dim];
	T* descr = d;
	while (nrsamples<MAXNRSAMPLES){
		if (in.eof())	break;

		in.read((char*)&tmplbl, 4);
		int d = 0;
#if 0 //buffering not, very useful
		const int chunk = 2000;
		while (d + chunk < dim){
			in.read((char*)(descr + d), sizeof(T)*chunk);//RRR change back
			d += chunk;
		}
#endif
		in.read((char*)(descr + d), sizeof(T)*(dim - d));

		if (nrsamples % RED == 0){
			if (tmplbl == 1)
				l[nrsamples] = 1.0;
			else
				l[nrsamples] = -1.0;
			w[nrsamples] = 1.0;
			sumw += w[nrsamples];
			descr += dim;
		}
		nrsamples++;
	}
	in.close();

	N = nrsamples;
	M = dim;
	for (int i = 0; i<nrsamples; i++)
		w[i] /= sumw;

#if 0//reallocate with exact size

#endif
}

void Traindata::normalizeData(Traindata* tdata){
	const int nrsamples = tdata->N;
	const int nrfeatures = tdata->M;

	T* means = new T[nrfeatures];
	T* stds = new T[nrfeatures];
	memset(means, 0, nrfeatures*sizeof(T));
	memset(stds, 0, nrfeatures*sizeof(T));

	//calculate mean
	for (int i = 0, i2=0; i<nrsamples; i++, i2+=nrfeatures)
	{
		for (int j = 0; j<nrfeatures; j++)
		{
			means[j] += tdata->d[i2 + j];
		}
	}

	T inrsamples = 1.f / nrsamples;
	for (int j = 0; j<nrfeatures; j++)
		means[j] *= inrsamples;

	//calculate standard deviation
	for (int i = 0, i2 = 0; i<nrsamples; i++, i2+=nrfeatures)
	{
		for (int j = 0; j<nrfeatures; j++)
		{
			stds[j] += sqr(tdata->d[i2 + j] - means[j]);
		}
	}

	//normalize
	for (int i = 0, i2 = 0; i<nrsamples; i++, i2+=nrfeatures)
	{
		for (int j = 0; j<nrfeatures; j++)
		{
			tdata->d[i2 + j] = (tdata->d[i2 + j] - means[j]) / sqrt(stds[j] + EPS);
		}
	}

	delete[] means;
	delete[] stds;
}