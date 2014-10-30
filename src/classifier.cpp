#include "classifier.h"

Classifier::Classifier(){
	tdata = 0;
}

Classifier::~Classifier(){
	if (tdata) delete tdata;
}

void Classifier::setData(Traindata* td){
	tdata = td;
}

T Classifier::calculateError(){	
	//data must be loaded previously
	T err = 0;
	if (!tdata->d || !tdata->l) return err;

	T* sample = tdata->d;
	for (int i = 0; i < tdata->N; i++)
	{
		T lbl = predict(sample);
		if (lbl*tdata->l[i] < 0)
			err++;
		sample += tdata->M;
	}
	err /= tdata->N;
	return err;
}

void randperm(int n, int perm[])
{
	srand(0);
	int i, j, t;
	for (i = 0; i<n; i++)
		perm[i] = i;
	for (i = n-1; i>0; i--) {
		j = rand() % (i+1); // Knuth shuffle generates all perms in uniform distr.
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
}

T Classifier::randsplit(float trainpercent){
	//data must be loaded previously
	T errtest = 0;
	if (!tdata->d || !tdata->l) return errtest;

	int* b = new int[tdata->N];
	randperm(tdata->N, b);

	int ilimit = tdata->N*trainpercent;
	for (int i = 0; i < tdata->N; i++)
	{
		if (i < ilimit)
			tdata->w[b[i]] = 1.f/ilimit;
		else
			tdata->w[b[i]] = 0;
	}

	train();
	
	for (int i = 0; i < ilimit; i++)
	{
		T lbl = predict(tdata->d + b[i] * tdata->M);
		if (lbl*tdata->l[b[i]] < 0)
			errtest++;
	}
	errtest /= (tdata->N - ilimit);
	printf("Error on the test split: %.4f %%, acc: %.4f %%\n", errtest * 100, 100 * (1 - errtest));
	return errtest;
}


T Classifier::crossvalidation(int nfold, T* errs){
	//data must be loaded previously
	T erravg = 0;
	if (!tdata->d || !tdata->l) return erravg;

	const int MM = tdata->N / nfold;
	int* b = new int[tdata->N];
	randperm(tdata->N, b);

	for (int kfold = 0; kfold<nfold; kfold++){
		for (int i = 0; i<tdata->N; i++)
		if (i*nfold / tdata->N == kfold)
			tdata->w[b[i]] = 0.f;
		else
			tdata->w[b[i]] = 1.f/(tdata->N-MM);

		train();

		float err = 0, total = 0;
		for (int i = kfold*tdata->N / nfold; i<(kfold + 1)*tdata->N / nfold; i++){
			T lbl = predict(tdata->d + b[i] * tdata->M);
			if (lbl*tdata->l[b[i]]<0)
				err++;
			total++;
		}
		errs[kfold] = err / total;
		printf("%d fold error: %.5f %%\n", kfold, 100 * errs[kfold]);
		erravg += errs[kfold];
	}
	erravg /= nfold;
	printf("Crossvalidation summary:\n");
	for (int kfold = 0; kfold<nfold; kfold++)
		printf("%d fold error: %.5f %%\n", kfold, 100 * errs[kfold]);
	printf("average error: %.5f %%\n", 100 * erravg);

	delete[] b;
	return erravg;
}

void Classifier::speedTest(){

	double start = clock();
	T* sample = tdata->d;
	for (int i = 0; i < tdata->N; i++, sample+=tdata->M)
		predict(sample);
	double end = clock();
	printf("Execution time:\n");
	printf("\tWhole test set: %.2f sec \n", (end-start)/CLOCKS_PER_SEC);
	printf("\tSingle instance: %.2f sec \n", (end - start) / CLOCKS_PER_SEC/tdata->N);
	printf("\tClassifications per second %.2f sec \n", CLOCKS_PER_SEC * tdata->N/(end - start));

}