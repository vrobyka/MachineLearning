#include "classifier.h"

Classifier::Classifier(){
	data = 0;
}

Classifier::~Classifier(){
	if (data) delete data;
}

void Classifier::setData(Data* d){
	data = d;
}

T Classifier::calcError(){
	//data must be loaded previously
	T err = 0;
	if (!data->d || !data->l) return err;

	T* sample = data->d;
	for (int i = 0; i < data->N; i++)
	{
		//printf("%d of %d\n", i, data->N);
		T lbl = predict(sample);
		if (lbl*data->l[i] < 0)
			err++;
		sample += data->M;
	}
	err /= data->N;
	printf("Error on the dataset: %.4f %%, acc: %.4f %%\n", err * 100, 100 * (1 - err));
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
	if (!data->d || !data->l) return errtest;

	int* b = new int[data->N];
	randperm(data->N, b);

	int ilimit = data->N*trainpercent;
	for (int i = 0; i < data->N; i++)
	{
		if (i < ilimit)
			data->w[b[i]] = 1.f/ilimit;
		else
			data->w[b[i]] = 0;
	}

	train();
	
	for (int i = ilimit; i < data->N; i++)
	{
		T lbl = predict(data->d + b[i] * data->M);
		if (lbl*data->l[b[i]] < 0)
			errtest++;
	}
	errtest /= (data->N - ilimit);
	printf("Error on the test split: %.4f %%, acc: %.4f %%\n", errtest * 100, 100 * (1 - errtest));
	return errtest;
}


T Classifier::crossvalidation(int nfold, T* errs){
	//data must be loaded previously
	T erravg = 0;
	if (!data->d || !data->l) return erravg;

	const int MM = data->N / nfold;
	int* b = new int[data->N];
	randperm(data->N, b);

	for (int kfold = 0; kfold<nfold; kfold++){
		for (int i = 0; i<data->N; i++)
		if (i*nfold / data->N == kfold)
			data->w[b[i]] = 0.f;
		else
			data->w[b[i]] = 1.f/(data->N-MM);

		train();

		float err = 0, total = 0;
		for (int i = kfold*data->N / nfold; i<(kfold + 1)*data->N / nfold; i++){
			T lbl = predict(data->d + b[i] * data->M);
			if (lbl*data->l[b[i]]<0)
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
	T* sample = data->d;
	for (int i = 0; i < data->N; i++, sample+=data->M)
		predict(sample);
	double end = clock();
	printf("Execution time:\n");
	printf("\tWhole test set: %.2f sec \n", (end-start)/CLOCKS_PER_SEC);
	printf("\tSingle instance: %.2f sec \n", (end - start) / CLOCKS_PER_SEC/data->N);
	printf("\tClassifications per second %.2f sec \n", CLOCKS_PER_SEC * data->N/(end - start));

}