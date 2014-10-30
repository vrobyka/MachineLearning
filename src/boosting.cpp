#include "stdh.h"
#include "util.h"
#include "dtree.h"
#include "boosting.h"

Boosting::Boosting(){
	//prepare structures
	model = new Boostmodel;	
	model->bp = new Boostparams;
	model->bp->nrweak = -1;
	model->bp->maxDepth = 2;
	model->bp->type = 0;
	model->bp->pruning = 0;
}

Boosting::Boosting(int nrweak){
	//prepare structures
	model = new Boostmodel;		
	model->bp = new Boostparams;
	model->bp->nrweak = nrweak;
	model->bp->maxDepth = 2;
	model->bp->type = 0;
	model->bp->pruning = 0;
	model->thetas = 0;
	model->c = new T2[nrweak];
	model->trees = new DTree[nrweak];	
}

Boosting::~Boosting(){
	delete[] model->c;
	delete[] model->trees;		
	delete[] model->thetas;	
	delete model->bp;
	delete model;
}

void Boosting::saveModel(const char* fname){
	FILE* f = fopen(fname, "w");
	if (!model || !model->bp)
	{
		printf("Model not found\n");
		getch();
		exit(-3);
	}

	fprintf(f, "boosted-classifier-model-file\n");
	fprintf(f, "type:%d\n", model->bp->type);
	fprintf(f, "nrweak:%d\n", model->bp->nrweak);
	fprintf(f, "pruning:%f\n", model->bp->pruning);
	fprintf(f, "weakdepth:%d\n", model->bp->maxDepth);

	for (int i = 0; i<model->bp->nrweak; i++)
	{
		fprintf(f, "w:%f, rej:%f\n", model->c[i], model->thetas[i]);
		model->trees[i].save(f);
	}
	fclose(f);
}

void Boosting::loadModel(const char* fname){
	FILE* f = fopen2(fname, "r");
	
	char str[100];
	fscanf(f, "%s\n", str);
	fscanf(f, "type:%d\n", &model->bp->type);
	fscanf(f, "nrweak:%d\n", &model->bp->nrweak);
	fscanf(f, "pruning:%f\n", &model->bp->pruning);
	fscanf(f, "weakdepth:%d\n", &model->bp->maxDepth);

	model->thetas = new T[model->bp->nrweak];
	model->c = new T2[model->bp->nrweak];
	model->trees = new DTree[model->bp->nrweak];
	for (int i = 0; i<model->bp->nrweak; i++)
	{
		fscanf(f, "w:%f, rej:%f\n", &model->c[i], &model->thetas[i]);
		model->trees[i].load(f);
		model->trees[i].setLabel(model->c[i]);
	}
	fclose(f);
}

void Boosting::presortData(T3*& oi){
	printf("Presorting input data\n");
	//sort columns into ordered indexes	
	oi = new T3[tdata->M*tdata->N];
#pragma omp parallel for
	for (int j = 0; j<tdata->M; j++)
	{
		vector<Pair> pairs;
		/*
		pairs.reserve(tdata->N);
		for(int i=0, iM=0; i<tdata->N; i++, iM+=tdata->M)
		pairs.push_back(Pair( tdata->d[iM+j], (T3)i) );
		*/
		pairs.resize(tdata->N);
		for (int i = 0, iM = 0; i<tdata->N; i++, iM += tdata->M)
			pairs[i] = Pair(tdata->d[iM + j], (T3)i);
		sort(pairs.begin(), pairs.end(), pairs[0]);
		T3* oijn = oi + j*tdata->N;
		for (int i = 0; i<tdata->N; i++)
			oijn[i] = pairs[i].i;
	}
	printf("Done (%d x %d entries)\n", tdata->N, tdata->M);
}

T Boosting::train(T* data, T*labels, int N, int M){
	tdata->N = N;
	tdata->M = M;
	tdata->w = new T2[N];
#define COPY_DATA 0
#if COPY_DATA //copy data
	tdata->d = new T[N*M];
	tdata->l = new T[N];
	memcpy(tdata->d, data, N*M*sizeof(T));
	memcpy(tdata->l, labels, N*sizeof(T));
#else
  tdata->l = labels;
  tdata->d = data;
#endif
	for(int i=0; i<N; i++)
    tdata->w[i] = ((T2)1.0)/N;
	
	float ret = train();
#if !COPY_DATA //this way the destructor does not deallocate the memory locations ad d and l
	tdata->d = 0;
	tdata->l = 0;
#endif
	return ret;
}

T Boosting :: train(const char* fname){
	tdata = new Traindata();
	tdata->loadData(fname);
	return train();
}

T Boosting::train(){
	int nrweak = model->bp->nrweak;
	const int N = tdata->N;
	const int M = tdata->M;
	T2* lbls = new T2[N];
	T2* scores = new T2[N]; memset(scores, 0, N*sizeof(T2));
	T2* c = model->c;
	DTree* trees = model->trees;

	T3* oi = 0;
	presortData(oi);
		
	Timer t("time elapsed");
	int m=0;
	for(; m<nrweak; m++)
	{
		//printf("Training tree %d\n",m);
		trees[m].p.maxDepth = model->bp->maxDepth;
		//t.tic();
		trees[m].train( tdata, oi );		
		//t.toc();		

		//find the weighted training error for the m-th weak classifier		
#pragma omp parallel for //reduction(+:errm)
		for(int i=0; i<N; i++)
		{
			lbls[i] = trees[m].predict( tdata->d+i*M ); //? maybe try to eliminate			
		}
		
		T2 errm = 0;
		for(int i=0; i<N; i++) 
			if (lbls[i] * tdata->l[i]<0) 
				errm += tdata->w[i];		
					
		if (errm > 0.5 )
			break;		
				
		T2 sumw = 0;
		c[m] = 0.5*log( ( 1-errm) / (errm+EPS) );

		T2 err = 0;
		for (int i = 0; i < N; i++)
		{
			scores[i] += c[m] * lbls[i];
			if (scores[i] * tdata->l[i] < 0)
				err++;
		}
		err /= tdata->N;
		printf("Error Using %d trees: %5.4f %%, acc %5.4f %%\n", m, err*100.f, (1 - err)*100.f);

		for(int i=0; i<N; i++)
		{
			tdata->w[i] *= exp( -c[m] * lbls[i] * tdata->l[i] );
			sumw += tdata->w[i];
		}
		trees[m].setLabel(c[m]);		

		for(int i=0; i<tdata->N; i++)
			tdata->w[i] /= sumw;
	}
	model->bp->nrweak = m;
	nrweak = m;

	//show the training error using all the weak classifiers
	for (int m2=nrweak; m2<=nrweak; m2++)
	{
		model->bp->nrweak = m2;
		T2 err = 0;
		for(int i=0; i<N; i++)
		{
			T lbl = predict( tdata->d+i*M );
			if ( lbl * tdata->l[i] < 0 )
				err++;
		}
		err /= tdata->N;
		printf("\nUsing all %d trees: err %5.4f %%, acc %5.4f %%\n", m2, err*100.f, (1-err)*100.f);
	}

	//find the rejection thresholds using direct backward pruning (Viola Jones)
	model->thetas = new T[nrweak];
	T* sampletrace  = new T[N];
	memset(sampletrace, 0, N*sizeof(T));

	for(int j=0; j<model->bp->nrweak; j++){
		T minsampletrace = INF;
		for(int i=0; i<N; i++)
			if ( tdata->l[i] == 1.0 )
			{				
				T pred = model->trees[j].predict( tdata->d+i*M );
				sampletrace[i] += pred;			
				if ( sampletrace[i] < minsampletrace )
					minsampletrace = sampletrace[i];
			}
		model->thetas[j] = minsampletrace - EPS;
	}
	
	//show the training error using all the weak classifiers and the rejection thrs
	double start = clock();
	T err2 = 0;
	T avg = 0;
	for(int i=0; i<tdata->N; i++)
	{
		T lbl = 0;
		int j;
		for(j=0; j<nrweak; j++)
		{
			lbl += model->trees[j].predict( tdata->d+i*M );
				if (lbl<model->thetas[j])
				{
					lbl = -1;
					j++;
					break;
				}
		}
		avg += j;
		if ( lbl * tdata->l[i] < 0 )
			err2++;
	}
	double end = clock();
	avg /= N;
	err2 /= tdata->N;
	printf("Training data results: err2 %5.4f %%, acc %5.4f %%\n", err2*100.f, (1-err2)*100.f);	
	
	printf("\nMade %d classifications in %f seconds, speed: %f ms/instance\n", N, (end-start)/CLOCKS_PER_SEC, (end-start)*1000/CLOCKS_PER_SEC/N );
	printf("Average number of feature evaluations %.2f out of %d\n", avg, model->bp->nrweak);
	
	delete[] oi;
	delete[] lbls;	
	delete[] scores;
	delete[] sampletrace;
	printf("Done!\n");
	return err2;
}

T Boosting::predict(T* sample){
	T pred = 0;
	for(int i=0; i<model->bp->nrweak; i++)
		pred += model->trees[i].predict(sample);
	return pred;
}

T Boosting::predictCascade(T* sample, float cascade_th){ //predict using all the weak classifiers 
	T pred = 0;
	for(int i=0; i<model->bp->nrweak; i++)
	{
		pred += model->trees[i].predict(sample);
		if (pred<model->thetas[i]-cascade_th)
		{
				pred = -1000; //assign a negative score as prediction
				break;
		}
	}
	return pred;	
}

void Boosting::recalculateThresholds(float Q){
	//recalculates the cascade thesholds so that thetas_new[end] = thetas[end]+Q and thetas_new[0]=thetas[0]
	float t0 = model->thetas[0];
	float tend = model->thetas[model->bp->nrweak-1]+Q;
	float incr = 1.f/(model->bp->nrweak-1);
	int i;
	float t;
	for(t=0.f, i=0; i<model->bp->nrweak; i++, t+=incr)
		model->thetas[i] = (1-t)*t0+t*tend;
}

#if 0 //probability estimate (real adaboost)
T pm = 0;
T sumw = 0;
for (int i = 0; i<N; i++)
{
	pm = trees[m].predictp(data[i]);
	if (pm == 0)
		pm += EPS;
	if (pm >= 1)
		pm = 1 - EPS;
	fm = 0.5* log(pm[i]) / log(1 - pm[i]);
	data[i][wi] *= exp(-data[i][data[0].size() - 1]] * fm[i]);
	sumw += data[i][wi];
}
#endif