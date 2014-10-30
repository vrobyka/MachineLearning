#include "boosting.h"
#include "traindata.h"
#include "util.h"

void main(){

	int nrweak = 20;
	Boosting b(nrweak);	
	
	Timer t("Training time");
	Traindata* td = new Traindata();
	char tfname[256] = "d:\\Cproj\\MachineLearningGit\\data\\simple_train_3.csv";
	//char tfname[256] = "d:\\Cproj\\MachineLearningGit\\data\\pallets.csv";
	td->loadData(tfname);	
	b.setData(td);

	t.tic();
	//b.train();	
	b.randsplit(0.6);

	//float errs[5];
	//b.crossvalidation(3, errs);
	t.toc();

	b.saveModel("model0.txt");
	getch();
}