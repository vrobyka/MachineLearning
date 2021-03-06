#include "boosting.h"
#include "knn.h"
#include "neural.h"
#include "data.h"
#include "util.h"

void main(){
	//Boosting classifier(200);
	//Knn classifier(3);
	int ll[2] = { 3, 1 };	
	Neural classifier(2, ll);

	Data* trainingdata = new Data();
	char trainname[256] = "d:\\Cproj\\MachineLearningGit\\data\\adult_train_num.csv";
	//char tfname[256] = "d:\\Cproj\\MachineLearningGit\\data\\pallets.csv";
	//char trainname[256] = "d:\\Cproj\\MachineLearningGit\\data\\dexter_train.sparse";
	trainingdata->loadData(trainname);
	classifier.setData(trainingdata);
	classifier.train();
	/*
	char testname[256] = "d:\\Cproj\\MachineLearningGit\\data\\adult_test_num.csv"; //knn1 30%, knn3 25%
	Data* testdata = new Data();
	testdata->loadData(testname);
	classifier.setData(testdata);

	classifier.calcError();
	*/
	//classifier.randsplit(0.6);
	//float errs[5];
	//classifier.crossvalidation(5, errs);
	//classifier.speedTest();
	
	getch();
}
