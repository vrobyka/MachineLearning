#ifndef BOOSTING_H
#define BOOSTING_H
#include "dtree.h"
#include "classifier.h"


/**
Structure containing the boosting parameters
*/
struct Boostparams{
	int type;				/**< The type of the boosting algorithm. Currently not used. */
	int nrweak;			/**< The number of weak learners used in the model. */
	int maxDepth;			/**< The depth of each decisition tree. */
	float pruning;  /**< Constant for pruning. Currently not used. */
};

/**
Structure containing the model used for classification. 
*/
struct Boostmodel{
	Boostparams* bp;		/**< The boosting parameters*/
	T2* c;								/**< Array containing the weight of each weak learner. Has bp->nrweak elements*/
	T* thetas;					/**< Array containing the rejection thesholds at each stage. Has bp->nrweak elements*/
	DTree* trees;				/**< Array containing the decision trees. Has bp->nrweak elements*/
	T cverror;					/**< The crossvalidation error. May be filled at training. Currently not used*/
};

/**
Class implementing the AdaBoost algorithm.
To train: 
1. Load the training data using loadData() (csv format) or loadDataBinary(); 
2. Set the number of weak learners; 
3. Call train(); 
4. Save the model saveModel().

To predict: 
1. Load an existing model loadModel();
2. Call predict(T* sample) by providing a sample;
Call predictCascade() to predict using the rejection thresholds (thetas).
*/
class Boosting : public Classifier{
public:
	Boosting();
	Boosting(int nrweak);
	Boosting(Boostparams bp);
	~Boosting();	

	//model io
	void loadModel(const char* fname) override; 
	void saveModel(const char* fname) override;

	//training
	T train() override;		
	T train(const char* fname);
	T train(T* data, T*labels, int N, int M);	
	
	//predicting/classifying
	T predict(T* sample) override; /**< Predict using all weak learners */
	T predictCascade(T* sample, float cascade_th=0);	/** Predict from the sample array (fastest) using rejection thresholds. Evaluates the decision function until it is larger than the rejection threshold for each stage. \param cascade_th Amount to decrease the rejection thresholds. Positive values delay rejection.*/
	
	/** Perform nfold crossvalidation.
	\param nfold The number of folds.
	\param errs The error for each fold.
	\return The average error.
	*/		
	void recalculateThresholds(float Q); /**< Recalculate rejection thresholds for different ending threshold. (Q=0 by default). Consider using predictCascade with positive cascade_th value.*/
	

private:
	void presortData(T3*& oi);
	Boostmodel *model;
	
};

#endif