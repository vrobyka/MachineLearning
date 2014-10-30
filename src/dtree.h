#ifndef DTREE_H
#define DTREE_H
#pragma warning(disable: 4996)
#pragma warning(disable: 4244) // possible loss of data
#pragma warning(disable: 4267) // possible loss of data

#include "stdh.h"
#include "traindata.h"
// 1 GINI, 
// 2 enthropy, 
// 3 local classification error - recommended (a.k.a. node error)
// 4 classification error impurity change - not working
// 5 twoing
#define IMPURITY 3 //use 3

/**
This structure represents a single node from the decision tree.
*/
struct Node{
	//tree information: 
	int child; /**< Index to left child in the Node array. If zero node is a leaf. The right child is next.*/
	int depth; /**< The depth of the node. Starts at 1. */

	//decision information: 
	int index;				/**< The index of the feature. \see predict. */ 
	T theta;				/**< The decision boundary. \see predict. */  
	T label;				/**< The label returned if feature[index] < theta. */
	T2 err;					/**< Training error. */ 
	T weight;				/**< The ratio between the instances at the node and the total instances. */

	Node(){ depth = 0;  index = 0;	theta = 0;	label = 1;	err = 0;	weight = 0;};
};

struct DTreeParams{
	int maxDepth;	/**< Maximum depth of the decision tree */
	int dim; 		/**< The dimension of a feature vector */
	bool saveBinary; /**< Set to true to save/load in binary format, otherwise readable text format */
	bool isBinary; 	/**< If isBinary = 1, then the tree returns +/-1*tree weight, otherwise +/-prob. estimate * tree weight */
	
	DTreeParams(){ maxDepth = 2; isBinary = 1; saveBinary = 0; };
};

/**
Implementation of a Decision Tree. Contains both training and predicting procedures.
At prediction the tree is traversed from the root to a leaf node.
Every node corresponds to the decision:

if feature_index < theta
	return label
else 
	return -label
	
A node is a leaf node if the child field is zero.
*/
class DTree{
public:
	::DTreeParams p;  
	int nrnodes; 	/** < 2^(maxDepth)-1 nodes. */
	Node* nodes; 	/** < BF order array of the tree. */
	//		
	DTree();
	DTree(::DTreeParams *p);
	~DTree();
	//
		
	/** Main training procedure works with already loaded training data from tdata.*/
	void train(Traindata* tdata, T3* oi);
	/** Predict from sample array, array length is known. Involves descending from the root to a leaf node. */
	T predict(T* sample);
	
	/** Function for saving the tree in binary/text format.*/
	void save(FILE* f);
	/** Function for saving the tree in binary/text format.*/
	void load(FILE* f);
	/** Function for modifying the label from +/-1 to +/-c[m].*/
	void setLabel(T lbl);

private:	
	void normalizeData(T* data, int nrfeatures, int nrsamples);
	/** Recursive helper function for training. Not called by the user. */
	void trainrec( Traindata* tdata, vector<int>* indexes, T3* oi, int treeindex );
	//io stuff
	ofstream fout;
	ifstream fin;
	/** Recursive helper function for saving in binary format.*/
	void saverec(Node* n);
	/** Recursive helper function for loading in binary format.*/
	void loadrec(Node* n);	
};

/**
Structure used for storing temporary information during training
*/
struct Info{
	T3 i;
	T2 imp;
	T2 err;
	T theta;
	T label;
};

#endif