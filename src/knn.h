#include "classifier.h"

class Knn : public Classifier{
	int K; /**< The K from the Knn, nr of neighbors to consider*/
	Data* trainingdata;
public:
	Knn();
	Knn(int K);
	~Knn();
	T train() override;
	T predict(T* sample) override;

	void saveModel(const char* fname) override;
	void loadModel(const char* fname) override;
};