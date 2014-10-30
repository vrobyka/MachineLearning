#ifndef STDH_H
#define STDH_H

#pragma warning(disable: 4996) // fscanf unsafe
#pragma warning(disable: 4244) // possible loss of data
#pragma warning(disable: 4267) // possible loss of data

#include <string.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <conio.h>
#include <time.h>
using namespace std;

#define MIN(x,y) (x)<(y)?(x):(y);
#define MAX(x,y) (x)>(y)?(x):(y);

#define INF 1e5
#define EPS 1e-5 
#define ERR_THR 1e-5
typedef float T;
typedef float T2;
typedef unsigned int T3;

template<class T> T sqr(T x){ return x*x; }

//pair of index and value for sorting and remembering indexes
struct Pair{
	T3 i; 	T x;
	Pair(){ i=0; x=0; };
	Pair(T xx, T3 ii) :x(xx), i(ii){};
	bool operator() ( const Pair& l, const Pair& r)	{ 	 return l.x < r.x; 	}
};

#endif