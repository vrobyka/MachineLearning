#pragma once

#include <vector>
#include <cmath>
#define VERBOSE 0

#include "dtree.h"
DTree::DTree(){	
	nrnodes = (1 << (p.maxDepth+1)) - 1;
	nodes = new Node[nrnodes];
}

DTree::DTree(DTreeParams* p2){
	p = *p2;
	nrnodes = (1 << (p.maxDepth+1)) - 1;
	nodes = new Node[nrnodes];
}

DTree::~DTree(){
	if( nodes ) delete[] nodes;
}

void DTree::train(Data* tdata, T3* oi){	
	//run recursive training procedure
	vector<int> indexes; 
	for(int i=0; i<tdata->N; i++) 
		//if (tdata->w[i])
			indexes.push_back(i);
	nodes[0].depth = 0;
	trainrec( tdata, &indexes, oi, 0 );
	//cleanup for oi
}

void DTree::trainrec( Data* tdata, vector<int>* indexes, T3* oi, int treeindex ){
	const int nrsamples = indexes->size();
	const int nrfeatures = tdata->M;
	Node* node = nodes + treeindex;

	T lbl = 1; int nn=0, np=0;
	for(int i=0; i<nrsamples; i++)
		if ( tdata->l[(*indexes)[i]]>0 )		
			np++;
		else
			nn++;
	T2 errhere = ((T2)min( nn, np))/nrsamples;

	if( nn==0 || np ==0 || node->depth == p.maxDepth || errhere < ERR_THR) //stopping conditions: small error or maxDepth reached
	{
		if (nn>np) lbl = -1;
		//node->label = lbl;
		node->theta = INF; //everything is classified as lbl - not needed since child = 0
		node->index = 0;
		node->err = errhere;
		node->child = 0;		
		node->weight = ((T2)nrsamples)/tdata->N;
		return;
	}

	T2 sumw = 0, sumwp = 0, sumwn = 0; 
	for(int ii=0; ii<nrsamples; ii++)
	{
		int i0 = (*indexes)[ii];		
		if( tdata->l[i0] == 1 )
			sumwp += tdata->w[i0];					
		else
			sumwn += tdata->w[i0];					
	}
	sumw = sumwp+sumwn;

	T2 globalminerr = 1e1;
	int globalminerri = -1;
	T2 globaltheta = 0;
	T2 globallabel = 0;
	T2 globalmaximp = -INF;

	Info* errs = new Info[nrfeatures];
#pragma omp parallel for
	for(int j=0; j<nrfeatures; j++)
	{
		T2 maximp = -INF;
		T2 minerr = 1e1;
		T theta = -1;
		T label = -1;
		int minerri = -1;

		T2 errn, errp;
		T2 wcur=0, w0=0, w1=0;		
		T3* oip = oi + j*nrsamples;
		for(int ii=0; ii<nrsamples-1; ii++)
		{
			T3 i0 = oip[ii];	//crt index
			T3 i1 = oip[ii+1]; //next index
			wcur += tdata->w[i0];
			if (tdata->l[i0] == -1 )
				w0 += tdata->w[i0];
			else
				w1 += tdata->w[i0];

			errn = ( w1 + sumwn - w0); 
			errp = ( w0 + sumwp - w1); 

			T2 err;
			T labelj;
			if ( errn <= errp )
			{
				labelj = -1;
				err = errn/sumw;
			}
			else //( errn > errp )
			{
				labelj = 1;
				err = errp/sumw;
			}			

			T2 impuritychange = 0;
#if IMPURITY == 1
			//GINI impurity
	#if 0
			impuritychange = 0;
			impuritychange = 1 - sqr2(sumwn/sumw+EPS) - sqr2(sumwp/sumw+EPS);
			impuritychange -= wcur/(sumw+EPS) * ( 1 - sqr2(w0/wcur+EPS) - sqr2(w1/wcur+EPS) );
			impuritychange -= (sumw-wcur)/(sumw+EPS) * ( 1 - sqr2((sumwn-w0)/(sumw-wcur+EPS)) 
				- sqr2((sumwp-w1)/(sumw-wcur+EPS)));				
	#else
			//http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-cart_split-criteria_categorical_gini.htm
			//float it = sumwp/sumw * sumwp/sumw  + sumwn/sumw * sumwn/sumw; //p1^2+p2^2
			//float itl = w0/wcur*w0/wcur +  w1/wcur*w1/wcur;
			//float itr = (sumwn-w0)/(sumw-wcur) *(sumwn-w0)/(sumw-wcur) + (sumwp-w1)/(sumw-wcur)*(sumwp-w1)/(sumw-wcur);			
			//impuritychange = 1-it + 1-itl + 1-itr;
			double p0l = w0/wcur;
			double itl = sqr(p0l) + sqr(1-p0l);
			double p0r = (sumwn-w0)/(sumw-wcur);
			double itr = sqr(p0r) + sqr(1-p0r);
			double pl = wcur / sumw;
			double pr = 1-pl;			
			impuritychange = pl*(1-itl) + pr*(1-itr); // + (1-it)
			impuritychange = 1-impuritychange;
	#endif
#elif IMPURITY == 2
			//entropy
			impuritychange = 0;
			T2 pp = MAX(sumwp/sumw, EPS);
			T2 pn = MAX(sumwn/sumw, EPS);
			impuritychange = -pp * log(pp)  - pn * log(pn);
			T2 ppl = MAX(w1/wcur, EPS);
			T2 pnl = MAX(w0/wcur, EPS);
			impuritychange -= -wcur/sumw*(ppl * log(ppl) + pnl * log(pnl));
			T2 ppr = MAX((sumwp-w1)/(sumw-wcur), EPS);
			T2 pnr = MAX((sumwn-w0)/(sumw-wcur), EPS);
			impuritychange -= -(sumw-wcur)/sumw*(ppr * log(ppr) + pnr * log(pnr));
#elif IMPURITY == 3
			//local classification error
			impuritychange = 1-err;
#elif IMPURITY == 4
			// (1-err) - pl*(1-errl) - pr*(1-errr)
			impuritychange = -err 
				+ wcur/sumw*MAX(w1/wcur,w0/wcur) 
				+ (sumw-wcur)/sumw*MAX((sumwp-w1)/(sumw-wcur), (sumwn-w0)/(sumw-wcur));
#elif IMPURITY == 5
			T pp = MAX(sumwp/sumw, EPS);
			T pn = MAX(1-pp, EPS);
			T ppl = MAX(w1/wcur, EPS);
			T pnl = MAX(1-ppl, EPS);
			T ppr = MAX((sumwp-w1)/(sumw-wcur), EPS);
			T pnr = MAX(1-ppr, EPS);
			T diff = ( abs(ppl-ppr) + abs(pnl-pnr) );
			impuritychange = pn*pp*diff*diff;
#endif
			
			if ( impuritychange > maximp && tdata->d[i0*nrfeatures+j] != tdata->d[i1*nrfeatures+j] )
			{
				maximp = impuritychange;
				minerr = err;
				minerri = i0;
				theta = ( tdata->d[i0*nrfeatures+j] + tdata->d[i1*nrfeatures+j] ) * 0.5;
				label = labelj;
			}
		}	//end for samples	

		//#pragma omp critical
		{
			errs[j].imp = maximp;
			errs[j].err = minerr;
			errs[j].i = minerri;
			errs[j].theta = theta;
			errs[j].label = label;
		}
		
		
	}

	for(int j=0; j<nrfeatures; j++)
		if ( errs[j].imp > globalmaximp )
		{
			globalmaximp = errs[j].imp;
			globalminerr = errs[j].err;
			globalminerri = j;
			globaltheta = errs[j].theta;
			globallabel = errs[j].label;
		}

	if ( p.isBinary)
		node->label = globallabel;
	else
		node->label = globallabel*(1 - 2*globalminerr); // = +1*l(1-err) -1*l*err
	node->index = globalminerri;
	node->theta = MIN(globaltheta, INF);	
	node->err = globalminerr;
	node->weight = nrsamples * 1.0 / tdata->N;
	node->child = (treeindex << 1) + 1;
#if VERBOSE == 1
	printf("\tindex=%5d\t theta=%10.5f error=%.5f\n", node->index, node->theta, node->err );	
#endif

	vector<int> indexesleft, indexesright;
	//indexesleft.reserve( tdata->N );
	//indexesright.reserve( tdata->N );

	bool* isgoingleft = new bool[tdata->N];
	memset(isgoingleft, 0, tdata->N);
	for(int i=0; i<nrsamples; i++)	
	{
		int ii = (*indexes)[i];
		if ( tdata->d[ii*nrfeatures+node->index] < node->theta )			
		{
			indexesleft.push_back(ii);
			isgoingleft[ii] = 1;		
		}
		else
			indexesright.push_back(ii);			
	}

  //separate already sorted columns for the two subtrees
	const int leftsize = indexesleft.size(), rightsize = indexesright.size();
	T3* oil = new T3[nrfeatures*leftsize];
	T3* oir = new T3[nrfeatures*rightsize];
	//T3* oilp = oil, *oirp = oir, *oip = oi;
#pragma omp parallel for //O(NM)
	for(int j=0; j<nrfeatures; j++)
	{
		T3* oilp = oil+j*leftsize;
		T3* oirp = oir+j*rightsize;
		T3* oip = oi+j*nrsamples;
		int il = 0, ir = 0;
		for(int i=0; i<nrsamples; i++)	
			if( isgoingleft[ oip[i] ] )
				oilp[(il++)] = oip[i];
			else
				oirp[(ir++)] = oip[i];	

		//assert(il==leftsize); assert(ir==rightsize);
		//oip += nrsamples;
		//oilp += leftsize;
		//oirp += rightsize;
	}		
	delete[] isgoingleft;

//#pragma omp parallel sections
	{
//#pragma omp section
		{
			int childi = node->child; //left child
			nodes[childi].depth = node->depth+1;
			nodes[childi].label = node->label;
			trainrec( tdata, &indexesleft, oil, childi );
			delete[] oil;
		}
//#pragma omp section
		{
			int childi = node->child + 1; //right child
			nodes[childi].depth = node->depth+1;
			nodes[childi].label = - node->label;
			trainrec( tdata, &indexesright, oir, childi );	
			delete[] oir;
		}
	}	
	int childleft = (treeindex << 1) + 1;
	node->err = nodes[childleft].weight* nodes[childleft].err + nodes[childleft+1].weight* nodes[childleft + 1].err;
	delete[]  errs;
}

T DTree::predict(T* sample){ //pointer version of predict
	int k=0;
	while( nodes[k].child )
		if ( sample[ nodes[k].index ] < nodes[k].theta )
			k = nodes[k].child;
		else
			k = nodes[k].child + 1;
	return nodes[k].label;	
}

// io functions:
void DTree::save(FILE* f){
	if ( p.saveBinary)
	{
		char fname[256]; //?
		fout.open( fname, ios::binary );
		for(int i=0; i<nrnodes; i++)
			fout.write( (char*)(nodes+i), sizeof(Node));	
		fout.close();
	}
	else
	{	
		for(int i=0; i<nrnodes; i++)
		{
			Node* n = nodes+i;
			fprintf(f,"depth:%d index:%d theta:%f label:%f err:%f\n",n->depth, n->index, n->theta, n->label/abs(n->label), n->err);
		}		
	}
}

void DTree::load(FILE* f){
	if ( p.saveBinary )
	{
		char fname[256]; //?
		fin.open( fname, ios::binary );
		for(int i=0; i<nrnodes; i++)
			fin.read( (char*)(nodes+i), sizeof(Node));	
		fin.close();
	}
	else
	{
		for(int i=0; i<nrnodes; i++)
		{
			Node* n = nodes+i;
			fscanf(f,"depth:%d index:%d theta:%f label:%f err:%f\n",&n->depth, &n->index, &n->theta, &n->label, &n->err);
			n->child = 0;
		}
		nodes[0].child = 1; nodes[1].child = 3; nodes[2].child = 5; //?
	}
}

void DTree::setLabel(T lbl){
	for(int i=0; i<nrnodes; i++)
		nodes[i].label *= lbl;
}