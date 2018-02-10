#include "somoclu.h"

#include<stdexcept>
#include<list>
#include<iomanip>
#include<iostream>

using namespace std;

struct JuliaMessage{
	float  r;
	unsigned int d;
	float* v1;
	float* v2;
	JuliaMessage(unsigned int td, float* tv1, float* tv2):
		v1(tv1), v2(tv2), d(td), r(0.0f){}
};

typedef float (*compute_distance_t)(void*); 

class JuliaDistance: public Distance{
protected:   
	compute_distance_t cd;
public:
	JuliaDistance(unsigned int d, compute_distance_t tcd):Distance(d), cd(tcd){}
	virtual ~JuliaDistance(){}
	virtual float operator()(float* v1, float* v2) const {
		JuliaMessage *msg = createMessage(Dim(), v1, v2);
		compute(msg);
		float r = msg->r;
		delete msg;
		return r;
	}
	virtual JuliaMessage* createMessage(unsigned int d, float* v1, float* v2) const{
		return new JuliaMessage(Dim(), v1, v2);
	}
	virtual void precompute(){}
	virtual void compute(JuliaMessage* msg) const{cd(msg);}
};

struct JuliaMessageMT: public JuliaMessage{
	JuliaMessageMT* next;
	JuliaMessageMT* prev;
	bool updated;
	JuliaMessageMT(unsigned int td, float* tv1, float* tv2):
		JuliaMessage(td, tv1, tv2), updated(false), prev(this), next(this){
	}
};

class JuliaDistanceMT: public JuliaDistance{
private:
	void* cond;
	JuliaMessageMT *head;
	mutable int count;
	bool alive, master_done;
public:
	JuliaDistanceMT(unsigned int d, compute_distance_t tfp):
		JuliaDistance(d, tfp), count(0), master_done(false), alive(true){
		head = new JuliaMessageMT(0, 0, 0);
	}
	virtual ~JuliaDistanceMT(){
		alive = false;
#pragma omp flush (alive)
		while(true){
#pragma omp flush (master_done)
			if (master_done) break;
#pragma omp taskyield
		}
		delete head;
	}
	
	virtual JuliaMessage* createMessage(unsigned int d, float* v1, float* v2) const{
		return new JuliaMessageMT(Dim(), v1, v2);
	}

	void precompute(){
		do {
			JuliaMessageMT* msg = detach_messages();
			if (msg != NULL){
				do{
					cd(msg);
					bool &updated = msg->updated;
					updated = true;
#pragma omp flush (updated)				
				}while(msg->next != msg);
			}
#pragma omp taskyield
#pragma omp flush (alive)
		} while(alive);
		master_done = true;
#pragma omp flush (master_done)
	}

	void compute(JuliaMessage* msg) const{
		JuliaMessageMT* msgMT = (JuliaMessageMT*)msg;
		attach_message(msgMT);
		while(true){
			bool &updated = msgMT->updated;
#pragma omp flush (updated)	
			if (updated) break;
#pragma omp taskyield			
		}
	}

	void attach_message(JuliaMessageMT* msg) const{
#pragma omp critical(async_head)
		{
			head->prev->next = msg;
			msg->prev = head->prev;
			head->prev = msg;
			msg->next = head;
		}
	}
	JuliaMessageMT* detach_messages(){
		JuliaMessageMT* ret = 0;
#pragma omp critical(async_head) 
		{
			if (head->next != head){
				head->next->prev = head->prev;
				head->prev->next = head->next;
				ret = head->next;
				head->next = head->prev = head;
			}
		}
		return ret;
	}	
};

static JuliaDistance* GetJuliaDistance(unsigned int d, void* fp){
#ifdef _OPENMP
	return new JuliaDistanceMT(d, (compute_distance_t)fp);
#else
	return new JuliaDistance(d, (compute_distance_t)(fp));
#endif
}

extern "C"
void julia_train(float *data, int data_length, unsigned int nEpoch,
				 unsigned int nSomX, unsigned int nSomY,
				 unsigned int nDimensions, unsigned int nVectors,
				 float radius0, float radiusN, unsigned int _radiusCooling,
				 float scale0, float scaleN, unsigned int _scaleCooling,
				 unsigned int kernelType, unsigned int _mapType,
				 unsigned int _gridType, bool compact_support, bool gaussian,
				 float std_coeff, unsigned int verbose,
				 float *codebook, int codebook_size,
				 int *globalBmus, int globalBmus_size,
				 float *uMatrix, int uMatrix_size,
				 void* get_distance){
  
	string radiusCooling = (_radiusCooling == 0)? "linear" : "exponential";
	string scaleCooling	 = (_scaleCooling == 0) ? "linear" : "exponential";
	string mapType		 = (_mapType == 0)		? "planar" : "toroid";
	string gridType		 = (_gridType == 0)		? "square" : "hexagonal";


	Distance* pdist = (get_distance == NULL)?
		(Distance*)new EuclideanDistance(nDimensions):
		(Distance*)GetJuliaDistance(nDimensions, get_distance);

#pragma omp parallel num_threads(2) default(shared) if(get_distance != NULL)
	{
#pragma omp master
		{
			if (get_distance)
				pdist->precompute();
		}
#pragma omp single
		{
			train(0, data, NULL, codebook, globalBmus, uMatrix, nSomX, nSomY,
				  nDimensions, nVectors, nVectors,
				  nEpoch, radius0, radiusN, radiusCooling,
				  scale0, scaleN, scaleCooling,
				  kernelType, mapType,
				  gridType, compact_support, gaussian, std_coeff, verbose,
				  *pdist
#ifdef CLI
				  , "", 0);
#else
			);
#endif
	        calculateUMatrix(uMatrix, codebook, nSomX, nSomY, nDimensions, mapType,
							 gridType, *pdist);
			delete pdist;
	    }
    }
}
