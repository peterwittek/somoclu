#include "somoclu.h"
#include <omp.h>

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
    compute_distance_t compute_distance;
public:
    JuliaDistance(unsigned int d, compute_distance_t tcd):
        Distance(d), compute_distance(tcd){}
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
    virtual void compute(JuliaMessage* msg) const{compute_distance(msg);}
};

struct JuliaMessageMT: public JuliaMessage{
    JuliaMessageMT* next;
    bool updated;
    JuliaMessageMT(unsigned int td, float* tv1, float* tv2):
        JuliaMessage(td, tv1, tv2), updated(false), next(NULL){
    }
};

class JuliaDistanceMT: public JuliaDistance{
private:
    void* cond;
    JuliaMessageMT *head;
    bool alive, master_done;
public:
    JuliaDistanceMT(unsigned int d, compute_distance_t tfp):
        JuliaDistance(d, tfp), master_done(false), alive(true),
        head(new JuliaMessageMT(0, 0, 0)){}
    virtual ~JuliaDistanceMT(){
        // Ensure the precompute thread is exited after completing pending tasks.
        // Unless that is carried out already allocated messages will not be cleaned.
        // Then clean up the head.
        // This cannot be called from the master thread.
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
        // This needs to be run under the master thread in a loop to ensure no other
        // thread calls back into Julia distance method. If they do Julia GC will
        // force a segment fault.
        // This does not unlink the messages as messages are to be cleaned up by
        // the owning threads after receiving the results. Mere detaching from head
        // ensures new threads requesting for compute are not starved.
        do {
            JuliaMessageMT* msg = detach_messages();
            if (msg != NULL){
                do{
                    compute_distance(msg);
                    bool &updated = msg->updated;
                    updated = true;
#pragma omp flush (updated)             
                }while(msg->next != NULL);
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

    // Messages are attached to the head node for computation by master thread
    // when scheduled. 
    void attach_message(JuliaMessageMT* msg) const{
#pragma omp critical(async_head)
        {
            msg->next = head->next;
            head->next = msg;
        }
    }
    
    // Detach the head quickly so that other threads who need to place their
    // computations can do so without being made to wait for the whole
    // computation chain to complete.
    JuliaMessageMT* detach_messages(){
        JuliaMessageMT* ret = NULL;
#pragma omp critical(async_head) 
        if (head->next != NULL){
            ret = head->next;
            head->next = NULL;
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
    string scaleCooling  = (_scaleCooling == 0) ? "linear" : "exponential";
    string mapType       = (_mapType == 0)      ? "planar" : "toroid";
    string gridType      = (_gridType == 0)     ? "square" : "hexagonal";


    Distance* pdist = (get_distance == NULL)?
        (Distance*)new EuclideanDistance(nDimensions):
        (Distance*)GetJuliaDistance(nDimensions, get_distance);

    // In case omp is enabled and custom distance function is used split the
    // execution into 2 blocks. Make the main thread to wait on the compute
    // distance callback requests into Julia. While the other thread executes
    // the training code.
    // When omp is not enabled or custom distance function is not used, the
    // parallel block does not kick-in and execution follows the sequence.
#pragma omp parallel num_threads(2) default(shared) if(get_distance != NULL)
    {
#pragma omp master
        if (get_distance)
            pdist->precompute();
#pragma omp single
        {
             som map = {
              .nSomX = nSomX,
              .nSomY = nSomY,
              .nDimensions = nDimensions,
              .nVectors = nVectors,
              .mapType = mapType,
              .gridType = gridType,
              .get_distance = *pdist,
              .uMatrix = uMatrix,
              .codebook = codebook,
              .bmus = globalBmus};

            train(0, data, NULL, map, nVectors, nEpoch, radius0, radiusN,
                  radiusCooling, scale0, scaleN, scaleCooling,
                  kernelType, compact_support, gaussian, std_coeff, verbose);
            calculateUMatrix(map);
            delete pdist;
        }
    }
}
