#include "somoclu.h"

#include<stdexcept>
#include<list>
#include<omp.h>

struct JuliaMessage{
    unsigned int d;
    float* v1;
    float* v2;
    float  r;
    JuliaMessage(unsigned int td, float* tv1, float* tv2):
        v1(tv1), v2(tv2), d(td), r(0.0f){}
};

class JuliaDistance: public Distance{
private:   
    float (*fp)(void*);
public:
    JuliaDistance(unsigned int d, float (*tfp)(void*)):Distance(d), fp(tfp){}
    virtual ~JuliaDistance(){}
    virtual float operator()(float* v1, float* v2) const {
        JuliaMessage *msg = new JuliaMessage(Dim(), v1, v2);
	fp(msg);
	float r = msg->r;
	delete msg;
	return r;
    }
};

#ifdef _OPENMP

struct JuliaMessageMT: public JuliaMessage{
    JuliaMessageMT* next;
    JuliaMessageMT* prev;
    bool updated;
    JuliaMessageMT(unsigned int td, float* tv1, float* tv2):
        JuliaMessage(td, tv1, tv2), updated(false), prev(this), next(this){}
};

struct my_uv_async_t{
    JuliaMessageMT* head;
    unsigned int    opaque[0];
};

#if defined(_WIN32) && !defined(__MINGW32__)

typedef int (WINAPI *uv_async_send_t)(my_uv_async_t*);

static uv_async_send_t get_uv_async_send(){
    HMODULE handle = LoadLibrary("libjulia.dll");
    if (handle == NULL)
        throw std::runtime_error("Cannot load libjulia.dll");
    WINAPI sym = GetProcAddress(handle, "uv_async_send");
    if (sym == NULL)
        throw std::runtime_error("Cannot find uv_async_send");        
    return (uv_async_send_t)sym;
}

#else

#include<dlfcn.h>

typedef int (*uv_async_send_t)(my_uv_async_t*);

static uv_async_send_t get_uv_async_send(){
    void* handle = dlopen("libjulia.so", RTLD_LAZY);
    if (handle == NULL)
        throw std::runtime_error(dlerror());
    void* sym = dlsym(handle, "uv_async_send");
    if (sym == NULL)
        throw std::runtime_error(dlerror());        
    return (uv_async_send_t)sym;
}
#endif

class JuliaDistanceMT: public JuliaDistance{
private:
    my_uv_async_t* cond;
public:
    JuliaDistanceMT(unsigned int d, void* tcond): JuliaDistance(d, NULL), cond((my_uv_async_t*)tcond){
        cond->head = new JuliaMessageMT(0, 0, 0);
    }
    virtual ~JuliaDistanceMT(){
        if (cond->head)
	    delete cond->head;
	cond->head = 0;
    }
    JuliaMessageMT* get_head() const{ return cond->head;}

    void attach_message(JuliaMessageMT* msg) const{
        JuliaMessageMT* head = get_head();
	//#pragma omp critical(async_head)
	{
	    head->prev->next = msg;
	    msg->prev = head->prev;
	    msg->next = head;
	    head->prev = msg;
	}
    }

    virtual float operator()(float* v1, float* v2) const {
        JuliaMessageMT *msg = new JuliaMessageMT(Dim(), v1, v2);
	//attach_message(msg);
	uv_async_send_t fpt = get_uv_async_send();

	printf("Send Async Msg to: %x on thread %d\n", cond, omp_get_thread_num());
	int ret = fpt(cond);
	if (ret < 0)
	    throw std::runtime_error("Unable to pass message to Julia.");
#if 0	
	while(true){
#pragma omp flush (msg)
            if (msg->updated) break;
        }
#endif
	float r = 1.0; msg->r;
	//delete msg;
	return r;
    } 
};

void* detach_messages(void* p){
    my_uv_async_t* async = (my_uv_async_t*)(p);
    JuliaMessageMT* head = async->head;
    void* ret = 0;
    //#pragma omp critical(async_head) 
    {
        if (head->next != head){
            head->next->prev = head->prev;
	    head->prev->next = head->next;
	    ret = (void*)head->next;
	    head->next = head->prev = head;
	}
    }
    return ret;

}

void update_messages(void* p){
    JuliaMessageMT* cur = (JuliaMessageMT*)(p);
    while (true){
        cur->prev->next = cur->next;
	cur->next->prev = cur->prev;
	bool &updated = cur->updated;
	updated = true;
	//#pragma omp flush (cur)	
	if (cur == cur->next)
	    break;
	else
	    cur = cur->next;
    }
    printf("Updated Message\n");
}

static JuliaDistance* GetJuliaDistance(unsigned int d, void* cond){
    return new JuliaDistanceMT(d, cond);
}

#else
static JuliaDistance* GetJuliaDistance(unsigned int d, void* fp){
    return new JuliaDistance(d, (float (*)(void*))(fp));
}
#endif

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
