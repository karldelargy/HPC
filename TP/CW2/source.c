#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 729
#define reps 100
#include <omp.h> 

double a[N][N], b[N][N], c[N];
int jmax[N];  


void init1(void);
void init2(void);
void runloop(int); 
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);


int main(int argc, char *argv[]) { 

  double start1,start2,end1,end2;
  int r;

  init1(); 

  start1 = omp_get_wtime(); 

  for (r=0; r<reps; r++){ 
    runloop(1);
  } 

  end1  = omp_get_wtime();  

  valid1(); 

  printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1)); 


  init2(); 

  start2 = omp_get_wtime(); 

  for (r=0; r<reps; r++){ 
    runloop(2);
  } 

  end2  = omp_get_wtime(); 

  valid2(); 

  printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2)); 
  
  printf("%f\t%f\n",(float)(end1-start1), (float)(end2-start2)); 

} 

void init1(void){
  int i,j; 

  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      a[i][j] = 0.0; 
      b[i][j] = 3.142*(i+j); 
    }
  }

}

void init2(void){ 
  int i,j, expr; 

  for (i=0; i<N; i++){ 
    expr =  i%( 3*(i/30) + 1); 
    if ( expr == 0) { 
      jmax[i] = N;
    }
    else {
      jmax[i] = 1; 
    }
    c[i] = 0.0;
  }

  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      b[i][j] = (double) (i*j+1) / (double) (N*N); 
    }
  
  }
 
} 

struct block //implementing the struct outside the function 
{
  int high;
  int remaining;
};

void runloop(int loopid)  {

 struct block* blocks; //Declaring the struct

 #pragma omp parallel default(none) shared(loopid, blocks)  //start of parallel region
  {
    int myid  = omp_get_thread_num();
    int nthreads = omp_get_num_threads(); 
    
    #pragma omp single 
    {
    	blocks=(struct block*)malloc(sizeof(struct block)*nthreads); //initialising the struct
    }
    
    int ipt = (int) ceil((double)N/(double)nthreads); 
    int lo = myid*ipt;
   
    
    int hi = (myid+1)*ipt;
    if (hi > N) hi = N; 
    int r = hi - lo;
    int num_iters= (int)ceil((double)r/(double)nthreads);
    int most_work;
    int loc_most_work;
    int max=0;
    
    #pragma omp critical  //members of the struct must be updated within critical regions to ensure synchronisation and avoid race condition
    {
		blocks[myid].high=hi;
		blocks[myid].remaining=r;
		printf("Thread %d has remaining %d and num iters is%d\n",myid, blocks[myid].remaining,num_iters);
	}
    
    //each thread does its own iterations in this while loop
    while(blocks[myid].remaining>0){ 
    	//critical region to update struct members
    	#pragma omp critical
    	{
			num_iters= (int)ceil((double)(blocks[myid].remaining)/(double)nthreads);
			lo=blocks[myid].high - blocks[myid].remaining;
			hi=lo + num_iters;
			blocks[myid].remaining = blocks[myid].remaining - num_iters;
			num_iters= (int)ceil((double)(blocks[myid].remaining)/(double)nthreads);
		
		}
		
        //printing working iterations
		printf("Thread %d iterating from %d to %d with %d remaining\n", myid, lo, hi, blocks[myid].remaining );
		//run through the loop
		if(blocks[myid].remaining>=0){
			switch (loopid) { 
				  case 1: loop1chunk(lo,hi); break;
				  case 2: loop2chunk(lo,hi); break;
			  }
		}
	
    }
    
    //do while loop for work stealing from most load thread by idle threads
    
    do {

		loc_most_work=-1;
		most_work=0;
		int remaining;
		
		//updating members and finding how much work the most loaded thread has, and which is most loaded
		//which also needs to be done inside a critical region
		
		#pragma omp critical
			{
			
				if(blocks[myid].remaining==0){
			
					int i;
					for(i=0;i<nthreads;i++){
						if (blocks[i].remaining>most_work){
							most_work = blocks[i].remaining;
							loc_most_work=i;
						}
					}
					if(loc_most_work>=0){
						if(most_work>=0){
				
					
							num_iters= (int)ceil((double)(blocks[loc_most_work].remaining)/(double)nthreads);
							lo=blocks[loc_most_work].high - blocks[loc_most_work].remaining;
							hi=lo + num_iters;
							if (hi > N) hi = N;
							blocks[loc_most_work].remaining -= num_iters;
						}
					}
				}
				
			}
			
			//ensuring synchronisation
			
			if(myid>=0){
				if(loc_most_work>=0){
					switch (loopid) { 
					  		case 1: loop1chunk(lo,hi); break;
					 		case 2: loop2chunk(lo,hi); break;
					}
					//printing the work steals
					printf("Thread %d stealing from thread %d iterating %d to %d with %d remaining\n",myid, loc_most_work, lo, hi, blocks[loc_most_work].remaining);
				}
			}
			
		}while(most_work>0); //iterations only done while other threads have work left to do
    

   
	}
 free(blocks); //freeing blocks so there are no memory leakages
}

void loop1chunk(int lo, int hi) { 
  int i,j; 
  
  for (i=lo; i<hi; i++){ 
    for (j=N-1; j>i; j--){
      a[i][j] += cos(b[i][j]);
    } 
  }

} 



void loop2chunk(int lo, int hi) {
  int i,j,k; 
  double rN2; 

  rN2 = 1.0 / (double) (N*N);  

  for (i=lo; i<hi; i++){ 
    for (j=0; j < jmax[i]; j++){
      for (k=0; k<j; k++){ 
	c[i] += (k+1) * log (b[i][j]) * rN2;
      } 
    }
  }

}

void valid1(void) { 
  int i,j; 
  double suma; 
  
  suma= 0.0; 
  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      suma += a[i][j];
    }
  }
  printf("Loop 1 check: Sum of a is %lf\n", suma);

} 


void valid2(void) { 
  int i; 
  double sumc; 
  
  sumc= 0.0; 
  for (i=0; i<N; i++){ 
    sumc += c[i];
  }
  printf("Loop 2 check: Sum of c is %f it should be  -2524264.460320\n", sumc);
} 
 


