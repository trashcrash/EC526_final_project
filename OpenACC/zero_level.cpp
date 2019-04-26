#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define RESGOAL 1E-6
#define NLEV 0                                      // If 0, only one level
#define PERIOD 100
#define PI 3.141592653589793
#define TSTRIDE 10
#define N_PER_LEV 10                                // Iterate 10 times for each level
#define SIZE 514

static const double scale = 1.0/(4.0*TSTRIDE + 1);
int relax(double *phi, double *phi_old, double *res, double *tmp);
void v_cycle(double *phi, double *phi_old, double *res, double *tmp);
//void proj_res(double *res_c, double *rec_f, double *phi_f, double *phi_old_f);
//void inter_add(double *phi_f, double *phi_c);
void GetResRoot(double *phi, double *phi_old, double *res, double *resmag);

int main() {
    double *phi, *res, *phi_old, *tmp, *resmag;
    #pragma acc init
    int i;

    // Initialize phi and res for each level
    phi = (double *) malloc(SIZE*SIZE * sizeof(double));
    phi_old = (double *) malloc(SIZE*SIZE * sizeof(double));
    res = (double *) malloc(SIZE*SIZE * sizeof(double));
    tmp = (double *)malloc((SIZE-2)*(SIZE-2)*sizeof(double));
    resmag = (double *)malloc(sizeof(double));
    for (i = 0; i < SIZE*SIZE; i++) {
        phi[i] = 0.0;
        phi_old[i] = 0.0;
        res[i] = 0.0;
    }
    for (i = 0; i < (SIZE-2)*(SIZE-2); i++)
        tmp[i] = 0.0;
    resmag[0] = 1.0;
    res[SIZE/2 + (SIZE/2)*SIZE] = 1.0*TSTRIDE*scale;
    // iterate to solve
    int ncycle = 1;
    int t = 0;
    //resmag = GetResRoot(phi, phi_old, res);
    printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);

    std::chrono::time_point<std::chrono::steady_clock> begin_time =
    std::chrono::steady_clock::now();
 
    // Total time steps = PERIOD
    for (t = 0; t < PERIOD; t++) {
    #pragma acc data copy(phi[0:SIZE*SIZE]) copy(res[0:SIZE*SIZE]) copy(phi_old[0:SIZE*SIZE]) copy(tmp[0:(SIZE-2)*(SIZE-2)]) copy(resmag[0:1])
    {
        while (resmag[0] > RESGOAL) {
            ncycle += 1; 
            v_cycle(phi, phi_old, res, tmp);
            GetResRoot(phi, phi_old, res, resmag);
            printf("At the %d cycle the mag residue is %g \n",ncycle,resmag[0]);
        }
    }
        // Source varies with time
        res[SIZE/2 + (SIZE/2)*SIZE] = 1.0*TSTRIDE*scale*(1+sin(2.0*PI*t/PERIOD));
        resmag[0] = 1.0;
        //#pragma acc data copyin(res[0:SIZE*SIZE])
        ncycle = 0;
        for (i = 0; i < SIZE*SIZE; i++) {
            phi_old[i] = phi[i];
        }
    }
    
    std::chrono::time_point<std::chrono::steady_clock> end_time =
    std::chrono::steady_clock::now();
    std::chrono::duration<double> difference_in_time = end_time - begin_time;
    double difference_in_seconds = difference_in_time.count();
    printf("Time spent: %f\n", difference_in_seconds);
    
    return 0;
}

int relax(double *phi, double *phi_old, double *res, double *tmp) {
    #pragma acc data present(phi[0:SIZE*SIZE]) present(phi_old[0:SIZE*SIZE]) present(res[0:SIZE*SIZE]) present(tmp[0:(SIZE-2)*(SIZE-2)])
    {
    int i, x, y;
    for (i = 0; i < N_PER_LEV; i++) {   
        #pragma acc parallel loop collapse(2)
        for (x = 1; x < SIZE-1; x++) {
            for (y = 1; y < SIZE-1; y++) {
                tmp[y-1+(x-1)*(SIZE-2)] = 0.5*(res[y + x*SIZE]
                            + TSTRIDE*scale*(phi[y+1 + x*SIZE] + phi[y-1 + x*SIZE] 
                            + phi[y + (x+1)*SIZE] + phi[y + (x-1)*SIZE])
                            + scale*phi_old[y + x*SIZE] + phi[y + x*SIZE]);
            }
        }
                // a coarse phi is the error of a fine phi
        #pragma acc parallel loop collapse(2)
        for (x = 1; x < SIZE-1; x++) {
            for (y = 1; y < SIZE-1; y++) {
                phi[y + x*SIZE] = tmp[y-1 + (x-1)*(SIZE-2)];
            }
        }
    }
}
    return 0;
}

void v_cycle(double *phi, double *phi_old, double *res, double *tmp) {
    //#pragma acc data present(phi[0:SIZE*SIZE]) present(phi_old[0:SIZE*SIZE]) present(res[0:SIZE*SIZE]) present(tmp[0:(SIZE-2)*(SIZE-2)])
    relax(phi, phi_old, res, tmp);
}

void GetResRoot(double *phi, double *phi_old, double *res, double *resmag) {
    #pragma acc data present(phi[0:SIZE*SIZE]) present(phi_old[0:SIZE*SIZE]) present(res[0:SIZE*SIZE]) present(resmag[0:1])
    {
    int x, y;
    double residue;
    double ResRoot = 0.0;
    #pragma acc parallel loop collapse(2) reduction(+:ResRoot)
    for (x = 1; x < SIZE-1; x++) {
        for (y = 1; y < SIZE-1; y++) {
            residue = res[y + x*SIZE]/scale/TSTRIDE - phi[y + x*SIZE]/scale/TSTRIDE  
                    + (phi[(y+1) + x*SIZE] + phi[(y-1) + x*SIZE] 
                    + phi[y + (x+1)*SIZE] + phi[y + (x-1)*SIZE])
                    + phi_old[y + x*SIZE]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
    }
    resmag[0] = sqrt(ResRoot);
}
    return;
}
