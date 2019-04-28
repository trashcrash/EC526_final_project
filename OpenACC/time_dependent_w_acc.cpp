#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define RESGOAL 1E-6
#define NLEV 7                                      // If 0, only one level
#define PERIOD 100
#define PI 3.141592653589793
#define TSTRIDE 10
#define N_PER_LEV 10                                // Iterate 10 times for each level
#define EDGE 514
#define SIZE 353648

static const double scale = 1.0/(4.0*TSTRIDE + 1);
int relax(double *phi, double *phi_old, double *res, double *tmp, int L, int startpt);
void w_cycle(double *phi, double *phi_old, double *res, double *tmp, int *edge_lev, int *startpt, int this_lev, int flag);
void proj_res(double *res, double *phi, double *phi_old, double *tmp, int *edge_lev, int *startpt, int lev);
void inter_add(double *phi, int *edge_lev, int *startpt, int lev);
void GetResRoot(double *phi, double *phi_old, double *res, double *resmag);

int main() {
for (int iter = 0; iter < 10; iter++) {
    double *phi, *res, *phi_old, *tmp, *resmag;
    int *edge_lev, *startpt;
    FILE* output;
    output = fopen("acc_w_512_7lev.dat", "a");
    #pragma acc init
    int i;

    // Initialize phi and res for each level
    phi = (double *) malloc(SIZE * sizeof(double));
    phi_old = (double *) malloc(SIZE * sizeof(double));
    res = (double *) malloc(SIZE * sizeof(double));
    tmp = (double *)malloc(SIZE * sizeof(double));
    resmag = (double *)malloc(sizeof(double));
    edge_lev = (int *)malloc((NLEV+1)*sizeof(int));
    startpt = (int *)malloc((NLEV+1)*sizeof(int));
    for (i = 0; i < SIZE; i++) {
        phi[i] = 0.0;
        phi_old[i] = 0.0;
        res[i] = 0.0;
        tmp[i] = 0.0;
    }
    resmag[0] = 1.0;
    res[EDGE/2 + (EDGE/2)*EDGE] = 1.0*TSTRIDE*scale;
    edge_lev[0] = EDGE;
    for (i = 1; i < NLEV+1; i++) {
        edge_lev[i] = edge_lev[i-1]/2+1;
    }
    startpt[0] = 0;
    for (i = 1; i < NLEV+1; i++) {
        startpt[i] = startpt[i-1]+edge_lev[i-1]*edge_lev[i-1];
    }
    // iterate to solve
    int ncycle = 1;
    int t = 0;
    //resmag = GetResRoot(phi, phi_old, res);
    printf("At the %d cycle the mag residue is %g \n",ncycle,resmag[0]);

    std::chrono::time_point<std::chrono::steady_clock> begin_time =
    std::chrono::steady_clock::now();
 
    // Total time steps = PERIOD
    for (t = 0; t < PERIOD; t++) {
    #pragma acc data copy(phi[0:SIZE]) copy(res[0:SIZE]) copy(phi_old[0:SIZE]) copy(tmp[0:SIZE]) copy(resmag[0:1]) copy(startpt[0:NLEV+1]) copy(edge_lev[0:NLEV+1])
    {
        while (resmag[0] > RESGOAL) {
            ncycle += 1; 
            w_cycle(phi, phi_old, res, tmp, edge_lev, startpt, 0, 1);
            GetResRoot(phi, phi_old, res, resmag);
            printf("At the %d cycle the mag residue is %g \n",ncycle,resmag[0]);
        }
    }
        // Source varies with time
        res[EDGE/2 + (EDGE/2)*EDGE] = 1.0*TSTRIDE*scale*(1+sin(2.0*PI*t/PERIOD));
        resmag[0] = 1.0;
        //#pragma acc data copyin(res[0:SIZE*SIZE])
        ncycle = 0;
        for (i = 0; i < EDGE*EDGE; i++) {
            phi_old[i] = phi[i];
        }
    }
    
    std::chrono::time_point<std::chrono::steady_clock> end_time =
    std::chrono::steady_clock::now();
    std::chrono::duration<double> difference_in_time = end_time - begin_time;
    double difference_in_seconds = difference_in_time.count();
    printf("Time spent: %f\n", difference_in_seconds);
    fprintf(output, "%.10f\n", difference_in_seconds);
    fclose(output);
}    
    return 0;
}

int relax(double *phi, double *phi_old, double *res, double *tmp, int L, int startpt) {
    #pragma acc data present(phi[0:SIZE]) present(phi_old[0:SIZE]) present(res[0:SIZE]) present(tmp[0:SIZE])
    {
    int i, x, y;
    for (i = 0; i < N_PER_LEV; i++) {   
        #pragma acc parallel loop collapse(2)
        for (x = 1; x < L-1; x++) {
            for (y = 1; y < L-1; y++) {
                tmp[startpt + y-1+(x-1)*(L-2)] = 0.5*(res[startpt + y + x*L]
                            + TSTRIDE*scale*(phi[startpt + y+1 + x*L] + phi[startpt + y-1 + x*L] 
                            + phi[startpt + y + (x+1)*L] + phi[startpt + y + (x-1)*L])
                            + scale*phi_old[startpt + y + x*L] + phi[startpt + y + x*L]);
            }
        }
                // a coarse phi is the error of a fine phi
        #pragma acc parallel loop collapse(2)
        for (x = 1; x < L-1; x++) {
            for (y = 1; y < L-1; y++) {
                phi[startpt + y + x*L] = tmp[startpt + y-1 + (x-1)*(L-2)];
            }
        }
    }
}
    return 0;
}

void w_cycle(double *phi, double *phi_old, double *res, double *tmp, int *edge_lev, int *startpt, int this_lev, int flag) {
    //#pragma acc data present(phi[0:SIZE*SIZE]) present(phi_old[0:SIZE*SIZE]) present(res[0:SIZE*SIZE]) present(tmp[0:(SIZE-2)*(SIZE-2)])
    if (this_lev == NLEV) {
        relax(phi, phi_old, res, tmp, edge_lev[this_lev], startpt[this_lev]);
    }
    else {
        relax(phi, phi_old, res, tmp, edge_lev[this_lev], startpt[this_lev]);
        proj_res(res, phi, phi_old, tmp, edge_lev, startpt, this_lev);
        w_cycle(phi, phi_old, res, tmp, edge_lev, startpt, this_lev+1, 0);
        w_cycle(phi, phi_old, res, tmp, edge_lev, startpt, this_lev+1, 1);
        inter_add(phi, edge_lev, startpt, this_lev+1);
        if (flag != 0) {
            relax(phi, phi_old, res, tmp, edge_lev[this_lev], startpt[this_lev]);
        }
    }
}

void proj_res(double *res, double *phi, double *phi_old, double *tmp, int *edge_lev, int *startpt, int lev) {  
    #pragma acc data present(phi[0:SIZE]) present(phi_old[0:SIZE]) present(res[0:SIZE]) present(tmp[0:SIZE]) present(edge_lev[0:NLEV+1]) present(startpt[0:NLEV+1])
    {
    int x, y;
    int L = edge_lev[lev];
    int Lc = edge_lev[lev+1];
  
    //get residue
    #pragma acc parallel loop collapse(2)
    for(x = 1; x < L-1; x++)
        for(y = 1; y < L-1; y++)
            tmp[startpt[lev] + x*L + y] = res[startpt[lev] + x*L + y] - phi[startpt[lev] + x*L + y] 
                                      + scale*phi_old[startpt[lev] + x*L + y] 
                                      + TSTRIDE*scale*
                                       (phi[startpt[lev] + y+1 + x*L]
                                      + phi[startpt[lev] + y-1 + x*L]
                                      + phi[startpt[lev] + y + (x+1)*L]
                                      + phi[startpt[lev] + y + (x-1)*L]);

    //project residue
    #pragma acc parallel loop collapse(2)
    for(x = 1; x < Lc-1; x++)
        for(y = 1; y < Lc-1; y++)
            res[startpt[lev+1] + x*Lc + y] = 0.25*(tmp[startpt[lev] + (2*x-1)*L + (2*y-1)] 
                                                 + tmp[startpt[lev] + (2*y) + (2*x-1)*L] 
                                                 + tmp[startpt[lev] + (2*y-1) + (2*x)*L] 
                                                 + tmp[startpt[lev] + (2*y) + (2*x)*L]);
    }
    return;
}

void inter_add(double *phi, int *edge_lev, int *startpt, int lev) {  
    #pragma acc data present(phi[0:SIZE]) present(edge_lev[0:NLEV+1]) present(startpt[0:NLEV+1])
    {
    int x, y;
    int L = edge_lev[lev-1];
    int Lc = edge_lev[lev]; 

    #pragma acc parallel loop collapse(2)
    for (x = 1; x < Lc-1; x++)
        for (y = 1; y < Lc-1; y++) {

            // Add the error back to phi
            phi[startpt[lev-1] + (2*y-1) + (2*x-1)*L] += phi[startpt[lev] + y + x*Lc];
            phi[startpt[lev-1] + (2*y) + (2*x-1)*L] += phi[startpt[lev] + y + x*Lc];
            phi[startpt[lev-1] + (2*y-1) + (2*x)*L] += phi[startpt[lev] + y + x*Lc];
            phi[startpt[lev-1] + (2*y) + (2*x)*L] += phi[startpt[lev] + y + x*Lc];
        }
    //set to zero so phi = error 
    #pragma acc parallel loop collapse(2)
    for (x = 1; x < Lc-1; x++)
        for (y = 1; y < Lc-1; y++)
            phi[startpt[lev] + y + x*Lc] = 0.0;
    }
    return;
}

void GetResRoot(double *phi, double *phi_old, double *res, double *resmag) {
    #pragma acc data present(phi[0:SIZE]) present(phi_old[0:SIZE]) present(res[0:SIZE]) present(resmag[0:1])
    {
    int x, y;
    double residue;
    double ResRoot = 0.0;
    #pragma acc parallel loop collapse(2) reduction(+:ResRoot)
    for (x = 1; x < EDGE-1; x++) {
        for (y = 1; y < EDGE-1; y++) {
            residue = res[y + x*EDGE]/scale/TSTRIDE - phi[y + x*EDGE]/scale/TSTRIDE  
                    + (phi[(y+1) + x*EDGE] + phi[(y-1) + x*EDGE] 
                    + phi[y + (x+1)*EDGE] + phi[y + (x-1)*EDGE])
                    + phi_old[y + x*EDGE]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
    }
    resmag[0] = sqrt(ResRoot);
}
    return;
}
