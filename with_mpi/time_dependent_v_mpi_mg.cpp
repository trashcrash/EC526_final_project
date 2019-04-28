#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
#include <chrono>

#define RESGOAL 1E-6
#define NLEV 0                                     // If 0, only one level
#define PERIOD 100
#define PI 3.141592653589793
#define TSTRIDE 10
#define N_PER_LEV 10                                // Iterate 10 times for each level

typedef struct{
    int N;
    int Lmax;
    int size[20];
    int localSize[20];
    int up_limit[20];
    int down_limit[20];
    int left_limit[20];
    int right_limit[20];
    // double a[20];
    double m_square;
    double scale;
} param;

// Useful globals
int world_size; // number of processes
int my_rank; // my process number
int lengthOfEdge;

void relax(double *phi, double *phi_old, double *res, int lev, param p);
void proj_res(double *res_c, double *rec_f, double *phi_f, double *phi_old_f, int lev, param p);
void inter_add(double *phi_f, double *phi_c, int lev,param p);
double GetResRoot(double *phi, double *phi_old, double *res, int lev, param p);
void v_cycle(double **phi, double **phi_old, double **res, param p);
void getR(double *res, double *phi, double *phi_old, double *r, int lev, param p);

int main(int argc, char** argv) {     

    // Initialize MPI
    MPI_Init(&argc, &argv);
   
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   
    // Get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   
    double *phi[20], *res[20], *phi_old[20];
    param p;
    int i, j, lev;
    lengthOfEdge = 2;
  
    // Initialize parameters
    p.Lmax = 8;
    p.N = 2*(int)pow(2,p.Lmax)+2;
    p.m_square = 0.0;                                     // Scaling parameter, a small number

    // Exception control
    if (NLEV > p.Lmax) { 
        printf("ERROR More levels than available in lattice! \n");
        return 0;
    }
  
    printf("\n V cycle for %d by %d lattice with NLEV = %d out of max %d \n", p.N, p.N, NLEV, p.Lmax); 
  
    // Initialize arrays
    p.size[0] = p.N;
    p.localSize[0]=p.size[0]/2;
    p.left_limit[0] = (my_rank % lengthOfEdge == 0) ? 1 : 0;
    p.right_limit[0] = (my_rank % lengthOfEdge == lengthOfEdge-1) ? p.localSize[0]-2 : p.localSize[0]-1;
    p.up_limit[0] = (my_rank / lengthOfEdge == 0) ? 1 : 0;
    p.down_limit[0] = (my_rank / lengthOfEdge == lengthOfEdge-1) ? p.localSize[0]-2 : p.localSize[0]-1;

    // p.a[0] = 1.0;                                    // Stride, may not be needed
    p.scale = 1.0/(4.0*TSTRIDE + 1);

    for (lev = 1; lev < p.Lmax+1; lev++) {
        p.size[lev] = p.size[lev-1]/2+1;
        p.localSize[lev]=p.size[lev]/2;
        p.left_limit[lev] = (my_rank % lengthOfEdge == 0) ? 1 : 0;
        p.right_limit[lev] = (my_rank % lengthOfEdge == lengthOfEdge-1) ? p.localSize[lev]-2 : p.localSize[lev]-1;
        p.up_limit[lev] = (my_rank / lengthOfEdge == 0) ? 1 : 0;
        p.down_limit[lev] = (my_rank / lengthOfEdge == lengthOfEdge-1) ? p.localSize[lev]-2 : p.localSize[lev]-1;
    }

    // Initialize phi and res for each level
    for (lev = 0; lev < p.Lmax+1; lev++) {
        phi[lev] = (double *) malloc(p.localSize[lev]*p.localSize[lev] * sizeof(double));
        phi_old[lev] = (double *) malloc(p.localSize[lev]*p.localSize[lev] * sizeof(double));
        res[lev] = (double *) malloc(p.localSize[lev]*p.localSize[lev] * sizeof(double));

        for (i = 0; i < p.localSize[lev]*p.localSize[lev]; i++) {
            phi[lev][i] = 0.0;
            phi_old[lev][i] = 0.0;
            res[lev][i] = 0.0;
        }
    }  

    int mySourceRank = 3;
    if(my_rank == mySourceRank) // Now as a fixed source.
        res[0][0] = 1.0*TSTRIDE*p.scale;   // Unit point source in middle of N by N lattice 
  
    // iterate to solve
    double resmag = 1.0;                            // Not rescaled.
    int ncycle = 1; 
    int t = 0;
    resmag = GetResRoot(phi[0], phi_old[0], res[0], 0, p);
    printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);
    // TIMING LINE 1: Get the starting timestamp. 
    std::chrono::time_point<std::chrono::steady_clock> begin_time =
    std::chrono::steady_clock::now();
 
 
    // Total time steps = PERIOD
    while (t < PERIOD) {
        ncycle += 1; 
        v_cycle(phi, phi_old, res, p);
        resmag = GetResRoot(phi[0], phi_old[0], res[0], 0, p);
        if(my_rank==0)
            printf("At the %d cycle the mag residue is %g\n",ncycle,resmag);


        // Source varies with time
        if (resmag < RESGOAL) {
            // TIMING LINE 2: Get the ending timestamp.
            std::chrono::time_point<std::chrono::steady_clock> end_time =
            std::chrono::steady_clock::now();

            // TIMING LINE 3: Compute the difference.
            std::chrono::duration<double> difference_in_time = end_time - begin_time;

            // TIMING LINE 4: Get the difference in seconds.
            double difference_in_seconds = difference_in_time.count();
            if(my_rank==0) printf("t is %d, time is %.15f\n",t,difference_in_seconds);

            t += 1;
            if(my_rank == mySourceRank)
                res[0][0] = 1.0*TSTRIDE*p.scale*(1+sin(2.0*PI*t/(PERIOD)));
            ncycle = 0;
            for (lev = 0; lev < p.Lmax+1; lev++) {
                for (i = 0; i < p.localSize[lev]*p.localSize[lev]; i++) {
                    phi_old[lev][i] = phi[lev][i];
                }
            }
        }
    }
    
    // Write result to file
    /*
    for (i = 0; i < p.N; i++) {
        for (j = 0; j < p.N; j++) {
            fprintf(output, "%f\t", phi[0][i*p.N+j]);
        }
        fprintf(output, "\n");
    }
    */
    MPI_Finalize();
    
    return 0;
}

void v_cycle(double **phi, double **phi_old, double **res, param p) {

    int lev;
    // Go down
    for (lev = 0; lev < NLEV; lev++) {    

        // Get the new phi (smooth) and use it to compute residue, then project to the coarser level
        relax(phi[lev], phi_old[lev], res[lev], lev, p);

        // Get the projected residue and use it to compute the error of the previous level, which is phi on this level (RECURSIVE). 
        proj_res(res[lev + 1], res[lev], phi[lev], phi_old[lev], lev,p);
    }

    // Go up
    for (lev = NLEV; lev >= 0; lev--) { 
        
        // Use the newly computed res to get a new phi. 
        relax(phi[lev], phi_old[lev], res[lev], lev, p);   // lev = NLEV -1, ... 0;

        // Interpolate to the finer level. the phi on the coarse level is the error on the fine level. 
        if (lev > 0) {
        inter_add(phi[lev-1], phi[lev], lev, p);   // phi[lev-1] += error = P phi[lev] and set phi[lev] = 0;
        }
    }
}

void inter_add(double *phi_f, double *phi_c, int lev, param p) {  
    int L, Lc, x, y;
    Lc = p.localSize[lev];  // coarse  level
    L = p.localSize[lev-1]; 

    //printf("From %d to %d for rank %d\n",lev,lev-1,my_rank);
    int left_limit = p.left_limit[lev];
    int right_limit = p.right_limit[lev];
    int up_limit = p.up_limit[lev];
    int down_limit = p.down_limit[lev];
  
    for (x = up_limit; x <= down_limit; x++)
        for (y = left_limit; y <= right_limit; y++) {
            // Add the error back to phi
            phi_f[(2*y-left_limit) + (2*x-up_limit)*L] += phi_c[y + x*Lc];
            phi_f[(2*y-left_limit+1) + (2*x-up_limit)*L] += phi_c[y + x*Lc];
            phi_f[(2*y-left_limit) + (2*x-up_limit+1)*L] += phi_c[y + x*Lc];
            phi_f[(2*y-left_limit+1) + (2*x-up_limit+1)*L] += phi_c[y + x*Lc];
        }

    //set to zero so phi = error 
    for (x = 0; x < Lc; x++)
        for (y = 0; y < Lc; y++){
            if(phi_c[y + x*Lc]>10) printf("for rank %d phi_c %d,%d is %f\n",my_rank,y,x,phi_c[y + x*Lc]);
                    phi_c[y + x*Lc] = 0.0;
        }
    //printf("Done %d to %d for rank %d\n",lev,lev-1,my_rank);
    return;
}

void proj_res(double *res_c, double *res_f, double *phi_f, double *phi_old_f, int lev, param p) {  
    int L, Lc, f_off, c_off, x, y;
    L = p.localSize[lev];
    double r[L*L];          // residue of Ae = r
    Lc = p.localSize[lev+1];     // coarse level

    //printf("From %d to %d for rank %d\n",lev,lev+1,my_rank);
  
    //get residue

    getR(res_f,phi_f,phi_old_f,r,lev,p);

    int left_limit = p.left_limit[lev+1];
    int right_limit = p.right_limit[lev+1];
    int up_limit = p.up_limit[lev+1];
    int down_limit = p.down_limit[lev+1];

    //project residue
    for(x = up_limit; x <= down_limit; x++)
        for(y = left_limit; y <= right_limit; y++)
            res_c[x*Lc + y] = 0.25*(r[(2*y-left_limit) + (2*x-up_limit)*L] + r[(2*y-left_limit+1) + (2*x-up_limit)*L]
                                  + r[(2*y-left_limit) + (2*x-up_limit+1)*L] + r[(2*y-left_limit+1) + (2*x-up_limit+1)*L]);

    //printf("Done %d to %d for rank %d\n",lev,lev+1,my_rank);
    return;
}

void getR(double *res, double *phi, double *phi_old, double *r, int lev, param p){

    // Prepare for async send/recv
    MPI_Request request[8];
    int requests;
    MPI_Status status[8];

    int i, x, y;
    int L;
    L  = p.localSize[lev];

    double global_resmag = 0.0;

    double* leftBuff;
    double* rightBuff;
    double* upBuff;
    double* downBuff;
    double* toLeft;
    double* toRight;

    toLeft = (double*)malloc(L * sizeof(double));
    toRight = (double*)malloc(L * sizeof(double));

    leftBuff = (double*)malloc(L * sizeof(double));
    rightBuff = (double*)malloc(L * sizeof(double));
    upBuff = (double*)malloc(L * sizeof(double));
    downBuff = (double*)malloc(L * sizeof(double));

    int left_limit = p.left_limit[lev];
    int right_limit = p.right_limit[lev];
    int up_limit = p.up_limit[lev];
    int down_limit = p.down_limit[lev];

    for(i=0;i<L;i++){
        toLeft[i]=phi[i*L];
        toRight[i]=phi[L - 1 + i*L];
    }

    requests=0;

    // Fill the down buffer. Send to the up, listen from the down.
    MPI_Isend(&phi[0], L, MPI_DOUBLE, (my_rank-2+world_size)%world_size, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(downBuff, L, MPI_DOUBLE, (my_rank+2)%world_size, 1, MPI_COMM_WORLD, request + requests++);


    // Fill the up buffer. Send to the down, listen from the up.
    MPI_Isend(&phi[L*(L-1)], L, MPI_DOUBLE, (my_rank+2)%world_size, 2, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(upBuff, L, MPI_DOUBLE, (my_rank-2+world_size)%world_size, 2, MPI_COMM_WORLD, request + requests++);

    // Fill the right buffer. Send to the left, listen from the right.
    MPI_Isend(toLeft,   L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 3, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(rightBuff, L, MPI_DOUBLE, (my_rank+1)%world_size, 3, MPI_COMM_WORLD, request + requests++);

    // Fill the left buffer. Send to the right, listen from the left.
    MPI_Isend(toRight,   L, MPI_DOUBLE, (my_rank+1)%world_size, 4, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(leftBuff, L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 4, MPI_COMM_WORLD, request + requests++);

    for (x = up_limit+1; x <= down_limit-1; x++)
        for (y = left_limit+1; y <= right_limit-1; y++){
            r[y+x*L] = res[y + x*L] - phi[y + x*L]  
                    + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                    + p.scale*phi_old[y + x*L];
        }

    if(up_limit-1==0)                                                     // Up is boundary
        for(y = left_limit+1; y <= right_limit-1; y++){
            r[y+up_limit*L] = res[y + up_limit*L] - phi[y + up_limit*L]  
                + TSTRIDE*p.scale*(phi[y+1 + up_limit*L] + phi[y-1 + up_limit*L] 
                + phi[y + (up_limit+1)*L] + phi[y + (up_limit-1)*L])
                + p.scale*phi_old[y + up_limit*L];
    }

    if(down_limit+1==L-1)
        for(y = left_limit+1; y <= right_limit-1; y++){
            r[y+down_limit*L] = res[y + down_limit*L] - phi[y + down_limit*L]  
                    + TSTRIDE*p.scale*(phi[y+1 + down_limit*L] + phi[y-1 + down_limit*L] 
                    + phi[y + (down_limit+1)*L] + phi[y + (down_limit-1)*L])
                    + p.scale*phi_old[y + down_limit*L];
    }

    if(left_limit-1==0)
        for(x = up_limit + 1;x <= down_limit - 1;x++){
            r[left_limit+x*L] = res[left_limit + x*L] - phi[left_limit + x*L]  
                    + TSTRIDE*p.scale*(phi[left_limit+1 + x*L] + phi[left_limit-1 + x*L] 
                    + phi[left_limit + (x+1)*L] + phi[left_limit + (x-1)*L])
                    + p.scale*phi_old[left_limit + x*L];
    }

    if(right_limit + 1 == L - 1)
        for(x = up_limit + 1;x <= down_limit - 1;x++){
            r[right_limit+x*L] = res[right_limit + x*L] - phi[right_limit + x*L]  
                    + TSTRIDE*p.scale*(phi[right_limit+1 + x*L] + phi[right_limit-1 + x*L] 
                    + phi[right_limit + (x+1)*L] + phi[right_limit + (x-1)*L])
                    + p.scale*phi_old[right_limit + x*L];
    }
    //printf("rank %d res %f\n",my_rank,ResRoot);

    MPI_Waitall (requests, request, status);

    if(up_limit-1==(-1))                                             // Up is Buffer
        for(y = left_limit+1; y <= right_limit-1; y++){
        r[y+up_limit*L] = res[y + up_limit*L] - phi[y + up_limit*L]  
                    + TSTRIDE*p.scale*(phi[y+1 + up_limit*L] + phi[y-1 + up_limit*L] 
                    + phi[y + (up_limit+1)*L] + upBuff[y])
                    + p.scale*phi_old[y + up_limit*L];
    }

    if(down_limit+1==L)
        for(y = left_limit+1; y <= right_limit-1; y++){
        r[y+down_limit*L] = res[y + down_limit*L] - phi[y + down_limit*L]  
                    + TSTRIDE*p.scale*(phi[y+1 + down_limit*L] + phi[y-1 + down_limit*L] 
                    + downBuff[y] + phi[y + (down_limit-1)*L])
                    + p.scale*phi_old[y + down_limit*L];
    }

    if (left_limit-1 == (-1))
        for(x = up_limit + 1;x <= down_limit - 1;x++){
        r[left_limit+x*L] = res[left_limit + x*L] - phi[left_limit + x*L]  
                + TSTRIDE*p.scale*(phi[left_limit+1 + x*L] + leftBuff[x] 
                + phi[left_limit + (x+1)*L] + phi[left_limit + (x-1)*L])
                + p.scale*phi_old[left_limit + x*L];
    }

    if(right_limit+1==L)
        for(x = up_limit + 1;x <= down_limit - 1;x++){
        r[right_limit+x*L] = res[right_limit + x*L] - phi[right_limit + x*L]  
                + TSTRIDE*p.scale*(rightBuff[x] + phi[right_limit-1 + x*L] 
                + phi[right_limit + (x+1)*L] + phi[right_limit + (x-1)*L])
                + p.scale*phi_old[right_limit + x*L];
    }                                                

    // U -> up, D -> down; L -> left, R -> Right, [A][B][C] {A -> y.pos, B -> x.pos, C -> point.relativePos}
      
    double ULU = (up_limit-1==0) ? phi[left_limit + (up_limit-1)*L] : upBuff[left_limit];
    double ULL = (left_limit-1==0) ? phi[left_limit-1 + up_limit*L] : leftBuff[up_limit];
    r[left_limit+up_limit*L] = res[left_limit + up_limit*L] - phi[left_limit + up_limit*L]  
                + TSTRIDE*p.scale*(rightBuff[up_limit] + ULL 
                + phi[left_limit + (up_limit+1)*L] + ULU)
                + p.scale*phi_old[left_limit + up_limit*L];
       
    double URU = (up_limit-1==0) ? phi[right_limit + (up_limit-1)*L] : upBuff[right_limit];
    double URR = (right_limit + 1 == L - 1) ? phi[right_limit+1 + up_limit*L] : rightBuff[up_limit];
    r[right_limit+up_limit*L] = res[right_limit + up_limit*L] - phi[right_limit + up_limit*L]  
                + TSTRIDE*p.scale*(URR + phi[right_limit-1 + up_limit*L] 
                + phi[right_limit + (up_limit+1)*L] + URU)
                + p.scale*phi_old[right_limit + up_limit*L];
       
    double DLD = (down_limit + 1 == L - 1) ? phi[left_limit + (down_limit+1)*L] : downBuff[left_limit];
    double DLL = (left_limit-1==0) ? phi[left_limit-1 + down_limit*L] : leftBuff[down_limit];
    r[left_limit+down_limit*L] = res[left_limit + down_limit*L] - phi[left_limit + down_limit*L]  
                + TSTRIDE*p.scale*(phi[left_limit+1 + down_limit*L] + DLL 
                + DLD + phi[left_limit + (down_limit-1)*L])
                + p.scale*phi_old[left_limit + down_limit*L];
      
    double DRD = (down_limit + 1 == L - 1) ? phi[right_limit + (down_limit+1)*L] : downBuff[right_limit];
    double DRR = (right_limit + 1 == L - 1) ? phi[right_limit+1 + down_limit*L] : rightBuff[down_limit];
    r[right_limit+down_limit*L] = res[right_limit + down_limit*L] - phi[right_limit + down_limit*L]  
                + TSTRIDE*p.scale*(DRR + phi[right_limit-1 + down_limit*L] 
                + DRD + phi[right_limit + (down_limit-1)*L])
                + p.scale*phi_old[right_limit + down_limit*L];

    //printf("Rank%d RES is %.15f\n",my_rank,ResRoot);

    free(leftBuff);
    free(rightBuff);
    free(upBuff);
    free(downBuff);
    free(toLeft);
    free(toRight);    

    return;
}

double GetResRoot(double *phi, double *phi_old, double *res, int lev, param p) {
    // Prepare for async send/recv
    MPI_Request request[8];
    int requests;
    MPI_Status status[8];

    int i, x, y;
    double residue;
    double ResRoot = 0.0;
    int L;
    L  = p.localSize[lev];

    double global_resmag = 0.0;

    double* leftBuff;
    double* rightBuff;
    double* upBuff;
    double* downBuff;
    double* toLeft;
    double* toRight;

    toLeft = (double*)malloc(L * sizeof(double));
    toRight = (double*)malloc(L * sizeof(double));

    leftBuff = (double*)malloc(L * sizeof(double));
    rightBuff = (double*)malloc(L * sizeof(double));
    upBuff = (double*)malloc(L * sizeof(double));
    downBuff = (double*)malloc(L * sizeof(double));

    int left_limit = p.left_limit[lev];
    int right_limit = p.right_limit[lev];
    int up_limit = p.up_limit[lev];
    int down_limit = p.down_limit[lev];

    for(i=0;i<L;i++){
        toLeft[i]=phi[i*L];
        toRight[i]=phi[L - 1 + i*L];
    }

    requests=0;

    // Fill the down buffer. Send to the up, listen from the down.
    MPI_Isend(&phi[0], L, MPI_DOUBLE, (my_rank-2+world_size)%world_size, 1, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(downBuff, L, MPI_DOUBLE, (my_rank+2)%world_size, 1, MPI_COMM_WORLD, request + requests++);


    // Fill the up buffer. Send to the down, listen from the up.
    MPI_Isend(&phi[L*(L-1)], L, MPI_DOUBLE, (my_rank+2)%world_size, 2, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(upBuff, L, MPI_DOUBLE, (my_rank-2+world_size)%world_size, 2, MPI_COMM_WORLD, request + requests++);

    // Fill the right buffer. Send to the left, listen from the right.
    MPI_Isend(toLeft,   L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 3, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(rightBuff, L, MPI_DOUBLE, (my_rank+1)%world_size, 3, MPI_COMM_WORLD, request + requests++);

    // Fill the left buffer. Send to the right, listen from the left.
    MPI_Isend(toRight,   L, MPI_DOUBLE, (my_rank+1)%world_size, 4, MPI_COMM_WORLD, request + requests++);
    MPI_Irecv(leftBuff, L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 4, MPI_COMM_WORLD, request + requests++);

    for (x = up_limit+1; x <= down_limit-1; x++)
        for (y = left_limit+1; y <= right_limit-1; y++){
            residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                    + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                    + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                    + phi_old[y + x*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }

    if(up_limit-1==0)                                                     // Up is boundary
        for(y = left_limit+1; y <= right_limit-1; y++){
        residue = res[y + up_limit*L]/p.scale/TSTRIDE - phi[y + up_limit*L]/p.scale/TSTRIDE  
                + (phi[y+1 + up_limit*L] + phi[y-1 + up_limit*L] 
                        + phi[y + (up_limit+1)*L] + phi[y + (up_limit-1)*L])
                + phi_old[y + up_limit*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }

    if(down_limit+1==L-1)
        for(y = left_limit+1; y <= right_limit-1; y++){
        residue = res[y + down_limit*L]/p.scale/TSTRIDE - phi[y + down_limit*L]/p.scale/TSTRIDE  
                + (phi[y+1 + down_limit*L] + phi[y-1 + down_limit*L] 
                        + phi[y + (down_limit+1)*L] + phi[y + (down_limit-1)*L])
                + phi_old[y + down_limit*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }

    if(left_limit-1==0)
        for(x = up_limit + 1;x <= down_limit - 1;x++){
        residue = res[left_limit + x*L]/p.scale/TSTRIDE - phi[left_limit + x*L]/p.scale/TSTRIDE  
                + (phi[left_limit+1 + x*L] + phi[left_limit-1 + x*L] 
                        + phi[left_limit + (x+1)*L] + phi[left_limit + (x-1)*L])
                + phi_old[left_limit + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }

    if(right_limit + 1 == L - 1)
        for(x = up_limit + 1;x <= down_limit - 1;x++){
        residue = res[right_limit + x*L]/p.scale/TSTRIDE - phi[right_limit + x*L]/p.scale/TSTRIDE  
                + (phi[right_limit+1 + x*L] + phi[right_limit-1 + x*L] 
                        + phi[right_limit + (x+1)*L] + phi[right_limit + (x-1)*L])
                + phi_old[right_limit + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }
    //printf("rank %d res %f\n",my_rank,ResRoot);

    MPI_Waitall (requests, request, status);

    if(up_limit-1==(-1))                                             // Up is Buffer
        for(y = left_limit+1; y <= right_limit-1; y++){
        residue = res[y + up_limit*L]/p.scale/TSTRIDE - phi[y + up_limit*L]/p.scale/TSTRIDE  
                + (phi[y+1 + up_limit*L] + phi[y-1 + up_limit*L] 
                        + phi[y + (up_limit+1)*L] + upBuff[y])
                + phi_old[y + up_limit*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }

    if(down_limit+1==L)
        for(y = left_limit+1; y <= right_limit-1; y++){
        residue = res[y + down_limit*L]/p.scale/TSTRIDE - phi[y + down_limit*L]/p.scale/TSTRIDE  
                + (phi[y+1 + down_limit*L] + phi[y-1 + down_limit*L] 
                        + downBuff[y] + phi[y + (down_limit-1)*L])
                + phi_old[y + down_limit*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }

    if (left_limit-1 == (-1))
        for(x = up_limit + 1;x <= down_limit - 1;x++){
        residue = res[left_limit + x*L]/p.scale/TSTRIDE - phi[left_limit + x*L]/p.scale/TSTRIDE  
                + (phi[left_limit+1 + x*L] + leftBuff[x] 
                        + phi[left_limit + (x+1)*L] + phi[left_limit + (x-1)*L])
                + phi_old[left_limit + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }

    if(right_limit+1==L)
        for(x = up_limit + 1;x <= down_limit - 1;x++){
        residue = res[right_limit + x*L]/p.scale/TSTRIDE - phi[right_limit + x*L]/p.scale/TSTRIDE  
                + (rightBuff[x] + phi[right_limit-1 + x*L] 
                        + phi[right_limit + (x+1)*L] + phi[right_limit + (x-1)*L])
                + phi_old[right_limit + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
    }                                                

    // U -> up, D -> down; L -> left, R -> Right, [A][B][C] {A -> y.pos, B -> x.pos, C -> point.relativePos}
    double ULU = (up_limit-1==0) ? phi[left_limit + (up_limit-1)*L] : upBuff[left_limit];
    double ULL = (left_limit-1==0) ? phi[left_limit-1 + up_limit*L] : leftBuff[up_limit];
    residue = res[left_limit + up_limit*L]/p.scale/TSTRIDE - phi[left_limit + up_limit*L]/p.scale/TSTRIDE  
            + (rightBuff[up_limit] + ULL 
                + phi[left_limit + (up_limit+1)*L] + ULU)
            + phi_old[left_limit + up_limit*L]/TSTRIDE;
    ResRoot += residue*residue; // true residue
       
    double URU = (up_limit-1==0) ? phi[right_limit + (up_limit-1)*L] : upBuff[right_limit];
    double URR = (right_limit + 1 == L - 1) ? phi[right_limit+1 + up_limit*L] : rightBuff[up_limit];
    residue = res[right_limit + up_limit*L]/p.scale/TSTRIDE - phi[right_limit + up_limit*L]/p.scale/TSTRIDE  
            + (URR + phi[right_limit-1 + up_limit*L] 
                + phi[right_limit + (up_limit+1)*L] + URU)
            + phi_old[right_limit + up_limit*L]/TSTRIDE;
    ResRoot += residue*residue; // true residue
     
    double DLD = (down_limit + 1 == L - 1) ? phi[left_limit + (down_limit+1)*L] : downBuff[left_limit];
    double DLL = (left_limit-1==0) ? phi[left_limit-1 + down_limit*L] : leftBuff[down_limit];
    residue = res[left_limit + down_limit*L]/p.scale/TSTRIDE - phi[left_limit + down_limit*L]/p.scale/TSTRIDE  
            + (phi[left_limit+1 + down_limit*L] + DLL 
                + DLD + phi[left_limit + (down_limit-1)*L])
            + phi_old[left_limit + down_limit*L]/TSTRIDE;
    ResRoot += residue*residue; // true residue
      
    double DRD = (down_limit + 1 == L - 1) ? phi[right_limit + (down_limit+1)*L] : downBuff[right_limit];
    double DRR = (right_limit + 1 == L - 1) ? phi[right_limit+1 + down_limit*L] : rightBuff[down_limit];
    residue = res[right_limit + down_limit*L]/p.scale/TSTRIDE - phi[right_limit + down_limit*L]/p.scale/TSTRIDE  
            + (DRR + phi[right_limit-1 + down_limit*L] 
                + DRD + phi[right_limit + (down_limit-1)*L])
            + phi_old[right_limit + down_limit*L]/TSTRIDE;
    ResRoot += residue*residue; // true residue

    //printf("Rank%d RES is %.15f\n",my_rank,ResRoot);

    free(leftBuff);
    free(rightBuff);
    free(upBuff);
    free(downBuff);
    free(toLeft);
    free(toRight);
    MPI_Allreduce(&ResRoot, &global_resmag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 

    return sqrt(global_resmag);
}

void relax(double *phi, double *phi_old, double *res, int lev, param p) {

    // Prepare for async send/recv
    MPI_Request request[8];
    int requests;
    MPI_Status status[8];

    int i, x, y;
    int L;
    double* tmp; // Only phi needs surrounding values to calculate.
    double* leftBuff;
    double* rightBuff;
    double* upBuff;
    double* downBuff;
    double* toLeft;
    double* toRight;
    L  = p.localSize[lev];

    int left_limit = p.left_limit[lev];
    int right_limit = p.right_limit[lev];
    int up_limit = p.up_limit[lev];
    int down_limit = p.down_limit[lev];

    tmp = (double*)malloc((L)*(L)*sizeof(double)); // For all cores one row and one column are boundaries.

    for(i=0;i<L*L;i++)
        tmp[i]=0;

    leftBuff = (double*)malloc(L * sizeof(double));
    rightBuff = (double*)malloc(L * sizeof(double));
    upBuff = (double*)malloc(L * sizeof(double));
    downBuff = (double*)malloc(L * sizeof(double));
    toLeft = (double*)malloc(L * sizeof(double));
    toRight = (double*)malloc(L * sizeof(double));

    for (i = 0; i < N_PER_LEV; i++) { 

        for(x=0;x<L;x++){
            toLeft[x]=phi[x*L];
            //if(toLeft[x]!=0) printf("toLeft %d from rank %d is %f\n",x,my_rank,toLeft[x]);
            toRight[x]=phi[L - 1 + x*L];
            //if(toRight[x]!=0) printf("toRight %d from rank %d is %f\n",x,my_rank,toRight[x]);
        }

        requests=0;

        // Fill the down buffer. Send to the up, listen from the down.
        MPI_Isend(&phi[0], L, MPI_DOUBLE, (my_rank-2+world_size)%world_size, 1, MPI_COMM_WORLD, request + requests++);
        MPI_Irecv(downBuff, L, MPI_DOUBLE, (my_rank+2)%world_size, 1, MPI_COMM_WORLD, request + requests++);


        // Fill the up buffer. Send to the down, listen from the up.
        MPI_Isend(&phi[L*(L-1)], L, MPI_DOUBLE, (my_rank+2)%world_size, 2, MPI_COMM_WORLD, request + requests++);
        MPI_Irecv(upBuff, L, MPI_DOUBLE, (my_rank-2+world_size)%world_size, 2, MPI_COMM_WORLD, request + requests++);

        // Fill the right buffer. Send to the left, listen from the right.
        MPI_Isend(toLeft,   L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 3, MPI_COMM_WORLD, request + requests++);
        MPI_Irecv(rightBuff, L, MPI_DOUBLE, (my_rank+1)%world_size, 3, MPI_COMM_WORLD, request + requests++);

        // Fill the left buffer. Send to the right, listen from the left.
        MPI_Isend(toRight,   L, MPI_DOUBLE, (my_rank+1)%world_size, 4, MPI_COMM_WORLD, request + requests++);
        MPI_Irecv(leftBuff, L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 4, MPI_COMM_WORLD, request + requests++);

        for (x = up_limit+1; x <= down_limit-1; x++)
            for (y = left_limit+1; y <= right_limit-1; y++)                                                   // Loop over the rest.
                tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];

        if(up_limit-1==0)                                 // Up is boundary
            for(y = left_limit+1; y <= right_limit-1; y++)
                tmp[y + up_limit * L] = res[y + up_limit * L]                           
                            + TSTRIDE*p.scale*(phi[y+1 + up_limit*L] + phi[y-1 + up_limit*L] 
                            + phi[y + (up_limit+1)*L] + phi[y + (up_limit-1)*L])
                            + p.scale*phi_old[y + up_limit*L];

        if(down_limit+1==L-1)
            for(y = left_limit+1; y <= right_limit-1; y++)
                tmp[y + down_limit * L] = res[y + down_limit * L]                           
                            + TSTRIDE*p.scale*(phi[y+1 + down_limit*L] + phi[y-1 + down_limit*L] 
                            + phi[y + (down_limit+1)*L] + phi[y + (down_limit-1)*L])
                            + p.scale*phi_old[y + down_limit*L];

        if(left_limit-1==0)
            for(x = up_limit + 1;x <= down_limit - 1;x++)
                tmp[left_limit + x * L] = res[left_limit + x * L]                           
                            + TSTRIDE*p.scale*(phi[left_limit+1 + x*L] + phi[left_limit-1 + x*L] 
                            + phi[left_limit + (x+1)*L] + phi[left_limit + (x-1)*L])
                            + p.scale*phi_old[left_limit + x*L];

        if(right_limit + 1 == L - 1)
            for(x = up_limit + 1;x <= down_limit - 1;x++)
                tmp[right_limit + x * L] = res[right_limit + x * L]                           
                            + TSTRIDE*p.scale*(phi[right_limit+1 + x*L] + phi[right_limit-1 + x*L] 
                            + phi[right_limit + (x+1)*L] + phi[right_limit + (x-1)*L])
                            + p.scale*phi_old[right_limit + x*L];
        MPI_Waitall (requests, request, status);

        if(up_limit-1==(-1))                                             // Up is Buffer
            for(y = left_limit+1; y <= right_limit-1; y++)
                tmp[y + up_limit * L] = res[y + up_limit * L]            
                            + TSTRIDE*p.scale*(phi[y+1 + up_limit*L] + phi[y-1 + up_limit*L] 
                            + phi[y + (up_limit+1)*L] + upBuff[y])
                            + p.scale*phi_old[y + up_limit*L];

        if(down_limit+1==L)
            for(y = left_limit+1; y <= right_limit-1; y++)
                tmp[y + down_limit * L] = res[y + down_limit * L]            
                            + TSTRIDE*p.scale*(phi[y+1 + down_limit*L] + phi[y-1 + down_limit*L] 
                            + downBuff[y] + phi[y + (down_limit-1)*L])
                            + p.scale*phi_old[y + down_limit*L];

        if (left_limit-1 == (-1))
            for(x = up_limit + 1;x <= down_limit - 1;x++)
                tmp[left_limit + x * L] = res[left_limit + x * L]                           
                            + TSTRIDE*p.scale*(phi[left_limit+1 + x*L] + leftBuff[x] 
                            + phi[left_limit + (x+1)*L] + phi[left_limit + (x-1)*L])
                            + p.scale*phi_old[left_limit + x*L];

        if(right_limit+1==L)
            for(x = up_limit + 1;x <= down_limit - 1;x++)
                tmp[right_limit + x * L] = res[right_limit + x * L]                           
                            + TSTRIDE*p.scale*(rightBuff[x] + phi[right_limit-1 + x*L] 
                            + phi[right_limit + (x+1)*L] + phi[right_limit + (x-1)*L])
                            + p.scale*phi_old[right_limit + x*L];                                                    

        // U -> up, D -> down; L -> left, R -> Right, [A][B][C] {A -> y.pos, B -> x.pos, C -> point.relativePos}

        double ULU = (up_limit-1==0) ? phi[left_limit + (up_limit-1)*L] : upBuff[left_limit];
        double ULL = (left_limit-1==0) ? phi[left_limit-1 + up_limit*L] : leftBuff[up_limit];
        tmp[left_limit + up_limit * L] = res[left_limit + up_limit * L]                           
                    + TSTRIDE*p.scale*(rightBuff[up_limit] + ULL 
                    + phi[left_limit + (up_limit+1)*L] + ULU)
                    + p.scale*phi_old[left_limit + up_limit*L];
      
        double URU = (up_limit-1==0) ? phi[right_limit + (up_limit-1)*L] : upBuff[right_limit];
        double URR = (right_limit + 1 == L - 1) ? phi[right_limit+1 + up_limit*L] : rightBuff[up_limit];
        tmp[right_limit + up_limit * L] = res[right_limit + up_limit * L]                           
                    + TSTRIDE*p.scale*(URR + phi[right_limit-1 + up_limit*L] 
                    + phi[right_limit + (up_limit+1)*L] + URU)
                    + p.scale*phi_old[right_limit + up_limit*L];  
    
        double DLD = (down_limit + 1 == L - 1) ? phi[left_limit + (down_limit+1)*L] : downBuff[left_limit];
        double DLL = (left_limit-1==0) ? phi[left_limit-1 + down_limit*L] : leftBuff[down_limit];
        tmp[left_limit + down_limit * L] = res[left_limit + down_limit * L]                           
                    + TSTRIDE*p.scale*(phi[left_limit+1 + down_limit*L] + DLL 
                    + DLD + phi[left_limit + (down_limit-1)*L])
                    + p.scale*phi_old[left_limit + down_limit*L];    
   
        double DRD = (down_limit + 1 == L - 1) ? phi[right_limit + (down_limit+1)*L] : downBuff[right_limit];
        double DRR = (right_limit + 1 == L - 1) ? phi[right_limit+1 + down_limit*L] : rightBuff[down_limit];
        tmp[right_limit + down_limit * L] = res[right_limit + down_limit * L]                           
                    + TSTRIDE*p.scale*(DRR + phi[right_limit-1 + down_limit*L] 
                    + DRD + phi[right_limit + (down_limit-1)*L])
                    + p.scale*phi_old[right_limit + down_limit*L];   

        // a coarse phi is the error of a fine phi
        for (x = 0; x <L; x++)
            for (y = 0; y < L; y++) 
                {
                    phi[y + x*L] = tmp[y + x*L];
                    //if(phi[y + x*L]>0) printf("Rank %d, phi %d,%d is %f\n",my_rank,y,x,phi[y + x*L]);
                }
    }    

    free(tmp);
    free(leftBuff);
    free(rightBuff);
    free(upBuff);
    free(downBuff);
    free(toLeft);
    free(toRight);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}
