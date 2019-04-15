#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
#include <opencv2/opencv.hpp>                       // opencv

using namespace cv;                                 // opencv

#define RESGOAL 1E-6
#define NLEV 0                                     // If 0, only one level
#define PERIOD 500
#define PI 3.141592653589793
#define TSTRIDE 10
#define N_PER_LEV 10                                // Iterate 10 times for each level

typedef struct{
    int N;
    int Lmax;
    int size[20];
    int localSize[20];
    // double a[20];
    double m_square;
    double scale;
} param;

// Useful globals
int world_size; // number of processes
int my_rank; // my process number

void relax(double *phi, double *phi_old, double *res, int lev, param p);
void proj_res(double *res_c, double *rec_f, double *phi_f, double *phi_old_f, int lev, param p);
void inter_add(double *phi_f, double *phi_c, int lev,param p);
double GetResRoot(double *phi, double *phi_old, double *res, int lev, param p);
void v_cycle(double **phi, double **phi_old, double **res, param p);

int main(int argc, char** argv) {     

    // Initialize MPI
    MPI_Init(&argc, &argv);
   
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   
    // Get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   
    Mat image;
    double *phi[20], *res[20], *phi_old[20];
    param p;
    int i, j, lev;
  

    // Initialize parameters
    p.Lmax = 7;
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
    // p.a[0] = 1.0;                                    // Stride, may not be needed
    p.scale = 1.0/(4.0*TSTRIDE + 1);

    for (lev = 1; lev < p.Lmax+1; lev++) {
        p.size[lev] = p.size[lev-1]/2+1;
        p.localSize[lev]=p.size[lev]/2;
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
        //if(my_rank==0)
            //printf("At the %d cycle the mag residue is %g\n",ncycle,resmag);


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
                res[0][0] = 3.0*TSTRIDE*p.scale*(1+sin(2.0*PI*t/(PERIOD/10)));
            ncycle = 0;
            for (lev = 0; lev < p.Lmax+1; lev++) {
                for (i = 0; i < p.localSize[lev]*p.localSize[lev]; i++) {
                    phi_old[lev][i] = phi[lev][i];
                }
            }

            
            if(my_rank==0){
                image = Mat(p.localSize[0], p.localSize[0], CV_64F, phi[0]);                  // opencv
                imshow("core0", image);                                              // opencv
                waitKey(25);                                                        // opencv 25ms per frame
            }
            if(my_rank==1){
                image = Mat(p.localSize[0], p.localSize[0], CV_64F, phi[0]);                  // opencv
                imshow("core1", image);                                              // opencv
                waitKey(25);
                }                                                        // opencv 25ms per frame
           if(my_rank==2){
                image = Mat(p.localSize[0], p.localSize[0], CV_64F, phi[0]);                  // opencv
                imshow("core2", image);                                              // opencv
                waitKey(25); }
           if(my_rank==3){
                image = Mat(p.localSize[0], p.localSize[0], CV_64F, phi[0]);                  // opencv
                imshow("core3", image);                                              // opencv
                waitKey(25);} 
            
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
    Lc = p.size[lev];  // coarse  level
    L = p.size[lev-1]; 
  
    for (x = 1; x < Lc-1; x++)
        for (y = 1; y < Lc-1; y++) {

            // Add the error back to phi
            phi_f[(2*y-1) + (2*x-1)*L] += phi_c[y + x*Lc];
            phi_f[(2*y) + (2*x-1)*L] += phi_c[y + x*Lc];
            phi_f[(2*y-1) + (2*x)*L] += phi_c[y + x*Lc];
            phi_f[(2*y) + (2*x)*L] += phi_c[y + x*Lc];
        }
    //set to zero so phi = error 
    for (x = 1; x < Lc-1; x++)
        for (y = 1; y < Lc-1; y++)
            phi_c[y + x*Lc] = 0.0;
    return;
}

void proj_res(double *res_c, double *res_f, double *phi_f, double *phi_old_f, int lev, param p) {  
    int L, Lc, f_off, c_off, x, y;
    L = p.size[lev];
    double r[L*L];          // residue of Ae = r
    Lc = p.size[lev+1];     // coarse level
  
    //get residue
    for(x = 1; x < L-1; x++)
        for(y = 1; y < L-1; y++)
            r[x*L + y] = res_f[x*L + y] - phi_f[x*L + y] + p.scale*phi_old_f[x*L + y] 
                + TSTRIDE*p.scale*(phi_f[y+1 + x*L] + phi_f[y-1 + x*L] + phi_f[y + (x+1)*L] + phi_f[y + (x-1)*L]);

    //project residue
    for(x = 1; x < Lc-1; x++)
        for(y = 1; y < Lc-1; y++)
            res_c[x*Lc + y] = 0.25*(r[(2*x-1)*L + (2*y-1)] + r[(2*y) + (2*x-1)*L] + r[(2*y-1) + (2*x)*L] + r[(2*y) + (2*x)*L]);
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

    if(my_rank == 0)
        for (x = 2; x < L-1; x++)
            for (y = 2; y < L-1; y++){                                                   // Loop over the rest.
                        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                                + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + phi_old[y + x*L]/TSTRIDE;
                        ResRoot += residue*residue; // true residue
                    }
    else if(my_rank == 1)
        for (x = 2; x < L-1; x++)
            for (y = 1; y < L-2; y++){                                                   // Loop over the rest.
                        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                                + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + phi_old[y + x*L]/TSTRIDE;
                        ResRoot += residue*residue; // true residue
                    }
    else if(my_rank == 2)
        for (x = 1; x < L-2; x++)
            for (y = 2; y < L-1; y++){                                                   // Loop over the rest.
                        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                                + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + phi_old[y + x*L]/TSTRIDE;
                        ResRoot += residue*residue; // true residue
                    }
    else if(my_rank == 3)
        for (x = 1; x < L-2; x++)
            for (y = 1; y < L-2; y++){                                                   // Loop over the rest.
                        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                                + (phi[(y+1) + x*L] + phi[(y-1) + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + phi_old[y + x*L]/TSTRIDE;
                        ResRoot += residue*residue; // true residue
                    }

    //printf("rank %d res %f\n",my_rank,ResRoot);

    MPI_Waitall (requests, request, status);

    if (my_rank == 0){
        x = 1;
        y = 1;                                                               // Common
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                + (phi[(y+1) + x*L] + phi[(y-1) + x*L] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=1;
        y=L-1;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                + (rightBuff[x] + phi[(y-1) + x*L] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=L-1;
        y=L-1;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                + (rightBuff[x] + phi[(y-1) + x*L] 
                + downBuff[y] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=L-1;
        y=1;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                + (phi[(y+1) + x*L] + phi[(y-1) + x*L] 
                + downBuff[y] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue

        int x1 = 1;
        int x2 = L-1;
        for(y = 2; y < L-1; y++){
            residue = res[y + x1*L]/p.scale/TSTRIDE - phi[y + x1*L]/p.scale/TSTRIDE   //Common
                + (phi[(y+1) + x1*L] + phi[(y-1) + x1*L] 
                + phi[y + (x1+1)*L] + phi[y + (x1-1)*L])
                + phi_old[y + x1*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
            residue = res[y + x2*L]/p.scale/TSTRIDE - phi[y + x2*L]/p.scale/TSTRIDE  
                + (phi[(y+1) + x2*L] + phi[(y-1) + x2*L] 
                + downBuff[y] + phi[y + (x2-1)*L])
                + phi_old[y + x2*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }

        int y1 = 1;
        int y2 = L-1;
        for(x = 2; x < L-1; x++){
            residue = res[y1 + x*L]/p.scale/TSTRIDE - phi[y1 + x*L]/p.scale/TSTRIDE    //Common
                + (phi[(y1+1) + x*L] + phi[(y1-1) + x*L] 
                + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                + phi_old[y1 + x*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue

            residue = res[y2 + x*L]/p.scale/TSTRIDE - phi[y2 + x*L]/p.scale/TSTRIDE  
                + (rightBuff[x] + phi[(y2-1) + x*L] 
                + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                + phi_old[y2 + x*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
    }
    else if (my_rank == 1){
        y = 0;
        x = 1;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                + (phi[(y+1) + x*L] + leftBuff[x] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=L-1;
        y=0;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                + (phi[(y+1) + x*L] + leftBuff[x] 
                + downBuff[y] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=L-1;
        y=L-2;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                + (phi[(y+1) + x*L] + phi[(y-1) + x*L] 
                + downBuff[y] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=1;
        y=L-2;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  //Common
                + (phi[(y+1) + x*L] + phi[(y-1) + x*L] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        int y1=0;
        int y2 = L-2;
        for(x = 2; x < L-1; x++){ 
            residue = res[y1 + x*L]/p.scale/TSTRIDE - phi[y1 + x*L]/p.scale/TSTRIDE 
                + (phi[y1+1 + x*L] + leftBuff[x]  
                + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                + phi_old[y1 + x*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue

            residue = res[y2 + x*L]/p.scale/TSTRIDE - phi[y2 + x*L]/p.scale/TSTRIDE  //Common
                + (phi[y2+1 + x*L] + phi[y2-1 + x*L] 
                + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                + phi_old[y2 + x*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
        int x1 = 1;
        int x2 = L-1;
        for(y = 1; y < L-2; y++){
            residue = res[y + x1*L]/p.scale/TSTRIDE - phi[y + x1*L]/p.scale/TSTRIDE  //Common
                + (phi[y+1 + x1*L] + phi[y-1 + x1*L] 
                + phi[y + (x1+1)*L] + phi[y + (x1-1)*L])
                + phi_old[y + x1*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue

            residue = res[y + x2*L]/p.scale/TSTRIDE - phi[y + x2*L]/p.scale/TSTRIDE 
                + (phi[y+1 + x2*L] + phi[y-1 + x2*L] 
                + downBuff[y] + phi[y + (x2-1)*L])
                + phi_old[y + x2*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue

        }
    }

    else if (my_rank == 2){
        x = 0;
        y = 1;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE
                + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                + phi[y + (x+1)*L] + upBuff[y])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=L-2;
        y=1;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  //Common
                + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        y=L-1;
        x=L-2;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE
                + (rightBuff[x] + phi[y-1 + x*L] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        y=L-1;
        x=0;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE
                + (rightBuff[x] + phi[y-1 + x*L] 
                + phi[y + (x+1)*L] + upBuff[y])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        int x1=0;
        int x2 = L-2;
        for(y = 2; y < L-1; y++){
            residue = res[y + x1*L]/p.scale/TSTRIDE - phi[y + x1*L]/p.scale/TSTRIDE
                + (phi[y+1 + x1*L] + phi[y-1 + x1*L] 
                + phi[y + (x1+1)*L] + upBuff[y])
                + phi_old[y + x1*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
            residue = res[y + x2*L]/p.scale/TSTRIDE - phi[y + x2*L]/p.scale/TSTRIDE  //Common
                + (phi[y+1 + x2*L] + phi[y-1 + x2*L] 
                + phi[y + (x2+1)*L] + phi[y + (x2-1)*L])
                + phi_old[y + x2*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
        int y1 = 1;
        int y2 = L-1;
        for(x = 1; x < L-2; x++){
            residue = res[y1 + x*L]/p.scale/TSTRIDE - phi[y1 + x*L]/p.scale/TSTRIDE
                + (phi[y1+1 + x*L] + phi[y1-1 + x*L] 
                + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                + phi_old[y1 + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
            residue = res[y2 + x*L]/p.scale/TSTRIDE - phi[y2 + x*L]/p.scale/TSTRIDE  //Common
                + (rightBuff[x] + phi[y2-1 + x*L] 
                + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                + phi_old[y2 + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue

        }

    }
    else if (my_rank == 3){
        x = 0;
        y = 0;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE 
                + (phi[y+1 + x*L] + leftBuff[x] 
                + phi[y + (x+1)*L] + upBuff[y])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=0;
        y=L-2;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE
                + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                + phi[y + (x+1)*L] + upBuff[y])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=L-2;
        y=L-2;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  //Common
                + (phi[y+1 + x*L] + phi[y-1 + x*L] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        x=L-2;
        y=0;
        residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE 
                + (phi[y+1 + x*L] + leftBuff[x] 
                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                + phi_old[y + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue

        int x1=0;
        int x2 = L-2;
        for(y = 1; y < L-2; y++){
            residue = res[y + x1*L]/p.scale/TSTRIDE - phi[y + x1*L]/p.scale/TSTRIDE  //Common
                + (phi[y+1 + x1*L] + phi[y-1 + x1*L] 
                + phi[y + (x1+1)*L] + upBuff[y])
                + phi_old[y + x1*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
            residue = res[y + x2*L]/p.scale/TSTRIDE - phi[y + x2*L]/p.scale/TSTRIDE  //Common
                + (phi[y+1 + x2*L] + phi[y-1 + x2*L] 
                + phi[y + (x2+1)*L] + phi[y + (x2-1)*L])
                + phi_old[y + x2*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        }

        int y1 = 0;
        int y2 = L-2;
        for(x = 1; x < L-2; x++){
            residue = res[y1 + x*L]/p.scale/TSTRIDE - phi[y1 + x*L]/p.scale/TSTRIDE  //Common
                + (phi[y1+1 + x*L] + leftBuff[x] 
                + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                + phi_old[y1 + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
            residue = res[y2 + x*L]/p.scale/TSTRIDE - phi[y2 + x*L]/p.scale/TSTRIDE  //Common
                + (phi[y2+1 + x*L] + phi[y2-1 + x*L] 
                + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                + phi_old[y2 + x*L]/TSTRIDE;
        ResRoot += residue*residue; // true residue
        }
    }

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

    const int left_limit = (my_rank == 0||my_rank==2) ? 1 : 0;
    const int right_limit = (my_rank == 1||my_rank==3) ? L-1 : L;
    const int up_limit = (my_rank==0||my_rank==1) ? 1 : 0;
    const int down_limit = (my_rank==2||my_rank==3) ? L-1 : L;

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

        if(my_rank == 0)
            for (x = 2; x < L-1; x++)
                for (y = 2; y < L-1; y++)                                                   // Loop over the rest.
                    tmp[y + x * L] = res[y + x * L]
                                + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + p.scale*phi_old[y + x*L];
        else if(my_rank == 1)
            for (x = 2; x < L-1; x++)
                for (y = 1; y < L-2; y++)                                                   // Loop over the rest.
                    tmp[y + x * L] = res[y + x * L]
                                + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + p.scale*phi_old[y + x*L];
        else if(my_rank == 2)
            for (x = 1; x < L-2; x++)
                for (y = 2; y < L-1; y++)                                                   // Loop over the rest.
                    tmp[y + x * L] = res[y + x * L]
                                + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + p.scale*phi_old[y + x*L];
        else if(my_rank == 3)
            for (x = 1; x < L-2; x++)
                for (y = 1; y < L-2; y++)                                                   // Loop over the rest.
                    tmp[y + x * L] = res[y + x * L]
                                + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                                + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                                + p.scale*phi_old[y + x*L];
    
         MPI_Waitall (requests, request, status);

        if (my_rank == 0){
            x = 1;
            y = 1;
            tmp[y + x * L] = res[y + x * L]                                                  // COMMON
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            x=1;
            y=L-1;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(rightBuff[x] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            x=L-1;
            y=L-1;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(rightBuff[x] + phi[y-1 + x*L] 
                            + downBuff[y] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            x=L-1;
            y=1;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + downBuff[y] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];

            int x1 = 1;
            int x2 = L-1;
            for(y = 2; y < L-1; y++){
                tmp[y + x1 * L] = res[y + x1 * L]                       // Common
                            + TSTRIDE*p.scale*(phi[y+1 + x1*L] + phi[y-1 + x1*L] 
                            + phi[y + (x1+1)*L] + phi[y + (x1-1)*L])
                            + p.scale*phi_old[y + x1*L];
                tmp[y + x2 * L] = res[y + x2 * L]                               
                            + TSTRIDE*p.scale*(phi[y+1 + x2*L] + phi[y-1 + x2*L] 
                            + downBuff[y] + phi[y + (x2-1)*L])
                            + p.scale*phi_old[y + x2*L];
            }

            int y1 = 1;
            int y2 = L-1;
            for(x = 2; x < L-1; x++){
                tmp[y1 + x * L] = res[y1 + x * L]                               // Common
                            + TSTRIDE*p.scale*(phi[y1+1 + x*L] + phi[y1-1 + x*L] 
                            + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                            + p.scale*phi_old[y1 + x*L];
                tmp[y2 + x * L] = res[y2 + x * L]
                            + TSTRIDE*p.scale*(rightBuff[x] + phi[y2-1 + x*L] 
                            + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                            + p.scale*phi_old[y2 + x*L];
            }
        }
        else if (my_rank == 1){
            y = 0;
            x = 1;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + leftBuff[x] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            x=L-1;
            y=0;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + leftBuff[x] 
                            + downBuff[y] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            x=L-1;
            y=L-2;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + downBuff[y] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            x=1;
            y=L-2;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            int y1=0;
            int y2 = L-2;
            for(x = 2; x < L-1; x++){ 
                tmp[y1 + x * L] = res[y1 + x * L]
                            + TSTRIDE*p.scale*(phi[y1+1 + x*L] + leftBuff[x]  
                            + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                            + p.scale*phi_old[y1 + x*L];
                tmp[y2 + x * L] = res[y2 + x * L]                             // Common
                            + TSTRIDE*p.scale*(phi[y2+1 + x*L] + phi[y2-1 + x*L] 
                            + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                            + p.scale*phi_old[y2 + x*L];
            }
            int x1 = 1;
            int x2 = L-1;
            for(y = 1; y < L-2; y++){
                tmp[y + x1 * L] = res[y + x1 * L]                            // Common
                            + TSTRIDE*p.scale*(phi[y+1 + x1*L] + phi[y-1 + x1*L] 
                            + phi[y + (x1+1)*L] + phi[y + (x1-1)*L])
                            + p.scale*phi_old[y + x1*L];
                tmp[y + x2 * L] = res[y + x2 * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x2*L] + phi[y-1 + x2*L] 
                            + downBuff[y] + phi[y + (x2-1)*L])
                            + p.scale*phi_old[y + x2*L];

            }

        }
        else if (my_rank == 2){
            x = 0;
            y = 1;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + upBuff[y])
                            + p.scale*phi_old[y + x*L];
            x=L-2;
            y=1;
            tmp[y + x * L] = res[y + x * L]                                   // Common
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            y=L-1;
            x=L-2;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(rightBuff[x] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            y=L-1;
            x=0;
            tmp[y + x * L] = res[y + x * L]
                            + TSTRIDE*p.scale*(rightBuff[x] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + upBuff[y])
                            + p.scale*phi_old[y + x*L];
            int x1=0;
            int x2 = L-2;
            for(y = 2; y < L-1; y++){
                tmp[y + x1 * L] = res[y + x1 * L]
                            + TSTRIDE*p.scale*(phi[y+1 + x1*L] + phi[y-1 + x1*L] 
                            + phi[y + (x1+1)*L] + upBuff[y])
                            + p.scale*phi_old[y + x1*L];
                tmp[y + x2 * L] = res[y + x2 * L]                              //Common
                            + TSTRIDE*p.scale*(phi[y+1 + x2*L] + phi[y-1 + x2*L] 
                            + phi[y + (x2+1)*L] + phi[y + (x2-1)*L])
                            + p.scale*phi_old[y + x2*L];
            }
            int y1 = 1;
            int y2 = L-1;
            for(x = 1; x < L-2; x++){
                tmp[y1 + x * L] = res[y1 + x * L]                                   // Common
                            + TSTRIDE*p.scale*(phi[y1+1 + x*L] + phi[y1-1 + x*L] 
                            + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                            + p.scale*phi_old[y1 + x*L];
                tmp[y2 + x * L] = res[y2 + x * L]
                            + TSTRIDE*p.scale*(rightBuff[x] + phi[y2-1 + x*L] 
                            + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                            + p.scale*phi_old[y2 + x*L];

            }

        }
        else if (my_rank == 3){
            x = 0;
            y = 0;
            tmp[y + x * L] = res[y + x * L]                                  
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + leftBuff[x] 
                            + phi[y + (x+1)*L] + upBuff[y])
                            + p.scale*phi_old[y + x*L];
            x=0;
            y=L-2;
            tmp[y + x * L] = res[y + x * L]                                 
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + upBuff[y])
                            + p.scale*phi_old[y + x*L];
            x=L-2;
            y=L-2;
            tmp[y + x * L] = res[y + x * L]                                   // Common
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];
            x=L-2;
            y=0;
            tmp[y + x * L] = res[y + x * L]                              
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + leftBuff[x] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L];

            int x1=0;
            int x2 = L-2;
            for(y = 1; y < L-2; y++){
                tmp[y + x1 * L] = res[y + x1 * L]                                
                            + TSTRIDE*p.scale*(phi[y+1 + x1*L] + phi[y-1 + x1*L] 
                            + phi[y + (x1+1)*L] + upBuff[y])
                            + p.scale*phi_old[y + x1*L];
                tmp[y + x2 * L] = res[y + x2 * L]                                   // Common
                            + TSTRIDE*p.scale*(phi[y+1 + x2*L] + phi[y-1 + x2*L] 
                            + phi[y + (x2+1)*L] + phi[y + (x2-1)*L])
                            + p.scale*phi_old[y + x2*L];
            }

            int y1 = 0;
            int y2 = L-2;
            for(x = 1; x < L-2; x++){
                tmp[y1 + x * L] = res[y1 + x * L]
                            + TSTRIDE*p.scale*(phi[y1+1 + x*L] + leftBuff[x] 
                            + phi[y1 + (x+1)*L] + phi[y1 + (x-1)*L])
                            + p.scale*phi_old[y1 + x*L];
                tmp[y2 + x * L] = res[y2 + x * L]                   // Common
                            + TSTRIDE*p.scale*(phi[y2+1 + x*L] + phi[y2-1 + x*L] 
                            + phi[y2 + (x+1)*L] + phi[y2 + (x-1)*L])
                            + p.scale*phi_old[y2 + x*L];
            }
        }


    // a coarse phi is the error of a fine phi
    for (x = 0; x < L; x++)
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
