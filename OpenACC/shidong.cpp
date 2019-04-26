#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
//#include <complex.h>
//#include <opencv2/opencv.hpp>                       // opencv

using namespace std;                                 // opencv

#define RESGOAL 1E-6
#define NLEV 6                                      // If 0, only one level
#define PERIOD 100
#define PI 3.141592653589793
#define TSTRIDE 10
#define N_PER_LEV 10                                // Iterate 10 times for each level

typedef struct{
    int N;
    int Lmax;
    int size[20];
    // double a[20];
    double m_square;
    double scale;
    int index[20];
} param;

void relax(double *phi, double *phi_old, double *res, int lev, param p);
void proj_res(double *res_c, double *rec_f, double *phi_f, double *phi_old_f, int lev, param p);
void inter_add(double *phi_f, double *phi_c, int lev,param p);
double GetResRoot(double *phi, double *phi_old, double *res, int lev, param p);
void v_cycle(double *phi, double *phi_old, double *res, param p);

int main() {  
    //Mat image;
    double *phi, *res, *phi_old;
    param p;
    int i, j, lev;

    // Initialize parameters
    p.Lmax = 7;
    p.N = 2*(int)pow(2,p.Lmax)+2;
    p.m_square = 0.0;                                    // Scaling parameter, a small number
    p.index[0] = 0;

    /*Size frame_size(p.N, p.N);
    int frames_per_second = 30;
    VideoWriter oVideoWriter("./MyVideo.mp4", CV_FOURCC('G','R','E','Y'), frames_per_second, frame_size, false);
    if (oVideoWriter.isOpened() == false) 
    {
        std::cout << "Cannot save the video to a file" << std::endl;
        std::cin.get(); //wait for any key press
        return -1;
    }*/

    // Exception control
    if (NLEV > p.Lmax) { 
        printf("ERROR More levels than available in lattice! \n");
        return 0;
    }
  
    printf("\n V cycle for %d by %d lattice with NLEV = %d out of max %d \n", p.N, p.N, NLEV, p.Lmax); 
  
    // Initialize arrays
    p.size[0] = p.N;
    // p.a[0] = 1.0;                                    // Stride, may not be needed
    p.scale = 1.0/(4.0*TSTRIDE + 1);

    for (lev = 1; lev < p.Lmax+1; lev++) {
        p.size[lev] = p.size[lev-1]/2+1;
        //p.a[lev] = 2.0 * p.a[lev-1];                  // Not needed
        //p.scale[lev] = 1.0/(4.0 + p.m*p.m*p.a[lev]*p.a[lev]);           // Seems p.a is not needed after all
    }

    for (i=1; i<=p.Lmax; i++){
        p.index[i] = p.index[i-1] + p.size[i-1] * p.size[i-1];
    }
    int sum = p.index[i-1] + p.size[i-1]*p.size[i-1];
    //std::cout<<i<<" "<<sum<<std::endl;                                     
    //return 0;

    // Initialize phi and res for each level
    phi = (double *) malloc(sum * sizeof(double));
    phi_old = (double *) malloc(sum * sizeof(double));
    res = (double *) malloc(sum * sizeof(double));
    for (i = 0; i < sum; i++) {
        phi[i] = 0.0;
        phi_old[i] = 0.0;
        res[i] = 0.0;
    }  
  
    res[p.N/2 + (p.N/2)*p.N] = 1.0*TSTRIDE*p.scale;   // Unit point source in middle of N by N lattice 

#pragma acc init
#pragma acc data copy(phi[0:sum]) copy(res[0:sum]) copy(phi_old[0:sum])
    // iterate to solve
    double resmag = 1.0;                            // Not rescaled.
    int ncycle = 1; 
    int t = 0;
    resmag = GetResRoot(phi, phi_old, res, 0, p);
    printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);
    std::chrono::time_point<std::chrono::steady_clock> begin_time = std::chrono::steady_clock::now();
    // Total time steps = PERIOD
    while (t < PERIOD) {
        ncycle += 1; 
//#pragma acc data present(phi) present(phi_old) present(res)
        v_cycle(phi, phi_old, res, p);
        resmag = GetResRoot(phi, phi_old, res, 0, p);
        printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);

        // Source varies with time
        if (resmag < RESGOAL) {
            t += 1;
            res[p.N/2 + (p.N/2)*p.N] = 1.0*TSTRIDE*p.scale*(1+sin(2.0*PI*t/(PERIOD)));
            ncycle = 0;
            for (lev = 0; lev < p.Lmax+1; lev++) {
                for (i = 0; i < p.size[lev]*p.size[lev]; i++) {
                    phi_old[p.index[lev]+i] = phi[p.index[lev]+i];
                }
            }
            double *draw;
            draw = (double *) malloc(p.N * p.N * sizeof(double));
            for (i=0; i<=p.N*p.N; i++)
            {
                draw[i]=phi[i];
            }
            /*image = Mat(p.size[0], p.size[0], CV_64FC1, draw);                  // opencv
            image.convertTo(image, CV_8UC1, 255.0);
            oVideoWriter.write(image); 
            imshow("test", image);                                              // opencv
            waitKey(1);*/                                                    // opencv 25ms per frame
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
    //oVideoWriter.release();
    std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> difference_in_time = end_time - begin_time;
    double difference_in_seconds = difference_in_time.count();
    printf("time: %f\n", difference_in_seconds);
    return 0;
}

void v_cycle(double *phi, double *phi_old, double *res, param p) {
//#pragma acc data present(phi) present(phi_old) present(res)
    int lev;
    // Go down
    for (lev = 0; lev < NLEV; lev++) {    

        // Get the new phi (smooth) and use it to compute residue, then project to the coarser level
        relax(phi, phi_old, res, lev, p);

        // Get the projected residue and use it to compute the error of the previous level, which is phi on this level (RECURSIVE). 
        proj_res(res, res, phi, phi_old, lev,p);
    }

    // Go up
    for (lev = NLEV; lev >= 0; lev--) { 
        
        // Use the newly computed res to get a new phi. 
        relax(phi, phi_old, res, lev, p);   // lev = NLEV -1, ... 0;

        // Interpolate to the finer level. the phi on the coarse level is the error on the fine level. 
        if (lev > 0) {
            inter_add(phi, phi, lev, p);   // phi[lev-1] += error = P phi[lev] and set phi[lev] = 0;
        }
    }
}


void relax(double *phi, double *phi_old, double *res, int lev, param p) {
//#pragma acc data present(phi) present(phi_old) present(res) //create(tmp[0:(L-2)*(L-2)])
    int i, x, y;
    int L;
    double* tmp;
    tmp = (double*)malloc((p.size[lev]-2)*(p.size[lev]-2)*sizeof(double));
    L  = p.size[lev];
    int point = p.index[lev];
//#pragma acc data create(tmp[0:(L-2)*(L-2)])
    for (i = 0; i < N_PER_LEV; i++) {
    #pragma acc loop independent
        for (x = 1; x < L-1; x++)
        #pragma acc loop independent
            for (y = 1; y < L-1; y++)
                tmp[y-1+(x-1)*(L-2)] = 0.5 * (res[point + y + x*L]
                            + TSTRIDE*p.scale*(phi[point + y+1 + x*L] + phi[point + y-1 + x*L] 
                            + phi[point + y + (x+1)*L] + phi[point + y + (x-1)*L])
                            + p.scale*phi_old[point + y + x*L] + phi[point + y + x*L]);
                // a coarse phi is the error of a fine phi
    #pragma acc loop independent
        for (x = 1; x < L-1; x++)
        #pragma acc loop independent
            for (y = 1; y < L-1; y++) 
                phi[point + y + x*L] = tmp[y-1 + (x-1)*(L-2)];
    }
    free(tmp);
    return;
}

void proj_res(double *res_c, double *res_f, double *phi_f, double *phi_old_f, int lev, param p) {  
    int L, Lc, f_off, c_off, x, y;
    L = p.size[lev];
    double r[L*L];          // residue of Ae = r
    Lc = p.size[lev+1];     // coarse level
    int point = p.index[lev];
    int pointc = p.index[lev+1];
//#pragma acc data present(res) present(phi) present(phi_old)
    //get residue
    for(x = 1; x < L-1; x++)
        //#pragma acc parallel loop
        for(y = 1; y < L-1; y++)
            r[x*L + y] = res_f[point + x*L + y] - phi_f[point + x*L + y] + p.scale*phi_old_f[point + x*L + y] 
                + TSTRIDE*p.scale*(phi_f[point + y+1 + x*L] + phi_f[point + y-1 + x*L] + phi_f[point + y + (x+1)*L] 
                    + phi_f[point + y + (x-1)*L]);

    //project residue
    for(x = 1; x < Lc-1; x++)
        //#pragma parallel loop
        for(y = 1; y < Lc-1; y++)
            res_c[pointc + x*Lc + y] = 0.25*(r[(2*x-1)*L + (2*y-1)] + r[(2*y) + (2*x-1)*L] + r[(2*y-1) + (2*x)*L] + r[(2*y) + (2*x)*L]);
    return;
}

void inter_add(double *phi_f, double *phi_c, int lev, param p) {  
    int L, Lc, x, y;
    Lc = p.size[lev];  // coarse  level
    L = p.size[lev-1]; 
    int point = p.index[lev];
    int pointc = p.index[lev-1];
//#pragma acc data present(res) present(phi) present(phi_old)
    for (x = 1; x < Lc-1; x++)
        //#pragma parallel loop
        for (y = 1; y < Lc-1; y++) {

            // Add the error back to phi
            phi_f[pointc + (2*y-1) + (2*x-1)*L] += phi_c[point + y + x*Lc];
            phi_f[pointc + (2*y) + (2*x-1)*L] += phi_c[point + y + x*Lc];
            phi_f[pointc + (2*y-1) + (2*x)*L] += phi_c[point + y + x*Lc];
            phi_f[pointc + (2*y) + (2*x)*L] += phi_c[point + y + x*Lc];
        }
    //set to zero so phi = error 
    for (x = 1; x < Lc-1; x++)
        //#pragma parallel loop
        for (y = 1; y < Lc-1; y++)
            phi_c[point + y + x*Lc] = 0.0;
    return;
}

double GetResRoot(double *phi, double *phi_old, double *res, int lev, param p) {
    int i, x, y;
    double residue;
    double ResRoot = 0.0;
    int L;
    int point = p.index[lev];
    L  = p.size[lev];
//#pragma acc data present(res) present(phi) present(phi_old)
    for (x = 1; x < L-1; x++)
        //#pragma parallel loop
        for (y = 1; y < L-1; y++) {
            residue = res[point + y + x*L]/p.scale/TSTRIDE - phi[point + y + x*L]/p.scale/TSTRIDE  
                    + (phi[point + (y+1) + x*L] + phi[point + (y-1) + x*L] 
                    + phi[point + y + (x+1)*L] + phi[point + y + (x-1)*L])
                    + phi_old[point + y + x*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
    return sqrt(ResRoot);
}