#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <opencv2/opencv.hpp>                       // opencv

using namespace cv;                                 // opencv

#define RESGOAL 1E-4
#define NLEV 4                                      // If 0, only one level
#define PERIOD 1
#define PI 3.141592653589793
#define TSTRIDE 1000000.0
#define N_PER_LEV 10                                // Iterate 10 times for each level

typedef struct{
    int N;
    int Lmax;
    int size[20];
    // double a[20];
    double m_square;
    double scale;
} param;

void relax(double *phi, double *phi_old, double *res, int lev, param p);
void proj_res(double *res_c, double *rec_f, double *phi_f, double *phi_old_f, int lev, param p);
void inter_add(double *phi_f, double *phi_c, int lev,param p);
double GetResRoot(double *phi, double *phi_old, double *res, int lev, param p);
void v_cycle(double **phi, double **phi_old, double **res, param p);

int main() {  
    Mat image;
    double *phi[20], *res[20], *phi_old[20];
    param p;
    int i, j, lev;

    // Initialize parameters
    p.Lmax = 7;
    p.N = 2*(int)pow(2,p.Lmax)+2;
    p.m_square = 0.0;                                     // Scaling parameter, a small number

    Size frame_size(p.N, p.N);
    int frames_per_second = 30;
    VideoWriter oVideoWriter("./MyVideo.mp4", CV_FOURCC('G','R','E','Y'), frames_per_second, frame_size, false);
    if (oVideoWriter.isOpened() == false) 
    {
        std::cout << "Cannot save the video to a file" << std::endl;
        std::cin.get(); //wait for any key press
        return -1;
    }

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

    // Initialize phi and res for each level
    for (lev = 0; lev < p.Lmax+1; lev++) {
        phi[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
        phi_old[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
        res[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
        for (i = 0; i < p.size[lev]*p.size[lev]; i++) {
            phi[lev][i] = 0.0;
            phi_old[lev][i] = 0.0;
            res[lev][i] = 0.0;
        }
    }  
  
    res[0][p.N/2 + (p.N/2)*p.N] = 1.0*TSTRIDE*p.scale;   // Unit point source in middle of N by N lattice 
  
    // iterate to solve
    double resmag = 1.0;                            // Not rescaled.
    int ncycle = 1; 
    int t = 0;
    resmag = GetResRoot(phi[0], phi_old[0], res[0], 0, p);
    printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);
 
    // Total time steps = PERIOD
    while (t < PERIOD) {
        ncycle += 1; 
        v_cycle(phi, phi_old, res, p);
        resmag = GetResRoot(phi[0], phi_old[0], res[0], 0, p);
        printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);
        // Source varies with time
        if (resmag < RESGOAL) {
            t += 1;
            res[0][p.N/2 + (p.N/2)*p.N] = 1.0*TSTRIDE*p.scale*(1+sin(2.0*PI*t/(PERIOD/1.0)));
            ncycle = 0;
            for (lev = 0; lev < p.Lmax+1; lev++) {
                for (i = 0; i < p.size[lev]*p.size[lev]; i++) {
                    phi_old[lev][i] = phi[lev][i];
                }
            }
            image = Mat(p.size[0], p.size[0], CV_64FC1, phi[0]);                  // opencv
            image.convertTo(image, CV_8UC1, 255.0);
            oVideoWriter.write(image); 
            imshow("test", image);                                              // opencv
            waitKey(1);                                                        // opencv 25ms per frame
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
    oVideoWriter.release();
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

void relax(double *phi, double *phi_old, double *res, int lev, param p) {
    int i, x, y;
    int L;
    double* tmp;
    tmp = (double*)malloc((p.size[lev]-2)*(p.size[lev]-2)*sizeof(double));
    L  = p.size[lev];
    for (i = 0; i < N_PER_LEV; i++) {   
        for (x = 1; x < L-1; x++) {
            for (y = 1; y < L-1; y++) {
                tmp[y-1+(x-1)*(L-2)] = 0.5*(res[y + x*L]
                            + TSTRIDE*p.scale*(phi[y+1 + x*L] + phi[y-1 + x*L] 
                            + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                            + p.scale*phi_old[y + x*L] + phi[y + x*L]);
            }
        }
                // a coarse phi is the error of a fine phi
        for (x = 1; x < L-1; x++) {
            for (y = 1; y < L-1; y++) {
                phi[y + x*L] = tmp[y-1 + (x-1)*(L-2)];
            }
        }
    }
    free(tmp);
    return;
}

void proj_res(double *res_c, double *res_f, double *phi_f, double *phi_old_f, int lev, param p) {  
    int L, Lc, f_off, c_off, x, y;
    L = p.size[lev];
    double r[L*L];          // residue of Ae = r
    Lc = p.size[lev+1];     // coarse level
  
    //get residue
    for(x = 1; x < L-1; x++) {
        for(y = 1; y < L-1; y++) {
            r[x*L + y] = res_f[x*L + y] - phi_f[x*L + y] + p.scale*phi_old_f[x*L + y] 
                + TSTRIDE*p.scale*(phi_f[y+1 + x*L] + phi_f[y-1 + x*L] + phi_f[y + (x+1)*L] + phi_f[y + (x-1)*L]);
        }
    }

    //project residue
    for(x = 1; x < Lc-1; x++) {
        for(y = 1; y < Lc-1; y++) {
            res_c[x*Lc + y] = 0.25*(r[(2*x-1)*L + (2*y-1)] + r[(2*y) + (2*x-1)*L] + r[(2*y-1) + (2*x)*L] + r[(2*y) + (2*x)*L]);
        }
    }
}

void inter_add(double *phi_f, double *phi_c, int lev, param p) {  
    int L, Lc, x, y;
    Lc = p.size[lev];  // coarse  level
    L = p.size[lev-1]; 
    //printf("%.17f\t%.17f\t%.17f\t%.17f\n", phi_f[1+1*L], phi_f[1+2*L], phi_f[2+1*L], phi_f[2+2*L]);
    //printf("%.17f\n", phi_c[1+1*Lc]);
    for (x = 1; x < Lc-1; x++) {
        for (y = 1; y < Lc-1; y++) {

            // Add the error back to phi
            phi_f[(2*y-1) + (2*x-1)*L] += phi_c[y + x*Lc];
            phi_f[(2*y) + (2*x-1)*L] += phi_c[y + x*Lc];
            phi_f[(2*y-1) + (2*x)*L] += phi_c[y + x*Lc];
            phi_f[(2*y) + (2*x)*L] += phi_c[y + x*Lc];
        }
    }
    //set to zero so phi = error 
    for (x = 1; x < Lc-1; x++)
        for (y = 1; y < Lc-1; y++)
            phi_c[y + x*Lc] = 0.0;
    return;
}

double GetResRoot(double *phi, double *phi_old, double *res, int lev, param p) {
    int i, x, y;
    double residue;
    double ResRoot = 0.0;
    int L;
    L  = p.size[lev];

    for (x = 1; x < L-1; x++)
        for (y = 1; y < L-1; y++) {
            residue = res[y + x*L]/p.scale/TSTRIDE - phi[y + x*L]/p.scale/TSTRIDE  
                    + (phi[(y+1) + x*L] + phi[(y-1) + x*L] 
                    + phi[y + (x+1)*L] + phi[y + (x-1)*L])
                    + phi_old[y + x*L]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
    return sqrt(ResRoot);
}