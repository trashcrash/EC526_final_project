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
void relax(double phi[SIZE][SIZE], double phi_old[SIZE][SIZE], double res[SIZE][SIZE], double tmp[SIZE-2][SIZE-2]);
//void proj_res(double *res_c, double *rec_f, double *phi_f, double *phi_old_f);
//void inter_add(double *phi_f, double *phi_c);
double GetResRoot(double phi[SIZE][SIZE], double phi_old[SIZE][SIZE], double res[SIZE][SIZE]);

int main() {
    int i, j;

    // Initialize phi and res for each level
    double phi[SIZE][SIZE];
    double phi_old[SIZE][SIZE];
    double res[SIZE][SIZE];
    double tmp[SIZE-2][SIZE-2];
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            phi[i][j] = 0.0;
            phi_old[i][j] = 0.0;
            res[i][j] = 0.0;
        }
    }
    for (i = 0; i < SIZE-2; i++) {
        for (j = 0; j < SIZE-2; j++) {
            tmp[i][j] = 0.0;
        }
    }
  
    res[SIZE/2][SIZE/2] = 1.0*TSTRIDE*scale;
    #pragma acc init
    #pragma data acc copy(phi[0:SIZE][0:SIZE]) copy(res[0:SIZE][0:SIZE]) copy(phi_old[0:SIZE][0:SIZE]) copy(tmp[0:SIZE-2][0:SIZE-2])
    {
    // iterate to solve
    double resmag = 1.0;
    int ncycle = 1;
    int t = 0;
    //resmag = GetResRoot(phi, phi_old, res);
    printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);

    //std::chrono::time_point<std::chrono::steady_clock> begin_time =
    //std::chrono::steady_clock::now();
 
    // Total time steps = PERIOD
    relax(phi, phi_old, res, tmp);
    while (t < PERIOD) {
        ncycle += 1; 
        relax(phi, phi_old, res, tmp);
        resmag = GetResRoot(phi, phi_old, res);
        printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);

        // Source varies with time
        if (resmag < RESGOAL) {
            t += 1;
            res[SIZE/2][SIZE/2] = 1.0*TSTRIDE*scale*(1+sin(2.0*PI*t/PERIOD));
            ncycle = 0;
            for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++) {
                    phi_old[i][j] = phi[i][j];
                }
            }
        }
    }
    }
    return 0;
}

void relax(double phi[SIZE][SIZE], double phi_old[SIZE][SIZE], double res[SIZE][SIZE], double tmp[SIZE-2][SIZE-2]) {
    int i, x, y;
    //for (i = 0; i < N_PER_LEV; i++) {   
    #pragma acc data present(phi[0:SIZE][0:SIZE]) present(phi_old[0:SIZE][0:SIZE]) present(res[0:SIZE][0:SIZE]) present(tmp[0:SIZE-2][0:SIZE-2])
    {
        #pragma acc parallel loop collapse(2)
        for (x = 1; x < SIZE-1; x++) {
            for (y = 1; y < SIZE-1; y++) {
                tmp[x-1][y-1] = 0.5*(res[x][y]
                            + TSTRIDE*scale*(phi[x][y+1] + phi[x][y-1] 
                            + phi[x+1][y] + phi[x-1][y])
                            + scale*phi_old[x][y] + phi[x][y]);
            }
        }
                // a coarse phi is the error of a fine phi
        #pragma acc parallel loop collapse(2)
        for (x = 1; x < SIZE-1; x++) {
            for (y = 1; y < SIZE-1; y++) {
                phi[x][y] = tmp[x-1][y-1];
            }
        }
    }
//}
    return;
}

double GetResRoot(double phi[SIZE][SIZE], double phi_old[SIZE][SIZE], double res[SIZE][SIZE]) {
    int x, y;
    double residue;
    double ResRoot = 0.0;

    for (x = 1; x < SIZE-1; x++)
        for (y = 1; y < SIZE-1; y++) {
            residue = res[x][y]/scale/TSTRIDE - phi[x][y]/scale/TSTRIDE  
                    + (phi[x][y+1] + phi[x][y-1] 
                    + phi[x+1][y] + phi[x-1][y])
                    + phi_old[x][y]/TSTRIDE;
            ResRoot += residue*residue; // true residue
        }
    return sqrt(ResRoot);
}
