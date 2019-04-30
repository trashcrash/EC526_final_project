# EC526_final_project
## Serial
The serial codes are stored in src folder. The four .cpp files for different cases are named as: </br>
```time_dependent_v.cpp```</br>
```time_dependent_w.cpp```</br>
```time_dependent_gauss.cpp```</br>
```time_dependent_gauss_w.cpp```</br></br>
Open any one of the .cpp files, there are a few parameters that can be adjusted. </br>
```#define NLEV``` allows you to adjust the total layers being used. </br>
```output = fopen(...)``` can be modified for different file names. If ran, the time consumption will be stored in the file. </br>
```p.Lmax``` variable controls the size of the grid, 8 means 512\*512 grid. </br></br>
To run the code, just ```make``` and run the corresponding executables. 
## MPI
The MPI code is in the mpi_new folder. </br>
The parameters can also be changed like the serial code. </br>
The .cpp files are named as: </br>
```time_dependent_v_mpi_mg.cpp```</br>
```time_dependent_w_mpi_mg.cpp```</br></br>
To compile, first load the modules: </br>
```module load intel/2018```</br>
```module load openmpi/3.0.0_intel-2018```</br>
Then ```make```. </br>
The compiled executables are ```mpi_v``` and ```mpi_w```. </br></br>
To run, use: </br>
```mpirun -np 4 NAME```</br>
## OpenACC
The OpenACC code is in the openACC folder. </br>
The parameters can also be changed like the serial code. </br>
The .cpp files are named as: </br>
```time_dependent_v_acc.cpp```</br>
```time_dependent_w_acc.cpp```</br></br>
To compile, first load the module: </br>
```module load pgi/18.4```</br>
Then ```make -f makeACC```. </br>
The compiled executable is named as ```test```. </br></br>
To run, first ask for a gpu on scc. </br>
Then simply run ```test```. 
