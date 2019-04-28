CC=mpicxx
CFLAGS=-c
all: test

test: mpi.o
	$(CC) time_dependent_v_mpi_mg.cpp -o mpi

mpi.o: time_dependent_v_mpi_mg.cpp
	$(CC) $(CFLAGS) time_dependent_v_mpi_mg.cpp

fixed_Lx: fixed_Lx.o
	$(CC) jacobi_mpi_2d_fixed_Lx.cpp -o fixed_Lx

fixed_Lx.o: jacobi_mpi_2d_fixed_Lx.cpp
	$(CC) $(CFLAGS) jacobi_mpi_2d_fixed_Lx.cpp

fixed_proc: fixed_proc.o
	$(CC) jacobi_mpi_2d_fixed_proc.cpp -o fixed_proc

fixed_proc.o: jacobi_mpi_2d_fixed_proc.cpp
	$(CC) $(CFLAGS) jacobi_mpi_2d_fixed_proc.cpp

clean: 
	rm -rf *.o* *.po* fixed_proc.o fixed_proc fixed_Lx.o fixed_Lx mpi mpi.o *~
