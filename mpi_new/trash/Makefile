CC=mpicxx
CFLAGS=-c
all: mpi

mpi: mpi.o
	$(CC) time_dependent_v_mpi_mg.cpp -o mpi

mpi.o: time_dependent_v_mpi_mg.cpp
	$(CC) $(CFLAGS) time_dependent_v_mpi_mg.cpp

clean: 
	rm -rf *.o* *.po* mpi mpi.o *~
