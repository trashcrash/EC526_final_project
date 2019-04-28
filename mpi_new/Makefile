CXX=mpicxx
CPPFLAGS=-c
all: mpi serial mpi_w

mpi: mpi.o
	$(CXX) time_dependent_v_mpi_mg.cpp -o mpi

mpi.o: time_dependent_v_mpi_mg.cpp
	$(CXX) $(CPPFLAGS) time_dependent_v_mpi_mg.cpp

serial: serial.o
	$(CXX) time_dependent_v.cpp -o serial

serial.o: time_dependent_v.cpp
	$(CXX) $(CPPFLAGS) time_dependent_v.cpp

mpi_w: mpi_w.o
	$(CXX) time_dependent_w_mpi_mg.cpp -o mpi_w

mpi_w.o: time_dependent_w_mpi_mg.cpp
	$(CXX) $(CPPFLAGS) time_dependent_w_mpi_mg.cpp

clean: 
	rm -rf mpi_v *.o* *.po* mpi_w mpi mpi.o serial *~
