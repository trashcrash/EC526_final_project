.SUFFIXES:
.SUFFIXES: .o .cpp

#============================================================
TARGET2 = multigrid_2d

C_SOURCES = multigrid_2d.cpp
C_OBJS2 = multigrid_2d.o
MY_INCLUDES = 

CCX = g++
CXXFLAGS = -g -O2  $(INC)

#============================================================
all: $(TARGET2)

.o:.cpp	$(MY_INCLUDES)
	$(CCX)  -c  $(CXXFLAGS) $<  

$(TARGET2) :   $(C_OBJS2)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

# Implicit rules: $@ = target name, $< = first prerequisite name, $^ = name of all prerequisites 
#============================================================

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)

NOTES =

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(C_OBJS1) $(C_OBJS2) $(C_OBJS3) core  *~


