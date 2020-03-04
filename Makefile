TARGET          = stencil

CC		= g++
CFLAGS		= -fopenmp

NVCC		= nvcc
NVCFLAGS	= -Xcompiler -fopenmp -dc

DCFLAGS         := -g -ggdb3 -O0
RCFLAGS         := -O2 -ftree-loop-vectorize -ftree-vectorize -finline-functions -funswitch-loops -s -pipe -march=native

NVDCFLAGS       := -g -G
NVRCFLAGS       := -O2 

SRC		= $(wildcard *.c)
INC		= $(wildcard *.h)
OBJ	        = $(SRC:%.c=%.o)

NVSRC		= $(wildcard *.cu)
NVINC		= $(wildcard *.cuh)
NVOBJ	        = $(NVSRC:%.cu=%.o)

LIBS		= -lglfw -lGLEW -lGLU -lGL -lm -lpthread -lgomp


.PHONY: debug 
debug: NVCFLAGS += $(NVDCFLAGS)
debug: CFLAGS += $(DCFLAGS)
debug: build

.PHONY: release
release: NVCFLAGS += $(NVRCFLAGS)
release: CFLAGS += $(RCFLAGS)
release: build


build: $(OBJ) $(NVOBJ)
	@$(NVCC) $(LDFLAGS) $(OBJ) $(NVOBJ) -o $(TARGET) $(LIBS)
	@echo [LD] Linked $(OBJ) $(NVOBJ) into binary $(TARGET)

%.o:%.cu
	@$(NVCC) $(NVCFLAGS) -c $^ -o $@
	@echo [NVCC] Compiled $^ into $@

%.o:%.c
	@$(CC) $(CFLAGS) -c $^ -o $@
	@echo [CC] Compiled $^ into $@

.PHONY: clean
clean:
	@rm -f $(OBJ) $(NVOBJ) $(TARGET)
	@echo Cleaned $(OBJ) $(NVOJB) and $(TARGET)
