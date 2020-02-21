TARGET          = stencil

CC		= g++
CFLAGS		= 

NVCC		= nvcc
NVCFLAGS	= -Xcompiler -fopenmp 

DCFLAGS         := -g -ggdb3 -O0
RCFLAGS         := -march=native -O2 -pipe

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
