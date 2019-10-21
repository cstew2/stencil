TARGET          = stencil

CC		= g++
CFLAGS		= 

NVCC		= nvcc
NVCFLAGS	=

DCFLAGS         := -g -g -O0
RCFLAGS         := -O2

SRC		= $(wildcard *.c)
INC		= $(wildcard *.h)
OBJ	        = $(SRC:%.c=%.o)

NVSRC		= $(wildcard *.cu)
NVINC		= $(wildcard *.cuh)
NVOBJ	        = $(NVSRC:%.cu=%.o)

LIBS		= -lglfw -lGLEW -lGLU -lGL -lm


.PHONY: debug 
debug: NVCFLAGS += $(DCFLAGS)
debug: CFLAGS += $(DCFLAGS)
debug: build

.PHONY: release
release: NVCFLAGS += $(RCFLAGS)
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
	@rm -f $(OBJ) $(NVOJB) $(TARGET)
	@echo Cleaned $(OBJ) $(NVOJB) and $(TARGET)
