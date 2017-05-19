SRC_DIR = ./
INC_DIR = ./
OBJ_DIR = ./obj
BIN_DIR = ./bin
CC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc
DEBUG=#-g -G


INCLUDE = -I${INC_DIR}
LIBS = -lstdc++ -lcudart -lcuda

CFLAGS = -Wall -Wextra 
NVCC_FLAG = -arch=sm_52 -Xcompiler ${DEBUG} #--default-stream per-thread

CPPOBJECTS = ${OBJ_DIR}/filterbank.o

CUDAOBJECTS = ${OBJ_DIR}/dedisperse.o ${OBJ_DIR}/kernels.o

all: dedisperse
dedisperse: ${CUDAOBJECTS} ${CPPOBJECTS}
	${NVCC} ${NVCC_FLAG} ${INCLUDE} ${LIBS} ${CUDAOBJECTS} ${CPPOBJECTS} -o ${BIN_DIR}/dedisperse

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu
	${NVCC} -c ${NVCC_FLAG} ${INCLUDE} $< -o $@

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp
	${CC} -c ${CFLAGS} ${INCLUDE} $< -o $@

.PHONY: clean

clean:
	rm -f ${OBJ_DIR}/*.o ${BIN_DIR}/*
