ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

volta: 
	nvcc $(INC) $(LIB)  -I../../include -arch=sm_70 paillier.cu -o paillier -lgmp

clean:
	rm -f paillier


