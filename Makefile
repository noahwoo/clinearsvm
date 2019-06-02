CC=gcc

# C-Compiler flags
CFLAGS= -O3 -g -pg -fno-strict-aliasing -Wall
# CFLAGS= -O3 -g -Wall

# linker
LD=gcc
LFLAGS= -O3 -Wall -lstdc++


all: linsvm_train linsvm_classify
	cp linsvm_train linsvm_classify ~/bin
clean:
	rm *.o linsvm_train linsvm_classify
		
snglin.o: snglin.h snglin.cpp
	$(CC) -c $(CFLAGS) snglin.cpp -o snglin.o
linsvm_train.o: linsvm_train.h linsvm_train.cpp
	$(CC) -c $(CFLAGS) linsvm_train.cpp -o linsvm_train.o
linsvm_main.o: linsvm_main.cpp
	$(CC) -c $(CFLAGS) linsvm_main.cpp -o linsvm_main.o
linsvm_main_classify.o: linsvm_main_classify.cpp
	$(CC) -c $(CFLAGS) linsvm_main_classify.cpp -o linsvm_main_classify.o
	
linsvm_train: snglin.o linsvm_train.o linsvm_main.o  
	$(LD) $(LFLAGS) snglin.o linsvm_train.o linsvm_main.o -lm -o linsvm_train

linsvm_classify: linsvm_main_classify.o linsvm_train.o snglin.o
	$(LD) $(LFLAGS) snglin.o linsvm_main_classify.o linsvm_train.o -o linsvm_classify
