CC = gcc
CFLAGS = -g -O3 -Wall -fopenmp -march=native -ffast-math

# create main executable
stonkfish: stonkfish.o
	$(CC) $(CFLAGS) *.o

%.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o a.out
