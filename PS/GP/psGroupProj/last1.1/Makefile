CC = gcc
all: directories target run

directories:
	mkdir -p haresDensity
	mkdir -p haresPPM
	mkdir -p pumasPPM
	mkdir -p pumasDensity
	mkdir -p togetherPPM

OBJH= printFunctions.h dataStructiars.h equations.h global_values.h
OBJC= printFunctions.c dataStructiars.c equations.c main.c
OBJO= printFunctions.o dataStructiars.o equations.o main.o

target : $(OBJC) $(OBJH)
	$(CC) $(OBJC)


.PHONY : run
run: target
	./target


.PHONY : clean
clean:
	rm main.o dataStructiars.o equations.o printFunctions.o
	rm -f
	rm -r haresDensity
	rm -r haresPPM
	rm -r pumasDensity
	rm -r pumasPPM
	rm -r togetherPPM
