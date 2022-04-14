all:	intersect-lines-2d.o fix-perspective.o
	g++ -O9 -Wall -o a.out intersect-lines-2d.o fix-perspective.o `pkg-config --libs opencv4`

intersect-lines-2d.o: intersect-lines-2d.c++
	g++ -c -O9 -Wall `pkg-config --cflags opencv4` `pkg-config --cflags eigen3`  intersect-lines-2d.c++

fix-perspective.o: fix-perspective.c++
	g++ -c -O9 -Wall `pkg-config --cflags opencv4` fix-perspective.c++
