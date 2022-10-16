GDI_SLAB:
	g++ -fopenmp GDI_SLAB.cpp spectralFunctions.cpp arithmeticFunctions.cpp -lfftw3 -lm -O3 -o test.out

clean:
	# build files
	rm -f test.out

	# GDI_SLAB files
	rm -f phi_* ne_* Y.txt X.txt tVec.txt *.gkyl phi*.txt Te*.txt Ti*.txt ne*.txt
