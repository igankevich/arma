CXX       = g++

DEFINES   = -DDISABLE_RANDOM_SEED

CXXFLAGS  =
CXXFLAGS += -std=c++11 -O2
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(DEFINES)

LDFLAGS   = 
LDFLAGS  += -llapack -lblas -lgfortran

SOURCES   = main.cc
BINARY    = autoreg

$(BINARY): $(SOURCES) *.hh Makefile
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $(BINARY)

run: ../tests ../tests/autoreg.model ../tests/MersenneTwister.dat
run: $(BINARY)
	(cd ../tests; $(PWD)/$(BINARY))

../tests:
	mkdir -p ../tests

../tests/autoreg.model: ../input/autoreg.model
	cp ../input/autoreg.model ../tests

../tests/MersenneTwister.dat: ../input/MersenneTwister.dat
	cp ../input/MersenneTwister.dat ../tests

clean:
	rm -f $(BINARY)
