CXX       = g++

DEFINES   = -DDISABLE_RANDOM_SEED

CXXFLAGS  =
CXXFLAGS += -std=c++11 -O2
CXXFLAGS += -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(DEFINES)
CXXFLAGS += $(shell pkg-config --cflags blitz)

LDFLAGS   = 
LDFLAGS  += -llapack -lblas -lgfortran
LDFLAGS  += $(shell pkg-config --libs blitz)

SOURCES   = main.cc
BINARY    = autoreg

VISUAL    = visual
VISUAL_SOURCES = visual.cc
VISUAL_LDFLAGS = -lglut -lGL

$(BINARY): $(SOURCES) *.hh Makefile
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $(BINARY)

$(VISUAL): Makefile *.hh $(VISUAL_SOURCES)
	$(CXX) $(CXXFLAGS) $(VISUAL_SOURCES) $(VISUAL_LDFLAGS) -o $(VISUAL)

run: ../tests ../tests/autoreg.model ../tests/MersenneTwister.dat
run: $(BINARY)
	(cd ../tests; $(PWD)/$(BINARY))

debug: ../tests ../tests/autoreg.model ../tests/MersenneTwister.dat
debug: $(BINARY)
	(cd ../tests; gdb $(PWD)/$(BINARY))

../tests:
	mkdir -p ../tests

../tests/autoreg.model: ../input/autoreg.model
	cp ../input/autoreg.model ../tests

../tests/MersenneTwister.dat: ../input/MersenneTwister.dat
	cp ../input/MersenneTwister.dat ../tests

clean:
	rm -f $(BINARY)
