CXX       = g++

DEFINES   = -DDISABLE_RANDOM_SEED

# флаги компиляции
CXXFLAGS  =
CXXFLAGS += -std=c++11 -O2
CXXFLAGS += -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(DEFINES)
CXXFLAGS += $(shell pkg-config --cflags blitz)
CXXFLAGS += $(shell pkg-config --cflags gsl)

# флаги сборки (библиотеки)
LDFLAGS   = 
LDFLAGS  += -llapack -lblas -lgfortran
LDFLAGS  += $(shell pkg-config --libs blitz)
LDFLAGS  += $(shell pkg-config --libs gsl)

SOURCES   = main.cc
BINARY    = autoreg

VISUAL    = visual
VISUAL_SOURCES = visual.cc
VISUAL_LDFLAGS = $(shell pkg-config --libs freeglut) -lGL

# DCMT
CXXFLAGS += -Idcmt0.6.2/include
LDFLAGS  += -Ldcmt0.6.2/lib -ldcmt

DCMT = dcmt
DCMT_SOURCES = dcmt.cc
DCMT_LDFLAGS = -Ldcmt0.6.2/lib -ldcmt

$(BINARY): $(SOURCES) *.hh Makefile
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $(BINARY)

$(VISUAL): Makefile *.hh $(VISUAL_SOURCES)
	$(CXX) $(CXXFLAGS) $(VISUAL_SOURCES) $(VISUAL_LDFLAGS) -o $(VISUAL)

$(DCMT): Makefile *.hh $(DCMT_SOURCES)
	$(CXX) $(CXXFLAGS) $(DCMT_SOURCES) $(DCMT_LDFLAGS) -o $(DCMT)

run: ../tests ../tests/autoreg.model
run: $(BINARY)
	(cd ../tests; $(PWD)/$(BINARY))

debug: ../tests ../tests/autoreg.model
debug: $(BINARY)
	(cd ../tests; gdb $(PWD)/$(BINARY))

../tests:
	mkdir -p ../tests

../tests/autoreg.model: autoreg.model
	cp autoreg.model ../tests

clean:
	rm -f $(BINARY) $(VISUAL) $(DCMT)
