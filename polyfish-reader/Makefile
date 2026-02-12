CXX := g++
CXXFLAGS := -O2 -std=c++17

SRC := polyfish-reader.cpp reader_util.cpp
OBJ := $(SRC:.cpp=.o)
TARGETS := polyfish-reader polyfish-scanner

all: $(TARGETS)

polyfish-reader: polyfish-reader.o reader_util.o
	$(CXX) $(CXXFLAGS) -o $@ $^

polyfish-scanner: polyfish-scanner.o reader_util.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)