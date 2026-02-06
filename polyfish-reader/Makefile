CXX := g++
CXXFLAGS := -O2 -std=c++17

SRC :=  json.hpp polyfish-reader.cpp reader_util.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := polyfish-reader

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)