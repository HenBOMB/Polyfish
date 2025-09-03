CXX := g++
CXXFLAGS := -O2 -std=c++17

SRC := polyai-reader.cpp reader_util.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := polyai-reader

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)