CXX := g++
CXXFLAGS := -O2 -std=c++17

SRC := check_runtime.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := check_runtime

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)