CXX      := g++
CXXFLAGS := -std=c++23 -O2 -Wall -Wextra -pedantic -MMD -MP
LDFLAGS  :=
LDLIBS   :=

SRC  := 0d.cc cell.cc
OBJ  := $(SRC:.cc=.o)
DEPS := $(OBJ:.o=.d)

TARGET := 0d

# Default target
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(DEPS)

.PHONY: all clean
clean:
	rm -f $(OBJ) $(DEPS) $(TARGET)