# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -mavx -fopenmp

# Executable and library names
EXEC = runme
TEST_EXEC = test_exec
LIB = libfilmmaster2000.a

# Target to build the executable
all: $(EXEC)

# Link the main executable
$(EXEC): main.o $(LIB)
	$(CXX) $(CXXFLAGS) -o $(EXEC) main.o $(LIB)

# Create the static library
$(LIB): FilmMaster2000.o
	ar rcs $(LIB) FilmMaster2000.o

# Compile main.cpp
main.o: main.cpp FilmMaster2000.h
	$(CXX) $(CXXFLAGS) -c main.cpp

# Compile FilmMaster2000.cpp
FilmMaster2000.o: FilmMaster2000.cpp FilmMaster2000.h
	$(CXX) $(CXXFLAGS) -c FilmMaster2000.cpp

# Compile test.cpp
test.o: test.cpp FilmMaster2000.h
	$(CXX) $(CXXFLAGS) -c test.cpp

# Link the test executable
$(TEST_EXEC): test.o $(LIB)
	$(CXX) $(CXXFLAGS) -o $(TEST_EXEC) test.o $(LIB)

# Target to run tests
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Run example tests
run_tests: $(EXEC)
	@echo "Running tests..."
	./$(EXEC) data/large.bin output_reverse.bin reverse
	./$(EXEC) data/large.bin output_swap_channel.bin swap_channel 1,2
	./$(EXEC) data/large.bin output_clip_channel.bin clip_channel 1 [10,200]
	./$(EXEC) data/large.bin output_scale_channel.bin scale_channel 1 1.5
	./$(EXEC) data/large.bin output_grayscale.bin grayscale
	./$(EXEC) data/large.bin output_reverse_speed.bin -S reverse
	./$(EXEC) data/large.bin output_swap_channel_speed.bin -S swap_channel 1,2
	./$(EXEC) data/large.bin output_clip_channel_speed.bin -S clip_channel 1 [10,200]
	./$(EXEC) data/large.bin output_scale_channel_speed.bin -S scale_channel 1 1.5
	./$(EXEC) data/large.bin output_grayscale_speed.bin -S grayscale
	./$(EXEC) data/large.bin output_reverse_memory.bin -M reverse
	./$(EXEC) data/large.bin output_swap_channel_memory.bin -M swap_channel 1,2
	./$(EXEC) data/large.bin output_clip_channel_memory.bin -M clip_channel 1 [10,200]
	./$(EXEC) data/large.bin output_scale_channel_memory.bin -M scale_channel 1 1.5
	./$(EXEC) data/large.bin output_grayscale_memory.bin -M grayscale
	@echo "Tests completed."

# Run example tests
run_smalltests: $(EXEC)
	@echo "Running tests..."
	./$(EXEC) data/test.bin output_reverse.bin reverse
	./$(EXEC) data/test.bin output_swap_channel.bin swap_channel 1,2
	./$(EXEC) data/test.bin output_clip_channel.bin clip_channel 1 [10,200]
	./$(EXEC) data/test.bin output_scale_channel.bin scale_channel 1 1.5
	./$(EXEC) data/test.bin output_grayscale.bin grayscale
	./$(EXEC) data/test.bin output_reverse_speed.bin -S reverse
	./$(EXEC) data/test.bin output_swap_channel_speed.bin -S swap_channel 1,2
	./$(EXEC) data/test.bin output_clip_channel_speed.bin -S clip_channel 1 [10,200]
	./$(EXEC) data/test.bin output_scale_channel_speed.bin -S scale_channel 1 1.5
	./$(EXEC) data/test.bin output_grayscale_speed.bin -S grayscale
	./$(EXEC) data/test.bin output_reverse_memory.bin -M reverse
	./$(EXEC) data/test.bin output_swap_channel_memory.bin -M swap_channel 1,2
	./$(EXEC) data/test.bin output_clip_channel_memory.bin -M clip_channel 1 [10,200]
	./$(EXEC) data/test.bin output_scale_channel_memory.bin -M scale_channel 1 1.5
	./$(EXEC) data/test.bin output_grayscale_memory.bin -M grayscale
	@echo "Tests completed."


# Clean up build files
clean:
	rm -f *.o $(EXEC) $(TEST_EXEC) $(LIB) output_file.bin

# Clean up test output files
clean_tests:
	 rm -f output_reverse.bin \
          output_swap_channel.bin \
          output_clip_channel.bin \
          output_scale_channel.bin \
		  output_grayscale.bin \
          output_reverse_speed.bin \
          output_swap_channel_speed.bin \
          output_clip_channel_speed.bin \
          output_scale_channel_speed.bin \
		  output_grayscale_speed.bin \
          output_reverse_memory.bin \
          output_swap_channel_memory.bin \
          output_clip_channel_memory.bin \
          output_scale_channel_memory.bin \
		  output_grayscale_memory.bin
