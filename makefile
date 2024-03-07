CC = gcc
CFLAGS = -Wall -Iinclude -fopenmp
SRCDIR = src
OBJDIR = $(SRCDIR)/obj
LDFLAGS= -fopenmp
OUTDIR = output

# List of source files
SRCS = $(wildcard $(SRCDIR)/*.c)

# List of object files
OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

# Target binary
TARGET = $(OUTDIR)/bellman-ford

# Main target
$(TARGET): $(OBJS)
	@mkdir -p $(OUTDIR)
	$(CC) -fopenmp $^ -o $@

# Rule for creating object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf $(OBJDIR) $(OUTDIR)

rebuild:
	make clean 
	make

.PHONY: clean

