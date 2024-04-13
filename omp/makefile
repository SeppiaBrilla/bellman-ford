CC = gcc
CFLAGS = -Wall -Iinclude -fopenmp
SRCDIR = src
OBJDIR = $(SRCDIR)/obj
OUTDIR = output

SRCS = $(wildcard $(SRCDIR)/*.c)

OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

TARGET = $(OUTDIR)/bellman-ford

$(TARGET): $(OBJS)
	@mkdir -p $(OUTDIR)
	$(CC) -fopenmp $^ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(OUTDIR)

rebuild:
	make clean 
	make

.PHONY: clean

