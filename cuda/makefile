CC = nvcc
SRCDIR = src
OBJDIR = $(SRCDIR)/obj
OUTDIR = output

SRCS = $(wildcard $(SRCDIR)/*.cu)

OBJS = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SRCS))

TARGET = $(OUTDIR)/bellman-ford

$(TARGET): $(OBJS)
	@mkdir -p $(OUTDIR)
	$(CC) $^ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(CC) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(OUTDIR)

rebuild:
	make clean 
	make

.PHONY: clean
