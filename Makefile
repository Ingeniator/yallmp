# Makefile

.PHONY: all clean build

all: build

build:
    @echo "Building..."
    @echo "Stage 1: Preparing environment."
    @echo "Stage 2: Compiling sources."
    @echo "Stage 3: Linking binaries."
    @echo "Build complete!"

clean:
    @rm -f *.o