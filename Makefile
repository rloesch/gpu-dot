.PHONY: all clean

all: gpu-dot

gpu-dot: gpu-dot.cu
	nvcc gpu-dot.cu -lcublas -o gpu-dot

clean:
	rm -f gpu-dot
