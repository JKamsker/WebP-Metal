include makefile.unix

.PHONY: metal benchmark-lossy-import clean-metal

metal: examples/cwebp examples/dwebp
	@ln -sf examples/cwebp cwebp-metal
	@ln -sf examples/dwebp dwebp-metal

benchmark-lossy-import: metal
	@mkdir -p build
	$(CXX) -O3 -std=c++17 -Isrc benchmarks/benchmark_lossy_import.cc \
		src/libwebp.a -framework Foundation -framework Metal -lpthread -lm \
		-o build/benchmark_lossy_import

clean-metal: clean
	@rm -f cwebp-metal dwebp-metal
