include makefile.unix

.PHONY: metal clean-metal

metal: examples/cwebp examples/dwebp
	@ln -sf examples/cwebp cwebp-metal
	@ln -sf examples/dwebp dwebp-metal

clean-metal: clean
	@rm -f cwebp-metal dwebp-metal
