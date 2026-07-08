CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
LDFLAGS ?= -lm

all: spectral_infer spectral_txt_to_bin

spectral_infer: main.c model.c spectral_scaler.c acids_spectral_preprocess.c acids_spectral_rfft_tables.h acids_samples.h
	$(CC) $(CFLAGS) -o $@ main.c model.c spectral_scaler.c acids_spectral_preprocess.c $(LDFLAGS)

spectral_txt_to_bin: spectral_txt_to_bin.c
	$(CC) $(CFLAGS) -o $@ spectral_txt_to_bin.c $(LDFLAGS)

clean:
	rm -f spectral_infer spectral_txt_to_bin libmodel_w8.so libspectral_scaler.so libacids_spectral_preprocess.so
