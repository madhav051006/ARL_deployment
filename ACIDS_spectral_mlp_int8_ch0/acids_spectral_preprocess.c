#include "acids_spectral_preprocess.h"

#include <stddef.h>

#include "acids_spectral_rfft_tables.h"

static void spectral_rfft_power(const float *time_in, double *power_out)
{
    const int n = ACIDS_SPECTRAL_N_FFT;
    const int n_bins = ACIDS_SPECTRAL_N_BINS;

    for (int k = 0; k < n_bins; ++k) {
        double real = 0.0;
        double imag = 0.0;
        for (int t = 0; t < n; ++t) {
            const double x = (double)time_in[t];
            real += x * (double)ACIDS_SPECTRAL_RFFT_REAL[k][t];
            imag += x * (double)ACIDS_SPECTRAL_RFFT_IMAG[k][t];
        }
        power_out[k] = real * real + imag * imag;
    }
}

int acids_spectral_preprocess(const float *input_1600, float *out_features_83)
{
    float time_buf[ACIDS_SPECTRAL_N_FFT];
    double power[ACIDS_SPECTRAL_N_BINS];
    double power_sum = (double)ACIDS_SPECTRAL_EPS;
    double weighted_freq_sum = 0.0;

    if (input_1600 == NULL || out_features_83 == NULL) {
        return -1;
    }

    for (int t = 0; t < ACIDS_SPECTRAL_N_FFT; ++t) {
        time_buf[t] = input_1600[t];
    }

    spectral_rfft_power(time_buf, power);

    double power_sum_raw = 0.0;
    for (int k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
        power_sum_raw += power[k];
        weighted_freq_sum += power[k] * (double)ACIDS_SPECTRAL_FREQS[k];
    }

    power_sum = power_sum_raw + (double)ACIDS_SPECTRAL_EPS;
    out_features_83[0] = (float)(weighted_freq_sum / power_sum);
    out_features_83[1] = (float)(power_sum_raw / (double)ACIDS_SPECTRAL_N_BINS);

    for (int k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
        out_features_83[2 + k] = (float)(power[k] / power_sum);
    }

    return 0;
}
