#include "c_speech_features.h"

void mfcc(const short* signal,
          unsigned int signal_len,
          int samplerate,
          csf_float winlen,
          csf_float winstep,
          int numcep,
          int nfilt,
          int nfft,
          int lowfreq,
          int highfreq,
          csf_float preemph,
          int ceplifter,
          int appendEnergy,
          csf_float* winfunc,
          csf_float** mfcc,
          int* mfcc_dim1,
          int* mfcc_dim2);

void fbank(const short* signal,
           unsigned int signal_len,
           int samplerate,
           csf_float winlen,
           csf_float winstep,
           int nfilt,
           int nfft,
           int lowfreq,
           int highfreq,
           csf_float preemph,
           csf_float* winfunc,
           csf_float** features,
           int* features_dim1,
           int* features_dim2,
           csf_float** energy,
           int* energy_dim1);

void logfbank(const short* signal,
              unsigned int signal_len,
              int samplerate,
              csf_float winlen,
              csf_float winstep,
              int nfilt,
              int nfft,
              int lowfreq,
              int highfreq,
              csf_float preemph,
              csf_float* winfunc,
              csf_float** features,
              int* features_dim1,
              int* features_dim2);

void ssc(const short* signal,
         unsigned int signal_len,
         int samplerate,
         csf_float winlen,
         csf_float winstep,
         int nfilt,
         int nfft,
         int lowfreq,
         int highfreq,
         csf_float preemph,
         csf_float* winfunc,
         csf_float** features,
         int* features_dim1,
         int* features_dim2);

csf_float hz2mel(csf_float hz);

csf_float mel2hz(csf_float mel);

void get_filterbanks(int nfilt,
                     int nfft,
                     int samplerate,
                     int lowfreq,
                     int highfreq,
                     csf_float** filterbanks,
                     int* filterbanks_dim1,
                     int* filterbanks_dim2);

void lifter(csf_float* cepstra,
            int cepstra_dim1,
            int cepstra_dim2,
            int L,
            csf_float** cepstra_out,
            int* cepstra_out_dim1,
            int* cepstra_out_dim2);

void delta(csf_float* feat,
           int feat_dim1,
           int feat_dim2,
           int N,
           csf_float** delta,
           int* delta_dim1,
           int* delta_dim2);
