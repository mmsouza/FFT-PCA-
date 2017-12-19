import FFT_PCA_BINARY
import PCA_KNN
import pca_ann_bin
import ANN_without_fft_pca
import matplotlib.pyplot as plt


ANN_without_fft_pca.ANN(n_modes=8, fault_prop=.5, pcs=52, repetitions=1, filename='ANN', batchsize=320)

