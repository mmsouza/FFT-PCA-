import FFT_PCA_BINARY
import PCA_KNN
import pca_ann_bin
import ANN_without_fft_pca
import matplotlib.pyplot as plt

#FFT_PCA_BINARY.pca_fft_ann()
PCA_KNN.pca_KNN()
plt.figure()
pca_ann_bin.pca_ann()
plt.figure()
ANN_without_fft_pca.ANN()
plt.show()

