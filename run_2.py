import fft_pca_ann
import pca_knn
import pca_ann
import plain_ann
import plain_knn
import fft_pca_knn
import matplotlib.pyplot as plt
import plain_svm
import fft_pca_svm
import pca_svm

a = [1, 2, 3, 4, 5, 6, 7]
pc = [10, 20, 30, 40, 52]

for i in pc:
    fft_pca_svm.run(n_modes=4, fault_prop=0.5, pcs=i * 100, filename='fft_pca_svm_4_pcs_' + str(i))
