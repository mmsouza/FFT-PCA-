import FFT_PCA_BINARY
import PCA_KNN
import pca_ann_bin
import ANN_without_fft_pca
import KNN
import FFT_PCA_KNN
import matplotlib.pyplot as plt
import svm
import fft_pca_svm
import pca_svm

# ANN_without_fft_pca.ann_run(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='ANN', batchsize=500)

a = [1, 2, 3, 4, 5, 6, 7]
pc = [10, 20, 30, 40, 52]
'''
for j in pc:

        FFT_PCA_BINARY.pca_fft_ann(n_modes=4, fault_prop=.5, pcs=j*100, repetitions=10, filename='FFT-PCA-ANN_4modes'+'PCs_'+str(j), batchsize=512)
        print(str(j) + " pca-fft ann finished")
        #FFT_PCA_KNN.pca_fft_knn(n_modes=4, fault_prop=.5, pcs=j*100, repetitions=30, filename='FFT-PCA-kNN_4modes'+'PCs_'+str(j), neighbors=5)
        #print(str(j) + 'pca-fft-knn finished')
        pca_ann_bin.pca_ann(n_modes=4, fault_prop=.5, pcs=j, repetitions=15, filename='PCA-ANN_'+'PCs_4modes'+str(j), batchsize=512)
        print(str(j) + ' pca-ann-fineshed')
        #PCA_KNN.pca_KNN(n_modes=4, fault_prop=.5,pcs=j,repetitions=30, filename='PCA-KNN_'+'PCs_4modes'+str(j), neighbors=5)
        #print(str(j) + ' pca-KNN-fineshed')



for i in a:
  print(i)
  #  KNN.run_KNN(n_modes=i, fault_prop=.5, repetitions=30, filename='KNN_'+str(i), neighbors=5)
  ANN_without_fft_pca.ann_run(n_modes=i, fault_prop=.5, repetitions=10, filename='ANN_'+str(i), batchsize=512)
'''

n = [5, 6, 7, 8, 9]
#for j in pc:
    # KNN.run_KNN(n_modes=7, fault_prop=.5, repetitions=1, filename='KNN_neigtest', neighbors=j)
    # svm.run_svm(n_modes=j, fault_prop=.5, repetitions=1, filename='svm_' + str(j))
    #fft_pca_svm.run_fftpca_svm(n_modes=7, filename='fft_pca_svm_7', pcs=j * 100)
    #pca_svm.run_pca_svm(n_modes=7, filename='pca_svm_7', pcs=j)

KNN.run_KNN(n_modes=7, fault_prop=.5, repetitions=30, filename='10foldKNN_'+str(7), neighbors=5)