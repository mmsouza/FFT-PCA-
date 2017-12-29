import FFT_PCA_BINARY
import PCA_KNN
import pca_ann_bin
import ANN_without_fft_pca
import KNN
import FFT_PCA_KNN
import matplotlib.pyplot as plt


#ANN_without_fft_pca.ann_run(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='ANN', batchsize=500)

a = [1,2,3,4,5,6,7]
pc=[10,20,30,40,52]

for j in pc:
        print(i)
        FFT_PCA_BINARY.pca_fft_ann(n_modes=4, fault_prop=.5, pcs=j*100, repetitions=30, filename='FFT-PCA-ANN_4modes'+'PCs_'+str(j), batchsize=512)
        FFT_PCA_KNN.pca_fft_knn(n_modes=4, fault_prop=.5, pcs=j*100, repetitions=30, filename='FFT-PCA-kNN_4modes'+'PCs_'+str(j), neighbors=5)

        pca_ann_bin.pca_ann(n_modes=4, fault_prop=.5, pcs=j, repetitions=30, filename='PCA-ANN_'+'PCs_4modes'+str(j), batchsize=512)
        PCA_KNN.pca_KNN(n_modes=4, fault_prop=.5,pcs=j,repetitions=30, filename='PCA-KNN_'+'PCs_4modes'+str(j), neighbors=5)




#for i in a:
 #   print(i)
  #  KNN.run_KNN(n_modes=i, fault_prop=.5, repetitions=30, filename='KNN_'+str(i), neighbors=5)
   # ANN_without_fft_pca.ann_run(n_modes=i, fault_prop=.5, repetitions=30, filename='ANN_'+str(i), batchsize=512)