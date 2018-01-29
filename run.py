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

# ANN_without_fft_pca.run(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='ANN', batchsize=500)

a = [1, 2, 3, 4, 5, 6, 7]
pc = [10, 20, 30, 40, 52]
n = [1, 2, 3, 4, 5]


plain_ann.run(filename='timmin_ann_1')
plain_svm.run(filename='timmin_svm_1')

for j in n:
    plain_knn.run(filename='timmin_knn_1', neighbors=j)

for i in pc:
    pca_svm.run(pcs=i, filename='timmin_pca_svm__pcs_' + str(i))
for i in pc:
    fft_pca_svm.run(pcs=i, filename='timmin_fft_pca_svm__pcs_' + str(i))
for i in pc:
    pca_ann.run(pcs=i, filename='timmin_pca_ann_' + str(i))
for i in pc:
    fft_pca_ann.run(pcs=i, filename='timmin_fft_pca_ann' + str(i))

for i in pc:
    for j in n:
        pca_knn.run(pcs=i, filename='timmin_pca_knn' + str(i), neighbors=j)

for i in pc:
    for j in n:
        fft_pca_knn.run(pcs=i, filename='timmin_fft_pca_knn' + str(i), neighbors=j)





'''
for i in pc:

    pca_svm.run(n_modes=4, fault_prop=0.5, pcs=i, repetitions=30, filename='svm_4_pcs_' + str(i))




fft_pca_ann.run(n_modes=1, fault_prop=0.5, pcs=5200, repetitions=1, filename='fftpca_ann_test',
                            batchsize=512)
pca_ann.run(n_modes=4, fault_prop=.5, pcs=30, repetitions=10, filename='PCA-ANN_' + 'PCs_4modes' + str(30),
                    batchsize=512)
pca_ann.run(n_modes=4, fault_prop=.5, pcs=40, repetitions=10, filename='PCA-ANN_' + 'PCs_4modes' + str(40),
                    batchsize=512)
pca_ann.run(n_modes=4, fault_prop=.5, pcs=52, repetitions=7, filename='PCA-ANN_' + 'PCs_4modes' + str(52),
                    batchsize=512)
for i in a:
    plain_svm.run(n_modes=1, fault_prop=0.5, repetitions=30, filename='svm_' + str(i) + '.csv')
    print('svm test nmodes: {0} completed'.format(str(i)))


KNN.run(n_modes=7, fault_prop=.5, repetitions=1, filename='10foldKNN_'+str(7), neighbors=5)
KNN.run(n_modes=7, fault_prop=.5, repetitions=1, filename='10foldKNN_'+str(7), neighbors=4)
KNN.run(n_modes=7, fault_prop=.5, repetitions=1, filename='10foldKNN_'+str(7), neighbors=3)
'''
