import pca_ann
import pca_knn
import pca_ann
import plain_ann
import plain_knn
import pca_knn
import matplotlib.pyplot as plt
import plain_svm
import pca_svm
import pca_svm

a = [1, 2, 3, 4, 5, 6, 7]
pc = [10, 20, 30, 40, 52]

for i in pc:
    pca_ann.run(n_modes=1, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_ann_m-1_pc-' + str(i),
                batchsize=512)

for i in pc:
    pca_ann.run(n_modes=2, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_ann_m-2_pc-' + str(i),
                batchsize=512)
for i in pc:
    pca_ann.run(n_modes=3, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_ann_m-3_pc-' + str(i),
                batchsize=512)
for i in pc:
    pca_ann.run(n_modes=5, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_ann_m-5_pc-' + str(i),
                batchsize=512)
for i in pc:
    pca_ann.run(n_modes=6, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_ann_m-6_pc-' + str(i),
                batchsize=512)
for i in pc:
    pca_ann.run(n_modes=7, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_ann_m-7_pc-' + str(i),
                batchsize=512)

# --------------------fft-pca-svm----------------------------------------------------------------------------------
for i in pc:
    pca_svm.run(n_modes=1, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_svm_m-1_pc-' + str(i))

for i in pc:
    pca_svm.run(n_modes=2, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_svm_m-2_pc-' + str(i))

for i in pc:
    pca_svm.run(n_modes=3, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_svm_m-3_pc-' + str(i))

for i in pc:
    pca_svm.run(n_modes=5, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_svm_m-5_pc-' + str(i))

for i in pc:
    pca_svm.run(n_modes=6, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_svm_m-6_pc-' + str(i))

for i in pc:
    pca_svm.run(n_modes=7, fault_prop=0.5, pcs=i, repetitions=40, filename='pca_svm_m-7_pc-' + str(i))

    # ---------------fft-pca-Knn---------------------------------------------------------------------------------    

    for i in pc:
        pca_knn.run(n_modes=1, fault_prop=0.5, pcs=i, repetitions=40,
                    filename='pca_knn_m-1_pc-' + str(i))

    for i in pc:
        pca_knn.run(n_modes=2, fault_prop=0.5, pcs=i, repetitions=40,
                    filename='pca_knn_m-2_pc-' + str(i))

    for i in pc:
        pca_knn.run(n_modes=3, fault_prop=0.5, pcs=i, repetitions=40,
                    filename='pca_knn_m-3_pc-' + str(i))

    for i in pc:
        pca_knn.run(n_modes=5, fault_prop=0.5, pcs=i, repetitions=40,
                    filename='pca_knn_m-5_pc-' + str(i))

    for i in pc:
        pca_knn.run(n_modes=6, fault_prop=0.5, pcs=i, repetitions=40,
                    filename='pca_knn_m-6_pc-' + str(i))

    for i in pc:
        pca_knn.run(n_modes=7, fault_prop=0.5, pcs=i, repetitions=40,
                    filename='pca_knn_m-7_pc-' + str(i))
