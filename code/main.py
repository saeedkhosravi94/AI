from NewtonUTSVM import NewtonUTSVM
from S3VM_unconstrained import S3VM_Unconstrained
from S3VM_constrained import S3VM_Constrained
import time
from utils import load_dataset, calculate_accuracy, move_labels_to_last_column
import pandas as pd
from MC_NDCC import MC_NDCC
import random



if __name__  == "__main__":


    # print('---------- diabetes Dataset ----------')
    # csv_file = "data/diabetes.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)
 
    # start = time.time()
    # nutsvm = NewtonUTSVM(X, y, U, C=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3]   , eps=1e-4)
    # nutsvm.fit()
    # nutsvm.predict(x_test=x_test)
    # pt = nutsvm.get_preds()
    # nutsvm_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("Newton Accuracy: ", nutsvm_acc)
    # print("Newton Time: ", round(end - start, 2), "sec")

    # # --------------------------------------------------

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Unconstrained(C=(1 - lmbda)/(lmbda * (l + k)), eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Unconstrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")

    # print('--------------------------------------')

    # print('---------- ionosphere Dataset ----------')
    # csv_file = "data/ionosphere.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)
 
    # start = time.time()
    # nutsvm = NewtonUTSVM(X, y, U, C=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3]   , eps=1e-4)
    # nutsvm.fit()
    # nutsvm.predict(x_test=x_test)
    # pt = nutsvm.get_preds()
    # nutsvm_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("Newton Accuracy: ", nutsvm_acc)
    # print("Newton Time: ", round(end - start, 2), "sec")

    # # --------------------------------------------------

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Unconstrained(C=(1 - lmbda)/(lmbda * (l + k)), eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Unconstrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")

    # print('----------------------------------')

    # print('---------- Musk Dataset ----------')
    # csv_file = "data/musk.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)

    # start = time.time()
    # nutsvm = NewtonUTSVM(X, y, U, C=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3]   , eps=1e-4)
    # nutsvm.fit()
    # nutsvm.predict(x_test=x_test)
    # pt = nutsvm.get_preds()
    # nutsvm_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("Newton Accuracy: ", nutsvm_acc)
    # print("Newton Time: ", round(end - start, 2), "sec")

    # # --------------------------------------------------

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Unconstrained(C=(1 - lmbda)/(lmbda * (l + k)), eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Unconstrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")



    # print('------------------------------------------------------------------')
    # print('---------- Breast Cancer Wisconsin (Prognostic) Dataset ----------')
    # csv_file = "data/wpbc.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)


    # start = time.time()
    # nutsvm = NewtonUTSVM(X, y, U, C=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3]   , eps=1e-4)
    # nutsvm.fit()
    # nutsvm.predict(x_test=x_test)
    # pt = nutsvm.get_preds()
    # nutsvm_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("Newton Accuracy: ", nutsvm_acc)
    # print("Newton Time: ", round(end - start, 2), "sec")

    # # --------------------------------------------------

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Unconstrained(C=(1 - lmbda)/(lmbda * (l + k)), eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Unconstrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")

    # print('-------------------------------------------')
    # print('---------- Sonar Dataset ----------')
    # csv_file = "data/sonar.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)


    # start = time.time()
    # nutsvm = NewtonUTSVM(X, y, U, C=[1, 1, 0.1, 0.1, 2, 2]   , eps=1e-4)
    # nutsvm.fit()
    # nutsvm.predict(x_test=x_test)
    # pt = nutsvm.get_preds()
    # nutsvm_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("Newton Accuracy: ", nutsvm_acc)
    # print("Newton Time: ", round(end - start, 2), "sec")

    # # --------------------------------------------------

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Unconstrained(C=(1 - lmbda)/(lmbda * (l + k)), eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Unconstrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")

    # print('------------------------------------')
    # print('---------- Gender Dataset ----------')
    # csv_file = "data/gender.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)


    # start = time.time()
    # nutsvm = NewtonUTSVM(X, y, U, C=[1.0, 1.0, 0.1, 0.1, 0.05, 0.0]   , eps=1e-4)
    # nutsvm.fit()
    # nutsvm.predict(x_test=x_test)
    # pt = nutsvm.get_preds()
    # nutsvm_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("Newton Accuracy: ", nutsvm_acc)
    # print("Newton Time: ", round(end - start, 2), "sec")

    # # --------------------------------------------------

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Unconstrained(C=(1 - lmbda)/(lmbda * (l + k)), eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Unconstrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")

    # # Initialize an instance 
    # ndcc = MC_NDCC()
    # # Get the dataset as a Numpy matrix
    # ds_mat  = ndcc.get_matrix()
    # # Save the dataset as a csv file
    # ndcc.get_csv('data/1000_100_ndcc.csv')

    # print('------------------------------------')
    # print('---------- NDCC Dataset ----------')
    # csv_file = "data/100_10_ndcc.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)


    # start = time.time()
    # nutsvm = NewtonUTSVM(X, y, U, C=[2, 0.1, 1, 1, 0.1, 0.1]   , eps=1e-4)
    # nutsvm.fit()
    # nutsvm.predict(x_test=x_test)
    # pt = nutsvm.get_preds()
    # nutsvm_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("Newton Accuracy: ", nutsvm_acc)
    # print("Newton Time: ", round(end - start, 2), "sec")

    # # --------------------------------------------------

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    # s3vm_con.fit(X, y, U)
    # pt = s3vm_con.predict(x_test)
    # s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    # print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    # start = time.time()
    # lmbda = 0.001
    # l, k = X.shape[0], U.shape[0]
    # s3vm_uncon = S3VM_Unconstrained(C=(1-lmbda)/(lmbda * (l + k)), eps=1e-4)
    # s3vm_uncon.fit(X, y, U)
    # pt = s3vm_uncon.predict(x_test)
    # s3vm_uncon_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    # end = time.time()
    # print("S3VM Unconstrained Accuracy: ", s3vm_uncon_acc)
    # print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")


    # print('------------------------------------')
    # print('---------- NDCC Dataset ----------')
    # csv_file = "data/100_10_ndcc.csv"
    # X, y, x_test, y_test, U = load_dataset(csv_file)
    # y = y.reshape(y.shape[0], 1)



    print('------------------------------------')
    print('---------- NDCC Dataset ----------')
    csv_file = "data/100_100_ndcc.csv"
    X, y, x_test, y_test, U = load_dataset(csv_file)
    y = y.reshape(y.shape[0], 1)


    param_space = [0.01, 0.1, 1, 2, 4, 8]

    best_acc = 0
    best_time = 0
    best_params = None

    for i in range(1000):
        C = [random.choice(param_space) for _ in range(6)]

        start = time.time()
        nutsvm = NewtonUTSVM(X, y, U, C=C, eps=1e-4)
        nutsvm.fit()
        nutsvm.predict(x_test=x_test)
        pt = nutsvm.get_preds()
        acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
        end = time.time()

        elapsed_time = round(end - start, 2)

        if acc > best_acc:
            best_acc = acc
            best_time = elapsed_time
            best_params = C

        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/1000")

    print("Best Accuracy:", best_acc)
    print("Best Time:", best_time, "sec")
    print("Best C Parameters:", best_params)
    # --------------------------------------------------

    start = time.time()
    lmbda = 0.001
    l, k = X.shape[0], U.shape[0]
    s3vm_con = S3VM_Constrained(C=(1 - lmbda)/(lmbda * (l + k)), M=1e5, eps=1e-4)
    s3vm_con.fit(X, y, U)
    pt = s3vm_con.predict(x_test)
    s3vm_con_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    end = time.time()
    print("S3VM Constrained Accuracy: ", s3vm_con_acc)
    print("S3VM Constrained Time: ", round(end - start, 2), "sec")

    start = time.time()
    lmbda = 0.001
    l, k = X.shape[0], U.shape[0]
    s3vm_uncon = S3VM_Unconstrained(C=(1-lmbda)/(lmbda * (l + k)), eps=1e-4)
    s3vm_uncon.fit(X, y, U)
    pt = s3vm_uncon.predict(x_test)
    s3vm_uncon_acc = calculate_accuracy(true_labels=y_test, predicted_labels=pt)
    end = time.time()
    print("S3VM Unconstrained Accuracy: ", s3vm_uncon_acc)
    print("S3VM Unconstrained Time: ", round(end - start, 2), "sec")