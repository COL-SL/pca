# ! /usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab
from matplotlib.transforms import Affine2D
from pylab import axes
import time
np.set_printoptions(threshold=np.nan)


def load_pca():
    initial_time = time.time()
    PATH_INITIAL = sys.argv[1]
    NUMBER_ROWS = int(sys.argv[2])
    NUMBER_COLUMNS = int(sys.argv[3])
    NUMBER_EIGENVECTORES = int(sys.argv[4])
    np.set_printoptions(suppress=True)

    number_columns_loop = 0
    number_rows_loop = 0
    sum_eigenvalues = 0
    total = 0
    exit_loop = 0
    file_initial = open('initial.txt', 'w')
    file_reconstruction = open('reconstruction.txt', 'w')

    Matrix = np.empty((NUMBER_ROWS,NUMBER_COLUMNS,))
    Matrix_subtraed = np.empty((NUMBER_ROWS,NUMBER_COLUMNS,))
    Matrix_original = np.empty((NUMBER_ROWS,NUMBER_COLUMNS,))
    Matrix_initial = np.empty((NUMBER_ROWS,NUMBER_COLUMNS,))

    print '\n', "Calculando datos ...", '\n'

    count_num_data = 0
    count_row = 1
    initial_file = np.loadtxt(PATH_INITIAL,  usecols=(0, ))
    list_original_data = []
    original_data = np.array(initial_file)

    for num in original_data:
        count_num_data = count_num_data + 1
        list_original_data.append(num)
        if count_num_data == int(NUMBER_ROWS):
            count_num_data = 0
            count_row =  count_row + 1
            list_original_data.append("|")

    list_original_data = str(list_original_data)
    list_original_data = list_original_data.replace(',', '')
    list_original_data = list_original_data.replace('\'', '')
    list_original_data = list_original_data.replace('[', '')
    list_original_data = list_original_data.replace(']', '')
    list_original_data = list_original_data.replace('"', '')
    list_original_data = list_original_data.split("|")
    list_original_data = str(list_original_data).replace('[', '')
    list_original_data = str(list_original_data).replace(']', '')
    list_original_data = str(list_original_data).replace('\'', '')
    list_original_data = str(list_original_data).replace(',', '')
    list_original_data = str(list_original_data).replace('\'\'','')
    list_original_data = str(list_original_data).split(" ")

    for i in range(0,len(list_original_data)):
        if 0  < len(list_original_data[i]):
            if number_rows_loop == NUMBER_ROWS:
                number_rows_loop = 0
            if number_columns_loop == NUMBER_COLUMNS:
                number_columns_loop = 0
            if number_columns_loop == 0 and number_rows_loop == 0:
                exit_loop =  exit_loop + 1
            if exit_loop < 2:
                #print "number_rows_loop: ",number_rows_loop
                #print "number_columns_loop: ",number_columns_loop
                #print list_original_data[i]
                Matrix[int(number_rows_loop)][int(number_columns_loop)]= float(list_original_data[i])
                #print float(list_original_data[i])
                number_rows_loop = number_rows_loop + 1
        else:
            number_columns_loop = number_columns_loop + 0.5
            number_rows_loop = 0

    Matrix = np.array(Matrix)
    Matrix_initial = np.array(Matrix)

    for i in Matrix:
        #print i
        file_initial.write(str(i))
        file_initial.write('\n')
    file_initial.close()

    '''
    count = 1
    for index in range(0, NUMBER_COLUMNS):
        for i in Matrix_initial[:,index]:
            plt.plot(i, count, marker='+', linestyle=' ', color='C0')
            count = count + 1
        count = 1

    plt.gca().invert_yaxis()
    plt.title('Initial Data')
    plt.legend()
    plt.show()
    '''
    plt.plot(Matrix_initial, marker='+', linestyle=' ')
    plt.title('Initial Data')
    plt.legend()
    plt.show()

    for i in range(0, int(NUMBER_COLUMNS)):
        mean = np.average(Matrix[:, i])
        corrected = [value - mean for value in (Matrix[:, i])]
        Matrix[:, i] =  np.array(corrected)

    print '\n',"Calculando Matriz de Covarianza ...",'\n'
    Matrix_subtraed = Matrix
    Matrix = np.array(Matrix)
    covData = np.cov(Matrix)
    #print covData

    eigenvalues, eigenvectors = np.linalg.eig(covData)

    j = eigenvalues.argsort()
    eigenvalues[j]

    print '\n',"Calculando Valores propios ...",'\n'

    for i in range(0,len(eigenvalues)):
        sum_eigenvalues = sum_eigenvalues + eigenvalues[i]

    print '\n','Valores propios selecccionados:'
    for i in range(0,NUMBER_EIGENVECTORES):
        aux = float(eigenvalues[i]).real
        aux_2 = (float(total) + float(eigenvalues[i]/sum_eigenvalues)).real
        print i + 1,':', aux
        total = aux_2
        print total, '\n'

    matrix_eigenvectors = eigenvectors
    matrix_eigenvectors = np.array(matrix_eigenvectors).real

    matrix_eigenvectors = np.delete(matrix_eigenvectors, np.s_[NUMBER_EIGENVECTORES:], 1)

    matrix_eigenvectors = matrix_eigenvectors.T
    data_zip = matrix_eigenvectors.dot(Matrix_subtraed)

    Matrix_original = matrix_eigenvectors.T.dot(data_zip)

    for i in range(0, int(NUMBER_COLUMNS)):
        mean = np.average(Matrix_initial[:, i])
        corrected = [value + mean for value in (Matrix_original[:, i])]
        Matrix_original[:, i] =  np.array(corrected)

    '''
    count = 1
    for index in range(0, NUMBER_COLUMNS):
        for i in Matrix_original[:, index]:
            plt.plot(i, count, marker='+', linestyle=' ', color='C0')
            count = count + 1
        count = 1

    plt.gca().invert_yaxis()
    plt.title('Reconstruction Data')
    plt.legend()
    plt.show()
    '''
    plt.plot(Matrix_original, marker='+', linestyle=' ')
    plt.title('Reconstruction Data')
    plt.legend()
    plt.show()

    for i in Matrix_original:
        #print i
        file_reconstruction.write(str(i))
        file_reconstruction.write('\n')
    file_reconstruction.close()

    final_time = time.time()
    total_time = final_time - initial_time
    print '\n\n', "Execution Time: %f seconds" % (total_time)

load_pca()