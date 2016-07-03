"""
k nearest neighbors module.
"""
import os
import numpy as np


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(x, dataset, labels, k):

    # Euclidian distance
    distance = ((np.tile(x, (dataset.shape[0], 1)) - dataset)**2).sum(axis=1)
    distance = distance ** 0.5
    sorted_dist_idx = distance.argsort()

    class_count = {}
    for i in range(k):
        vote_lable = labels[sorted_dist_idx[i]]
        class_count[vote_lable] = class_count.get(vote_lable, 0) + 1

    # It's a 2D list of tuple after sorting
    sorted_class_count = sorted(class_count.items(), 
                                key = lambda x:x[1], reverse=True)

    return sorted_class_count[0][0]


def file2matrix(filename):
    
    return_mat, labels = [], []
    with open(filename) as fr:
        for line in fr:
            list_from_line = line.strip('\n').split()
            return_mat.append(map(float, list_from_line[:3]))
            labels.append(int(list_from_line[-1]))

    return np.array(return_mat), np.array(labels)


def autoNorm(dataset):
    min_val = dataset.min(axis=0)
    max_val = dataset.max(axis=0)

    range_val = max_val - min_val
    m = dataset.shape[0]  # The number of row
    norm_dataset = (dataset - np.tile(min_val, (m,1))) / np.tile(range_val, (m,1))

    return norm_dataset, range_val, min_val


def datingClassTest():
    ho_ratio = 0.2
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, range_val, min_val = autoNorm(dating_data_mat)

    m = norm_mat.shape[0]
    num_test_vec = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vec):
        classifier_result = classify0(norm_mat[i], norm_mat[num_test_vec:m],
                                      dating_labels[num_test_vec:m], 3)
        print ("The classifier came back with: %d, the real answer is: %d"
               % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]: error_count += 1

    print "The total error rate is: %f" % (error_count/float(num_test_vec))
    

def classifyPerson():
    result_list = ['Not at all', 'In small does', 'In large doses']
    percent_tats = float(raw_input('Percentage of time spent playing video games?'))
    ffmiles = float(raw_input('Frequent flier miles earned per year?'))
    ice_cream = float(raw_input('liters of ice cream consumed per year?'))
    
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, range_val, min_val = autoNorm(dating_data_mat)

    x = (np.array([ffmiles, percent_tats, ice_cream]) - min_val) / range_val
    predict = classify0(x, norm_mat, dating_labels, 3)

    print 'You will probably like this person: ', result_list[predict - 1]
    

def img2vector(filename):
    """
    read the handwriting image
    """
    return_vec = []
    with open(filename) as fr:
        for line in fr:
            return_vec.extend(map(int, line.strip()))

    return return_vec


def handwritingClassTest():

    tr_directory = 'digits/trainingDigits'
    te_directory = 'digits/testDigits'

    training_mat, hw_labels = [], []
    training_file_list = os.listdir(tr_directory)
    for _, filename in enumerate(training_file_list):
        hw_labels.append(int(filename.split('.')[0].split('_')[0]))  # 0_13.txt
        training_mat.append(img2vector(tr_directory+'/'+filename))

    training_mat = np.array(training_mat)
    
    error_count = 0.0
    test_file_list = os.listdir(te_directory)
    for _, filename in enumerate(test_file_list):
        hw_num = int(filename.split('.')[0].split('_')[0])  # 0_13.txt
        num_vec = img2vector(te_directory+'/'+filename)
        classifier_num = classify0(num_vec, training_mat, hw_labels, 3)
    
#print ('The classifier came back with: %d, the real answer is: %d'
#% (classifier_num, hw_num))
        if classifier_num != hw_num: error_count += 1.0
        
    print '\nThe total number of errors is: %d' % error_count
    print '\nThe total error rate is: %f' % (error_count / len(test_file_list))





