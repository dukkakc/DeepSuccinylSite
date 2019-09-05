import keras
import numpy as np
from Bio import SeqIO
from numpy import array
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, classification_report
from keras.models import load_model


r_test_x = []
r_test_y = []
posit_1 = 1;
negat_0 = 0;
win_size = 33 # actual window size
win_size_kernel = int(win_size/2 + 1)


# define universe of possible input values
alphabet = 'ARNDCQEGHILKMFPSTWYV-'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))



#-------------------------TEST DATASET----------------------------------------
#for positive sequence
def innertest1():
    #Input
    data = seq_record.seq
    #rint(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(posit_1)
for seq_record in SeqIO.parse("test_positive_sites.fasta", "fasta"):
    innertest1()
#for negative sequence
def innertest2():
    #Input
    data = seq_record.seq
    #print(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(negat_0)
for seq_record in SeqIO.parse("test_negative_sites.fasta", "fasta"):
    innertest2()
# Changing to array (matrix)    
r_test_x = array(r_test_x)
r_test_y = array(r_test_y)


# Balancing test dataset
# Testing Data Balancing by undersampling####################################
rus = RandomUnderSampler(random_state=7)
x_res3, y_res3 = rus.fit_resample(r_test_x, r_test_y)
#Shuffling
r_test_x, r_test_y = shuffle(x_res3, y_res3, random_state=7)
r_test_x = np.array(r_test_x)
r_test_y = np.array(r_test_y)
############################################################################


##LOAD MODEL####
model = load_model('model.h5')
#print("This is final ",model.layers[0].get_weights()[0][16])
r_test_y_2 = keras.utils.to_categorical(r_test_y, 2)
score = model.evaluate(r_test_x, r_test_y_2, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
Y_pred = model.predict(r_test_x)
Y_pred = (Y_pred > 0.5)
y_pred1 = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred1 = np.array(y_pred1)

print("Matthews Correlation : ",matthews_corrcoef(r_test_y, y_pred1))
print("Confusion Matrix : \n",confusion_matrix(r_test_y, y_pred1))
# ROC

fpr, tpr, _ = roc_curve(r_test_y, y_pred1)
roc_auc = auc(fpr, tpr)
print("AUC : ", roc_auc)
print(classification_report(r_test_y, y_pred1))

print(model.summary())


