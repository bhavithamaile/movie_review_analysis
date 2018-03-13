import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

def make_Corpus(root_dir):
    polarity_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    corpus = []    
    for polarity_dir in polarity_dirs:
        reviews = [os.path.join(polarity_dir,f) for f in os.listdir(polarity_dir)]
        for review in reviews:
            doc_string = "";
            with open(review) as rev:
                for line in rev:
                    doc_string = doc_string + line
            if not corpus:
                corpus = [doc_string]
            else:
                corpus.append(doc_string)
    return corpus

#Create a corpus with each document having one string
root_dir = 'F:\\My_Projects\\Review_analysis\\review_polarity\\txt_sentoken'
corpus = make_Corpus(root_dir)
#print (corpus);

#Stratified 10-cross fold validation with SVM and Multinomial NB 
labels = np.zeros(2000);
labels[0:1000]=0;
labels[1000:2000]=1; 
      
kf = StratifiedKFold(n_splits=10)#splits reviews into 10 folds

totalsvm = 0           # Accuracy measure on 2000 files
totalNB = 0
totalMatSvm = np.zeros((2,2));  # Confusion matrix on 2000 files
totalMatNB = np.zeros((2,2));
#train_index and test_index contains numbers incremented one each time
for train_index, test_index in kf.split(corpus,labels):#train_index contains 1800 and test_index contains 200....split divides review into 10 folds
    X_train = [corpus[i] for i in train_index]#x_train contains all train reviews
    X_test = [corpus[i] for i in test_index]
    y_train, y_test = labels[train_index], labels[test_index]#tfi is a class
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.7, sublinear_tf=True, use_idf=True,stop_words='english')#creates a vocabulary
    #print(vectorizer)
    train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
    test_corpus_tf_idf = vectorizer.transform(X_test)#generate term document matrix for test set
   # print(train_corpus_tf_idf)
    #print(test_corpus_tf_idf)
    #print(vectorizer.vocabulary_)
    model1 = LinearSVC()#kernel means similarity bw test n train test
    model2 = MultinomialNB()    
    model1.fit(train_corpus_tf_idf,y_train)#it fits linear model....linear model assumes relationship bw x n y
    model2.fit(train_corpus_tf_idf,y_train)
    result1 = model1.predict(test_corpus_tf_idf)
    result2 = model2.predict(test_corpus_tf_idf)
 #  // print(result1)
#print(result2)
    
    totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1)#y_test contains correct values n result1 contains predicted values
    totalMatNB = totalMatNB + confusion_matrix(y_test, result2)
    totalsvm = totalsvm+sum(y_test==result1)
    totalNB = totalNB+sum(y_test==result2)
    
print ('confusion Matrix:')
print(totalMatSvm)
print ('Accuracy:',totalsvm/2000.0)#, totalMatNB, totalNB/2000.0)
print('Error Rate:',(totalMatSvm[0][1]+totalMatSvm[1][0])/2000)
print('Sensitivity:',totalMatSvm[0][0]/(totalMatSvm[0][0]+totalMatSvm[0][1]))
print('Specificity:',totalMatSvm[1][1]/(totalMatSvm[1][1]+totalMatSvm[1][0]))
print('Precision:',totalMatSvm[0][0]/(totalMatSvm[0][0]+totalMatSvm[1][0]))
print('False positive rate',totalMatSvm[1][0]/(totalMatSvm[1][0]+totalMatSvm[1][1]))
