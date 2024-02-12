'''
NLP Using Naive baye's classifier -  Group 4'
'''
'''
As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results. You can be very creative and even do more.
Present all the results and conclusions.
Drop code, report and power point presentation into the project assessment folder for grading.
'''

'''
STEP1-Load the data into a pandas data frame.
'''
import pandas as pd
df = pd.read_csv(r'C:\Users\ariya\Downloads\Youtube01-Psy.csv')
df
'''
STEP 2 i-Initial exploration  
Carry out some basic data exploration and present your results. (Note: You only need two columns for this project, make sure you identify them correctly, if any doubts ask your professor)
'''
df.head(3)
print(df.shape)
print(df.columns.values)
print(df.dtypes)
print(df.isnull().sum())
# print the unique counts
print(df.nunique())

'''
STEP 2 ii-from initial exploration we see comment_id, Author, date mostly has unique values which will not contribute much to our analysis.
we'll choose CONTENT, CLASS for analysis...'
'''
df = df.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1)

print(df.head(3));

'''
STEP 3-prepare the data
i- stop words
ii-lower case
iii-lemmatize
iv-Isalphabet
'''

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
   # lemmatized_text = [lemmatizer.lemmatize(filtered_tokens) for filtered_tokens in filtered_tokens]
    return " ".join(filtered_tokens)

df['CONTENT'] = df['CONTENT'].apply(preprocess_text)

#BOW--> VECTOR ---> MATRIX --> TRAINED ---> 

'''
Using nltk toolkit classes and methods prepare the data for model building, refer to the third lab tutorial in module 11 (Building a Category text predictor ). Use count_vectorizer.fit_transform().

STEP -3 COUNTER VECTORIZATION (BOW)
'''
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(df['CONTENT'])

'''STEP -4 
Present highlights of the output (initial features) such as the new shape of the data and any other useful information before proceeding.'''
print("sparse Matrix",train_tc.toarray())
print("\nDimensions of training data:", train_tc.shape)
num_unique_words = len(count_vectorizer.vocabulary_)
print("Number of unique words: ", num_unique_words)
non_zero_values = train_tc.data
sparse_matrix_stats = pd.Series(non_zero_values).describe()
print("Summary statistics of the sparse matrix:\n", sparse_matrix_stats)
cols = count_vectorizer.get_feature_names_out()
cols
#feature & count
bow = pd.DataFrame(train_tc.toarray(),columns=cols)
bow.head(5)
'''
STEP-5 Downscale the transformed data using tf-idf and again present highlights of the output (final features) such as the new shape of the data and any other useful information before proceeding.
'''
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
print("\nDimensions of training data:", train_tfidf.shape)
type(train_tfidf)
train_tfidf.toarray()

non_zero_values = train_tfidf.data
sparse_matrix_stats = pd.Series(non_zero_values).describe()
print("Summary statistics of the sparse matrix:\n", sparse_matrix_stats)

'''
STEP-6
Use pandas.sample to shuffle the dataset, set frac =1 '''
df = df.sample(frac=1,random_state=1).reset_index(drop=True)
print(df)

'''
STEP-7
Using pandas split your dataset into 75% for training and 25% for testing,
 make sure to separate the class from the feature(s). (Do not use test_train_ split)
'''
train_size = int(len(df) * 0.75) #75% traning remaning will be testing data set
train_df = df[:train_size]
test_df = df[train_size:]
X_train = tfidf.fit_transform(count_vectorizer.fit_transform(train_df['CONTENT']))
y_train = train_df['CLASS']
X_test = tfidf.transform(count_vectorizer.transform(test_df['CONTENT']))
y_test = test_df['CLASS']

'''
STEP-8
Fit the training data into a Naive Bayes classifier. 
'''
from sklearn.naive_bayes import MultinomialNB
# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

'''
STEP-9
Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb_classifier, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy: ", scores.mean())

'''
STEP-10
Test the model on the test data, print the confusion matrix and the accuracy of the model.
'''
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = nb_classifier.predict(X_test)
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("Accuracy:- ", accuracy_score(y_test, y_pred))

'''
STEP-11
As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results. You can be very creative and even do more
'''

category_map = {0: 'Ham', 1: 'Spam'}
# Define test data 
input_data = [
    'lame jokes dont watch',
    'please subrsribbb to my youtubbb channel',
    'Hi guys, Im offering free coupon of 500$ please click here',
    'Dont watch this video',
    'NICE VIDEO',
    'Super content, fantastic job',
    'Wow, i like this video',
    'omg, 5 million views',
    'Win Prize😀'
    
]

# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)
type(input_tc)
print(input_tc.toarray())
# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
print(input_tfidf.toarray())
# Predict the output categories
predictions = nb_classifier.predict(input_tfidf)

# Print the outputs
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', 
            category_map[category])