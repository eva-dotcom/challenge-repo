import re
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def load_reviews_from_file(location):
    reviews = []
    for line in open(location, 'r'):
        reviews.append(line.strip())    
    
    return reviews


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews



def run():
    reviews_pos = load_reviews_from_file("kiapp/data/pos.txt")
    reviews_neg = load_reviews_from_file("kiapp/data/neg.txt")
    # 2400 reviews for training 100 for testing (pos&neg)
    reviews_pos_train = reviews_pos[:2400]
    reviews_neg_train = reviews_neg[:2400]
    # 4800 train reviews
    reviews_train = reviews_pos_train + reviews_neg_train # Combine pos & neg reviews
    # 200 test reviews
    reviews_test = reviews_pos[-100:] + reviews_neg[-100:] # Combine pos & neg reviews
    
    # Clean reviews
    reviews_train_clean = preprocess_reviews(reviews_train)
    reviews_test_clean = preprocess_reviews(reviews_test)

    # Vectorization
    # Matrix Columns represent words, Rows represent reviews
    # Binary indication if word is in review 
    stop_words = ['the', 'a', 'in', 'of', 'at']
    cv = CountVectorizer(binary=True, ngram_range=(1,2), stop_words=stop_words)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)
    # Positive review y=1 Negative Review y=0
    # First 2400 positive, other negative
    target = [1 if i < 2400 else 0 for i in range(4800)]
    
    # This is for finding the right c value for regularization
     X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)
    # Find optimum c
     for c in [0.01, 0.05, 0.25, 0.5, 1]:

         svm = LinearSVC(C=c)
         svm.fit(X_train, y_train)
         print ("Accuracy for C=%s: %s" 
             % (c, accuracy_score(y_val, lr.predict(X_val))))


if __name__ == '__main__':
    run()