import csv
from kiapp.inc import load_reviews_from_file, preprocess_reviews

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load

class Ki:
    """ This class holds all function to create a KI Model & evaluate data based on the trained model  """
    c = 0.25
    model = 0
    vectorizer = 0

    def __init__(self):
        try:
            Ki.model = load('kiapp/model.joblib')
            Ki.vectorizer = load('kiapp/vectorizer.joblib')
        except :
            print('Model or Vectorizer file not found at kiapp - plese generate')

    def generate_model(self):
        """ Generates the KI Model based on pos.txt & neg.txt
            Outputs given accuracy based on data split and then calculates a final model with all data
            Model is saved as joblib file model.joblib
            Vectorizer is saved as joblib file vectorizer.joblib
            Both files can be used for evaluating a file or text without calculating the model again 
        """
        print("\nGenerating model...\n")
        # Loading training data
        reviews_pos = load_reviews_from_file("kiapp/data/pos.txt")
        reviews_neg = load_reviews_from_file("kiapp/data/neg.txt")
        reviews_val = [1 if i < 2500 else 0 for i in range(5000)] # first 2500 positive (1), other negative (0)
        
        reviews_train = reviews_pos + reviews_neg #combine
        reviews_train_clean = preprocess_reviews(reviews_train) #clean
        #New Vecotrizer
        vc = CountVectorizer(binary=True, stop_words='english', strip_accents='ascii')
        # Transform data into Matrix
        vc.fit(reviews_train_clean)
        X = vc.transform(reviews_train_clean)
        #Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, reviews_val, train_size = 0.75
        )
        #Accuracy on test data
        lr = LogisticRegression(C=Ki.c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" 
            % (Ki.c, accuracy_score(y_test, lr.predict(X_test))))
        print('Report:')
        print(classification_report(y_test, lr.predict(X_test)))
        #Final Model
        #Using Logistic Regression 
        final_model = LogisticRegression(C=Ki.c)
        #Fit final Model
        final_model.fit(X, reviews_val)
        Ki.model = final_model
        Ki.vectorizer = vc
        # Display positive and negative Words
        feature_to_coef = {
            word: coef for word, coef in zip(
                Ki.vectorizer.get_feature_names(), Ki.model.coef_[0]
            )
        }
        print("Positive words:\n")
        for best_positive in sorted(
            feature_to_coef.items(), 
            key=lambda x: x[1], 
            reverse=True)[:6]:
            print (best_positive)
        print("Negative words\n")
        for best_negative in sorted(
            feature_to_coef.items(), 
            key=lambda x: x[1])[:6]:
            print (best_negative)
        
        dump(Ki.model, 'kiapp/model.joblib')
        dump(Ki.vectorizer, 'kiapp/vectorizer.joblib')

    def evaluate_file(self, file="kiapp/data/evaluation.txt"):
        """ Evaluates a file based on a trained model """

        # Load file and clean data
        reviews_test = preprocess_reviews(load_reviews_from_file(file))
        X_test = Ki.vectorizer.transform(reviews_test)  #Vectorize evaluation data
        y_pred  = Ki.model.predict(X_test) # Predict data based on trained model
        # Export evaluated data as csv
        with open('evaluation.csv', mode='w') as eval_file:
            eval_writer = csv.writer(eval_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(len(reviews_test)):
                evaluation = 'Negative'
                if (y_pred[i] == 1):
                    evaluation = 'Positive'
                eval_writer.writerow([reviews_test[i], evaluation])
        print("File exported as evaluation.csv")


    def evaluate_text(self, text):
        """ Evaluates a given sentence/text e.g. for commandline input """
        review = [text]
        text_clean = preprocess_reviews(review)
        X_test = Ki.vectorizer.transform(text_clean)
        y_pred  = Ki.model.predict(X_test)
        result = y_pred[0]
        if (result == 1):
            return 'Positive'
        else:
            return 'Negative'

    def accuracy(self,):
        reviews_pos = load_reviews_from_file("kiapp/data/pos.txt")
        reviews_neg = load_reviews_from_file("kiapp/data/neg.txt")
        reviews_val = [1 if i < 2500 else 0 for i in range(5000)] # first 2500 positive (1), other negative (0)
        
        reviews_train = reviews_pos + reviews_neg #combine
        reviews_train_clean = preprocess_reviews(reviews_train) #clean
        X = Ki.vectorizer.transform(reviews_train_clean)
        X_train, X_test, y_train, y_test = train_test_split(
            X, reviews_val, train_size = 0.75
        )
        #Accuracy on test data
        lr = LogisticRegression(C=Ki.c)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print ("Accuracy for C=%s: %s" 
            % (Ki.c, accuracy_score(y_test, y_pred)))
        print('Report:')
        print(classification_report(y_test, y_pred))