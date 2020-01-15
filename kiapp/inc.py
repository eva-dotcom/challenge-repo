import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def load_reviews_from_file(location):
    reviews = []
    for line in open(location, 'r'):
        reviews.append(line.strip())    
    return reviews


def preprocess_reviews(reviews):
    reviews = [re.sub(r"n\'t", " not", line.lower()) for line in reviews]
    reviews = [re.sub(r"\'re", " are", line.lower()) for line in reviews]
    reviews = [re.sub(r"\'s", " is", line.lower()) for line in reviews]
    reviews = [re.sub(r"\'d", " would", line.lower()) for line in reviews]
    reviews = [re.sub(r"\'ll", " will", line.lower()) for line in reviews]
    reviews = [re.sub(r"\'t", " not", line.lower()) for line in reviews]
    reviews = [re.sub(r"\'ve", " have", line.lower()) for line in reviews]
    reviews = [re.sub(r"\'m", " am", line.lower()) for line in reviews]

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews