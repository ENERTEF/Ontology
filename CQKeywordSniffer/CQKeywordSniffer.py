import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import string

# Download resource
nltk.download('punkt')
nltk.download('wordnet')

# Read .csv file
csv_file = 'Competency questions.csv'
df = pd.read_csv(csv_file)

# Combine all the question to a whole text string
all_text = ' '.join(df['CQ'])

# Divide the words & transfer them all to lowercase letters
words = word_tokenize(all_text)
words = [word.lower() for word in words]

# Read stop words list
with open('English Stopwords.txt', 'r') as file:
    stop_words = [line.strip() for line in file.readlines()]

# Screen out punctuation, numbers and stop words
words = [word for word in words if word not in string.punctuation and not word.isnumeric() and word not in stop_words]

# Transfer the nouns and verbs to regular form
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word, pos='n') if lemmatizer.lemmatize(word, pos='n') != word else lemmatizer.lemmatize(word, pos='v') for word in words]

# Set up the frequency list
flist = FreqDist(words)

# Print the words and the frequency of each
for word, freq in flist.most_common():
    print(f'{word}: {freq}')
