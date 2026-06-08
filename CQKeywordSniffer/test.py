from collections import defaultdict, Counter
import string
import re

def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = {line.strip().lower() for line in file}
    return stop_words

def tokenize(text, stop_words):
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove punctuation except commas, lowercase
    text = text.translate(str.maketrans('', '', string.punctuation.replace(',', ''))).lower()
    tokens = text.split()
    # Filter out stop words
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def count_word_frequencies(tokens):
    word_counts = defaultdict(int)
    for word in tokens:
        word_counts[word] += 1
    return word_counts

def count_comma_separated_frequencies(text, stop_words):
    word_counts = defaultdict(int)
    segments = text.split(',')
    for segment in segments:
        word = segment.strip().lower()
        if word and word not in stop_words:
            word_counts[word] += 1
    return word_counts

def process_file(file_path, stop_words_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    stop_words = load_stop_words(stop_words_path)
    
    questions = []
    answers = []
    
    for line in lines:
        if '?' in line:
            question_part, answer_part = line.split('?', 1)
            questions.append(question_part.strip())
            answers.append(answer_part.strip())
    
    # Join all questions into a single string and all answers into a single string
    questions_text = ' '.join(questions)
    answers_text = ', '.join(answers)  # Keep commas for answer processing
    
    # Tokenize questions and filter stop words
    question_tokens = tokenize(questions_text, stop_words)
    # Filter stop words in answers text
    answer_frequencies = count_comma_separated_frequencies(answers_text, stop_words)
    
    # Count word frequencies in questions
    question_word_frequencies = count_word_frequencies(question_tokens)
    
    return question_word_frequencies, answer_frequencies

def main():
    file_path = 'Competency questions.csv'
    stop_words_path = 'English Stopwords.txt'
    
    question_frequencies, answer_frequencies = process_file(file_path, stop_words_path)
    
    print("Word frequencies in questions:")
    for word, count in question_frequencies.items():
        print(f'{word}: {count}')
    
    print("\nWord frequencies in answers:")
    for word, count in answer_frequencies.items():
        print(f'{word}: {count}')

if __name__ == "__main__":
    main()
