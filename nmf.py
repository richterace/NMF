import numpy as np

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def preprocess_text(text):
    # Simple preprocessing: lowercasing and removing punctuation
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text.split()

def build_term_frequency_matrix(sentences):
    word_index = {}
    index_word = {}
    current_index = 0
    
    for sentence in sentences:
        words = preprocess_text(sentence)
        for word in words:
            if word not in word_index:
                word_index[word] = current_index
                index_word[current_index] = word
                current_index += 1
    
    tf_matrix = np.zeros((len(sentences), len(word_index)))
    
    for i, sentence in enumerate(sentences):
        words = preprocess_text(sentence)
        for word in words:
            tf_matrix[i][word_index[word]] += 1
            
    return tf_matrix, word_index, index_word

def nmf(V, num_topics, max_iter=100, tol=1e-4):
    np.random.seed(42)
    W = np.random.rand(V.shape[0], num_topics)
    H = np.random.rand(num_topics, V.shape[1])
    
    for iteration in range(max_iter):
        WH = np.dot(W, H)
        cost = np.linalg.norm(V - WH, ord='fro')
        
        if cost < tol:
            break
            
        # Update H
        H *= np.dot(W.T, V) / np.dot(np.dot(W.T, W), H)
        # Update W
        W *= np.dot(V, H.T) / np.dot(W, np.dot(H, H.T))
    
    return W, H

def extract_summary(sentences, W, top_n=5):
    sentence_scores = np.sum(W, axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(-sentence_scores)]
    return ranked_sentences[:top_n]

def main(file_path, num_topics=5, top_n=5):
    sentences = read_text_file(file_path)
    tf_matrix, word_index, index_word = build_term_frequency_matrix(sentences)
    W, H = nmf(tf_matrix, num_topics)
    summary = extract_summary(sentences, W, top_n)
    
    print("Summary:")
    for sentence in summary:
        print(sentence.strip())

if __name__ == "__main__":
    main("terns_of_service.txt", num_topics=5, top_n=5)

