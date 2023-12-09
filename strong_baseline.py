# Standard library imports
import numpy as np
import pandas as pd
import re
import warnings

# Data processing and NLP imports
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Keras and TensorFlow imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Custom imports
from attention import AttentionLayer

# Clearing any existing TensorFlow session
K.clear_session()

# Optional: Configuring warnings, if needed
warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", 200)

max_text_len = 30
max_summary_len = 8

embedding_dim = 100
latent_dim = 300

NUM_EPOCHS = 10
NUM_ROWS = 100000

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

def text_cleaner(text, num):
    """
    Cleans the input text by lowering case, removing HTML tags, substituting contractions,
    removing non-alphabetic characters, and filtering out stop words based on the 'num' flag.
    Long words (length > 1) are kept in the final result.

    :param text: The text to be cleaned.
    :param num: Flag to determine if stop words should be removed (0) or not (non-zero).
    :return: Cleaned text as a string.
    """
    # Initialize stopwords
    stop_words = set(stopwords.words('english')) 

    # Lowercase the text and remove HTML tags
    clean_text = text.lower()
    clean_text = BeautifulSoup(clean_text, "lxml").text

    # Replace contractions and remove possessive terminations
    clean_text = re.sub(r'\([^)]*\)', '', clean_text)
    clean_text = re.sub('"', '', clean_text)
    clean_text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in clean_text.split(" ")])    
    clean_text = re.sub(r"'s\b", "", clean_text)

    # Remove non-alphabetic characters and repeated 'm's
    clean_text = re.sub("[^a-zA-Z]", " ", clean_text) 
    clean_text = re.sub('[m]{2,}', 'mm', clean_text)

    # Tokenize and filter out stop words if num is 0
    if num == 0:
        tokens = [word for word in clean_text.split() if word not in stop_words]
    else:
        tokens = clean_text.split()

    # Filter out short words
    long_words = [word for word in tokens if len(word) > 1]

    return " ".join(long_words).strip()

def get_data():
    """
    Reads and processes the Kindle reviews dataset.
    It cleans the text and summary columns, removes duplicates and NaN values, 
    and returns the cleaned dataset.

    :return: Processed DataFrame with cleaned text and summary.
    """
    # Read the first 10000 rows of the dataset
    data = pd.read_csv("data/kindle_reviews.csv", nrows=NUM_ROWS)

    # Rename columns for clarity
    data = data.rename(columns={'reviewText': 'Text', 'summary': 'Summary'})

    # Remove duplicates and NaN values
    data.drop_duplicates(subset=['Text'], inplace=True)
    data.dropna(axis=0, inplace=True)

    # Clean the 'Text' and 'Summary' columns
    cleaned_text = [text_cleaner(text, 0) for text in data['Text']]
    cleaned_summary = [text_cleaner(summary, 1) for summary in data['Summary']]

    # Display first few cleaned texts and summaries
    print(cleaned_text[:5])
    print(cleaned_summary[:10])

    # Add cleaned data to DataFrame and remove rows with empty strings
    data['cleaned_text'] = cleaned_text
    data['cleaned_summary'] = cleaned_summary
    data.replace('', np.nan, inplace=True)
    data.dropna(axis=0, inplace=True)

    return data

def calculate_word_frequency(word_counts, threshold):
    """
    Calculate the count and frequency of words in a given dictionary that are below a certain threshold.

    :param word_counts: Dictionary of word counts.
    :param threshold: Threshold value to filter rare words.
    :return: Tuple containing counts and frequencies of words below the threshold.
    """
    cnt_rare = sum(1 for key, value in word_counts.items() if value < threshold)
    tot_cnt = len(word_counts)
    freq_rare = sum(value for key, value in word_counts.items() if value < threshold)
    tot_freq = sum(word_counts.values())

    return cnt_rare, tot_cnt, freq_rare, tot_freq

def remove_short_sequences(x_data, y_data, x_original_data, y_original_data, min_length=2):
    """
    Removes sequences from the input and output datasets where the output sequence is shorter than a minimum length.

    :param x_data: Input data sequences.
    :param y_data: Output data sequences.
    :param min_length: Minimum length of the output sequence to keep.
    :return: Filtered input and output data sequences.
    """
    filtered_indices = [i for i, seq in enumerate(y_data) if len([y for y in seq if y != 0]) > min_length]
    x_data_filtered = np.array([x_data[i] for i in filtered_indices])
    y_data_filtered = np.array([y_data[i] for i in filtered_indices])
    x_original_data_filtered = np.array([x_original_data[i] for i in filtered_indices])
    y_original_data_filtered = np.array([y_original_data[i] for i in filtered_indices])

    return x_data_filtered, y_data_filtered, x_original_data_filtered, y_original_data_filtered

def prepare_tokenizer(x_tr, x_val, y_tr, y_val, x_original, y_original):
    """
    Prepares tokenizers for the training and validation datasets, both for the input (x) and output (y) data.
    Filters out rare words based on a defined threshold and returns tokenized and padded training and 
    validation data along with the vocabulary size for both input and output.

    :param x_tr: Training data for input
    :param x_val: Validation data for input
    :param y_tr: Training data for output
    :param y_val: Validation data for output
    :return: Tuple containing tokenizers, vocabulary sizes, and processed training and validation datasets
    """
    # Tokenizer for input data
    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_tr))

    # Define threshold for rare words
    x_thresh = 4

    # Calculate the percentage of rare words in vocabulary
    x_cnt_rare, x_tot_cnt, x_freq_rare, x_tot_freq = calculate_word_frequency(x_tokenizer.word_counts, x_thresh)

    print("% of rare words in vocabulary:", (x_cnt_rare / x_tot_cnt) * 100)
    print("Total Coverage of rare words:", (x_freq_rare / x_tot_freq) * 100)

    # Re-initialize tokenizer by considering only the most common words
    x_tokenizer = Tokenizer(num_words=x_tot_cnt - x_cnt_rare)
    x_tokenizer.fit_on_texts(list(x_tr))

    # Tokenization and padding for input data
    x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
    x_val_seq = x_tokenizer.texts_to_sequences(x_val)
    x_tr = pad_sequences(x_tr_seq, maxlen=max_text_len, padding='post')
    x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

    # Vocabulary size for input
    x_voc_size = x_tokenizer.num_words + 1

    # Tokenizer for output data
    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_tr))

    # Define threshold for rare words in output
    y_thresh = 6

    # Calculate the percentage of rare words in vocabulary for output
    y_cnt_rare, y_tot_cnt, y_freq_rare, y_tot_freq = calculate_word_frequency(y_tokenizer.word_counts, y_thresh)

    print("% of rare words in vocabulary:", (y_cnt_rare / y_tot_cnt) * 100)
    print("Total Coverage of rare words:", (y_freq_rare / y_tot_freq) * 100)

    # Re-initialize tokenizer for output
    y_tokenizer = Tokenizer(num_words=y_tot_cnt - y_cnt_rare)
    y_tokenizer.fit_on_texts(list(y_tr))

    # Tokenization and padding for output data
    y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
    y_val_seq = y_tokenizer.texts_to_sequences(y_val)
    y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
    y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

    # Vocabulary size for output
    y_voc_size = y_tokenizer.num_words + 1

    # Remove sequences with only start and end tokens
    x_tr, y_tr, x_original_tr, y_original_tr = remove_short_sequences(x_tr, y_tr, x_original, y_original)
    x_val, y_val, _, _ = remove_short_sequences(x_val, y_val, x_original, y_original)

    return x_tokenizer, y_tokenizer, x_voc_size, y_voc_size, x_tr, x_val, y_tr, y_val, x_original_tr, y_original_tr

def build_model():
    """
    Builds and compiles the sequence-to-sequence model for text summarization.

    :return: The compiled model along with the encoder and decoder models for inference,
             and dictionaries for word-index mappings.
    """
    K.clear_session()

    # Encoder
    encoder_inputs = Input(shape=(max_text_len,))

    # Embedding layer for encoder
    enc_emb = Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)

    # LSTM layers for encoder
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    # Decoder
    decoder_inputs = Input(shape=(None,))

    # Embedding layer for decoder
    dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # LSTM layer for decoder
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # Concatenation of attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    # Dense layer
    decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Model definition
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())

    # Compile the model
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=NUM_EPOCHS,
                        callbacks=[es], batch_size=128, validation_data=([x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))

    # Dictionaries for word-index mappings
    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index

    # Encoder model for inference
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

    # Decoder setup for inference
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

    # Embeddings and LSTM for decoder
    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    # Attention inference
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # Dense softmax layer for inference
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    # Final decoder model
    decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
                          [decoder_outputs2] + [state_h2, state_c2])

    return model, encoder_model, decoder_model, target_word_index, reverse_target_word_index, reverse_source_word_index

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token_index == 0:
            # If 0, you can choose to either break the loop or continue
            # For this example, let's continue to the next iteration
            continue

        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

data = get_data()

df = pd.DataFrame({'text':np.array(data['cleaned_text']),'summary':np.array(data['cleaned_summary']), 'original_text': np.array(data['Text']), 'original_summary': np.array(data['Summary'])})

df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')

# Split the data
x_tr, x_val, y_tr, y_val, original_text_tr, _, original_summaries_tr, _ = train_test_split(
    np.array(df['text']),
    np.array(df['summary']),
    np.array(df['original_text']),
    np.array(df['original_summary']),
    test_size=0.1,
    random_state=0,
    shuffle=False
)

x_tokenizer, y_tokenizer, x_voc, y_voc, x_tr, x_val, y_tr, y_val, x_original_tr, y_original_tr = prepare_tokenizer(x_tr, x_val, y_tr, y_val, original_text_tr, original_summaries_tr)

model, encoder_model, decoder_model, target_word_index, reverse_target_word_index, reverse_source_word_index = build_model()

output_file = f"output_{NUM_EPOCHS}_{NUM_ROWS}.txt"

# Open the file in write mode
with open(output_file, "w") as file:
    for i in range(min(500, len(x_tr))):  # Ensure we don't exceed the length of x_tr
        # filtered_indices = [i for i, seq in enumerate(y_data) if len([y for y in seq if y != 0]) > min_length]

        # Retrieve the original review and summary
        original_review = x_original_tr[i]
        original_summary = y_original_tr[i]

        # Retrieve the tokenized review and summary text
        tokenized_review = seq2text(x_tr[i])
        tokenized_summary = seq2summary(y_tr[i])

        # Generate the predicted summary
        predicted_summary = decode_sequence(x_tr[i].reshape(1, max_text_len))

        print(f"Review: {original_review}")
        print(f"Tokenized Review: {tokenized_review}")
        print(f"Original Summary: {original_summary}")
        print(f"Tokenized Summary: {tokenized_summary}")
        print(f"Predicted Summary: {predicted_summary}")

        # Write all information to file
        file.write(f"Review: {original_review}\n")
        file.write(f"Tokenized Review: {tokenized_review}\n")
        file.write(f"Original Summary: {original_summary}\n")
        file.write(f"Tokenized Summary: {tokenized_summary}\n")
        file.write(f"Predicted Summary: {predicted_summary}\n")
        file.write("\n------------------------------------------------------\n\n")

print(f"Data saved to {output_file}")