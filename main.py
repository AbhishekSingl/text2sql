import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import json
import math
from symspellpy import SymSpell
import contractions
from flask import Flask, render_template, request
app = Flask(__name__)


def convert(utterence):
    def data_import(file='train.json', max_length=200):
        file = 'train.json'
        with open(file, 'r') as f:
            split_data = json.load(f)

        query = []
        utterance = []

        for i in split_data:
            if len(i['interaction']) == 0:
                q = i['final']['query']
                u = i['final']['utterance']
            else:
                q = i['interaction'][0]['query']
                u = i['interaction'][0]['utterance']

            if len(q) <= 200:
                query.append(q)
                utterance.append(u)

        return query, utterance

    # Converts the unicode file to ascii
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(w):
        w = unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        w = re.sub(r"([^A-z0-9 ])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.

        w = '<start> ' + w + ' <end>'
        return w

    def create_dataset(path, num_examples, show_example=False):
        lines = path
        word_pairs = []
        for l in lines:
            word = preprocess_sentence(l)
            word_pairs.append(word)


        if show_example:
            print(word_pairs[0])
        return word_pairs

    def spell_checker(inputTerm, path='./dictionary.txt'):
        symspell = SymSpell()
        symspell.load_dictionary(path, term_index=0, count_index=1)
        maxEditDistance = 2
        # ignore_non_words = True means if a particular word is not present then
        # we'll return as is.
        correct_sent = []
        for i in inputTerm.split():
            if i.isalnum():
                suggestion = symspell.lookup_compound(i, maxEditDistance, ignore_non_words=True)
                suggestion = str(suggestion[0]).split(',')[0].strip()
            else:
                suggestion = i.strip()
            correct_sent.append(suggestion)

        return " ".join(correct_sent)

    def tokenize(lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', oov_token="<unk>")
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')

        return tensor, lang_tokenizer

    def glove_embeddings(GLOVE_DIR, GLOVE_FILE, EMBEDDING_DIM):
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
        emb_matrix = []
        for line in f:
            values = line.split()
            word = " ".join(values[:-1 * EMBEDDING_DIM])
            coefs = np.asarray(values[-1 * EMBEDDING_DIM:], dtype='float32')
            embeddings_index[word] = coefs
            emb_matrix.append(coefs)
        f.close()

        embeddings_index['<unk>'] = np.mean(emb_matrix, axis=0)

        scale = 1 / max(1., (len(emb_matrix[0])) / 2.)
        limit = math.sqrt(3.0 * scale)
        embeddings_index['<start>'] = np.random.uniform(-limit, limit, size=(1, EMBEDDING_DIM))
        embeddings_index['<end>'] = np.random.uniform(-limit, limit, size=(1, EMBEDDING_DIM))



        return embeddings_index

    def embedding_matrix(embeddings_index, EMBEDDING_DIM, word_index):
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = embeddings_index['<unk>']
                + np.random.rand(1, EMBEDDING_DIM)
        return embedding_matrix

    def retrieve_embedding(filename='Embedding'):
        with open(f'{filename}_shape_info.txt', 'r') as f:
            for i in f:
                shape = (int(i.split()[0]), int(i.split()[1]))

        emb_matrix = np.fromfile(f'{filename}.dat').reshape(shape)
        return emb_matrix

    class Encoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, emb_type='default', emb_mat=[]):
            super(Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            if emb_type == 'glove':
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[emb_mat])
            else:
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')

        def call(self, x, hidden):
            x = self.embedding(x)
            output, state = self.gru(x, initial_state=hidden)
            return output, state

        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.enc_units))

    class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            # query hidden state shape == (batch_size, hidden size)
            # query_with_time_axis shape == (batch_size, 1, hidden size)
            # values shape == (batch_size, max_len, hidden size)
            # we are doing this to broadcast addition along the time axis to calculate the score
            query_with_time_axis = tf.expand_dims(query, 1)

            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            score = self.V(tf.nn.tanh(
                self.W1(query_with_time_axis) + self.W2(values)))

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return context_vector, attention_weights

    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, emb_type='default', emb_mat=[]):
            super(Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units

            if emb_type == 'glove':
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[emb_mat])
            else:
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(vocab_size)

            # used for attention
            self.attention = BahdanauAttention(self.dec_units)

        def call(self, x, hidden, enc_output):
            # enc_output shape == (batch_size, max_length, hidden_size)
            context_vector, attention_weights = self.attention(hidden, enc_output)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size, vocab)
            x = self.fc(output)

            return x, state, attention_weights

    # Data Import
    query, utterance = data_import('train.json')

    # Calling data cleaning and processing function
    # print('---Data after preprocessing and cleaning---')
    en = create_dataset(utterance, None, show_example=False)
    sql = create_dataset(query, None, show_example=False)

    # Tokenization
    targ_lang, inp_lang = sql, en
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
        train_test_split(input_tensor, target_tensor, test_size=0.2)
    inp_lang = inp_lang_tokenizer
    targ_lang = targ_lang_tokenizer

    input_emb_matrix = retrieve_embedding('input_emb_matrix')
    target_emb_matrix = retrieve_embedding('target_emb_matrix')

    # Defining Variables for Modelling
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 32
    embedding_dim = 300
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    steps_per_epoch_val = len(input_tensor_val)//BATCH_SIZE
    units = 1024
    vocab_inp_size = len(inp_lang_tokenizer.word_index)+1
    vocab_tar_size = len(targ_lang_tokenizer.word_index)+1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset_val = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
    dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))

    # Encoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, 'glove', input_emb_matrix)

    # sample input
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

    # Attention Layer
    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    # Decoder
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, 'glove', target_emb_matrix)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                          sample_hidden, sample_output)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    checkpoint_dir = '../training_checkpoints_glove_model'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    # restoring the checkpoint in checkpoint_dir
    checkpoint.restore(checkpoint_dir+'/ckpt-15')

    def evaluate(sentence, preprocess=False):
        attention_plot = np.zeros((max_length_targ, max_length_inp))

        if preprocess:
            sentence = preprocess_sentence(sentence)

        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = []

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            predicted_id = tf.argmax(predictions[0]).numpy()
            if targ_lang.index_word[predicted_id] == '<end>':
                return " ".join(result), sentence, attention_plot

                # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result.append(targ_lang.index_word[predicted_id])

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return " ".join(result), sentence, attention_plot

    # function for plotting the attention weights
    def plot_attention(attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')
        for (i, j), z in np.ndenumerate(attention):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    def translate(sentence, plot=True, preprocess=False):
        result, sentence, attention_plot = evaluate(sentence, preprocess)

        if plot == True:
            print('Input: %s' % (sentence))
            print('Predicted translation: {}'.format(result))

            attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
            plot_attention(attention_plot, sentence.split(' '), result.split(' '))

        return result

    os.chdir('../')
    return translate(utterence,plot=False, preprocess= True)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')


@app.route("/get")
def get_bot_response():
    # function for the bot response
    query = []
    utterence = request.args.get('msg')
    os.chdir('data/')
    return convert(utterence)


if __name__ == '__main__':
    app.run(debug=True)