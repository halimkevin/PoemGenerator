import os
# os.rename("abstractive-news-summary","summary")


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import choice
from summary.attention import AttentionLayer  #External Library for Attention-based Algorithm



df = pd.read_csv("puisi.csv")
df = df[["puisi", "title"]]

puisi = [str(s).lower().replace("\n","newline") for s in df["puisi"]]  #Replace \n dengan kata newline
title = [str(s).lower() for s in df["title"]]

MAXLEN_PUISI = 100  #Maksimal panjang puisi (banyak kata)
MAXLEN_TITLE = 3    #Maksimal panjang judul (banyak kata)
MIN_WORD_FREQ = 10  #Threshold untuk kata-kata yang jarang muncul



def filterByLength(puisi, title, show=False):
    puisi_cnt = []
    title_cnt = []
    puisi_short = []
    title_short = []
    
    for p, t in zip(puisi, title):
        if (len(p.split())>MAXLEN_PUISI or len(t.split())>MAXLEN_TITLE):  #Hanya mengambil puisi yang panjangnya dibawah MAXLEN
            continue
        puisi_cnt.append(len(p.split()))
        title_cnt.append(len(t.split()))
        puisi_short.append(p)
        title_short.append(t)
    if (show):
        length_df = pd.DataFrame({
            'puisi': puisi_cnt,
            'title': title_cnt,
        })
        length_df.hist(bins=15)
        plt.show()
    
    return puisi_short, title_short

puisi, title = filterByLength(puisi, title)


def getDataset(puisi, title):
    df = pd.DataFrame({
        'puisi': puisi,
        'title': title
    })
    df['puisi'] = df['puisi'].apply(lambda x: 'sostok '+x+' eostok')  #start of string token + string + end of string token
 
    x_tr, x_val, y_tr, y_val = train_test_split(  #Training:Validation = 80:20
        np.array(df['title']),
        np.array(df['puisi']),
        test_size = 0.2,
        random_state = 0,
        shuffle = True
    )
    
    return x_tr, x_val, y_tr, y_val

x_tr, x_val, y_tr, y_val = getDataset(puisi, title)
# print(x_tr[0])
# print(y_tr[0])

def tokenize(tr, val, maxlen):  #Tokenize + Label Encoding
    tok = Tokenizer()
    tok.fit_on_texts(list(tr))

    total = 0
    cnt = 0

    for key, value in tok.word_counts.items():  #Mencari kata-kata yang jarang muncul
        total += 1
        if (value < MIN_WORD_FREQ):
            cnt += 1

    tok = Tokenizer(num_words=total-cnt)  #Buang kata-kata yang jarang muncul
    tok.fit_on_texts(list(tr))
    tr_seq = tok.texts_to_sequences(tr)
    val_seq = tok.texts_to_sequences(val)

    tr = pad_sequences(tr_seq, maxlen=maxlen, padding='pre')  #Zero-padding di awal agak tidak mengganggu prediksi selanjutnya
    val = pad_sequences(val_seq, maxlen=maxlen, padding='post')  #Zero-padding di akhir, tidak berpengaruh karena saat generate, 
                                                                 #jika outputnya eostok, maka langsung di-terminate

    voc = tok.num_words + 1
    
    return tr, val, tok, voc

x_tr, x_val, x_tok, x_voc = tokenize(x_tr, x_val, MAXLEN_TITLE)
y_tr, y_val, y_tok, y_voc = tokenize(y_tr, y_val, MAXLEN_PUISI)



K.clear_session()
 
latent_dim = 512
embedding_dim = 128
 
encoder_inputs = Input(shape=(MAXLEN_TITLE,))
 
enc_emb =  Embedding(x_voc, embedding_dim, trainable=True, mask_zero=True)(encoder_inputs)
 
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.8, recurrent_dropout=0.8)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
 
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
 
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.8, recurrent_dropout=0.5)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)
 
decoder_inputs = Input(shape=(None,))
 
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
 
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.8, recurrent_dropout=0.4)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
 
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
 
decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)
 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
 
# model.summary()


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  #Loss untuk Label-Encoding
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)  #Early stopping pada Validation Loss paling kecil
mc = ModelCheckpoint('A01-{epoch:02d}-{val_loss:.5f}.hdf5',  #Save model tiap epoch
                         monitor='val_loss', 
                         mode='min', 
                         verbose=0, 
                         save_freq='epoch', 
                         save_best_only=True,
                         save_weights_only=True
)
 
# history = model.fit(
#   [x_tr,y_tr[:,:-1]],  #Input: Judul puisi + isi puisi dari kata pertama hingga kata sebelum terakhir
#   [y_tr.reshape(y_tr.shape[0], y_tr.shape[1])[:,1:]],  #Output: Kata selanjutnya, yaitu kata kedua hingga kata terakhir
#   epochs=50,
#   callbacks=[es, mc],
#   batch_size=64,
#   validation_data=(
#     [x_val, y_val[:,:-1]],
#     [y_val.reshape(y_val.shape[0], y_val.shape[1])[:,1:]]
#   )
# )


encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(MAXLEN_TITLE,latent_dim))

dec_emb2= dec_emb_layer(decoder_inputs) 

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

decoder_outputs2 = decoder_dense(decoder_inf_concat) 

decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2]
)


reverse_target_word_index=y_tok.index_word
reverse_source_word_index=x_tok.index_word
target_word_index=y_tok.word_index

def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)  #Encode judul puisi
    
    target_seq = np.zeros((1,1))
    
    target_seq[0, 0] = target_word_index['sostok']  #Generate mulai dari sostok
    
    history = [-1, target_seq[0, 0]]

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])  #Generate kata selanjutnya
        while True:
            sampled_token_index = choice(range(len(output_tokens[0,-1,:])), p=output_tokens[0,-1,:])  #Menghindari kata ulang
            if (sampled_token_index==int(history[0]) or sampled_token_index==int(history[1])):
                continue
                
            sampled_token = reverse_target_word_index[sampled_token_index]  #Reverse dari vector embedding ke string
            break
        
        history[0] = history[1]
        history[1] = sampled_token_index
        
        if(sampled_token!='eostok'):
            decoded_sentence += sampled_token+' '

            if (len(decoded_sentence.split()) >= (MAXLEN_PUISI)):
                stop_condition = True

            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index  #Update kata sekarang menjadi kata yang baru diprediksi

            e_h, e_c = h, c  #Update juga hidden state dan cell state
        else:
            stop_condition = True
    
    return decoded_sentence.replace("newline ","\n")  #Replace "newline" dengan "\n" untuk newline yang sesungguhnya


model.load_weights("weight-epoch-best.hdf5")  #Masukkan weight yang sudah di-training

testTitle = "cinta ibu"  #Masukkan judul puisi

testTitle = np.array([testTitle])
x_test_seq = x_tok.texts_to_sequences(testTitle)
x_test = pad_sequences(x_test_seq, maxlen=MAXLEN_TITLE, padding='pre')  #Zero-padding di awal

# print(testTitle[0])
# print("Oleh Word2Vec + LSTM + Attention Layer")
# print(decode_sequence(x_test[0].reshape(1,MAXLEN_TITLE)))  #Have fun!
