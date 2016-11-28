import os
from six import moves
import ssl

import tflearn
from tflearn.data_utils import *

path = "US_Cities.txt"
if not os.path.isfile(path):
    context = ssl._create_unverified_context()
    moves.urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt", path, context=context)

maxlen = 20

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_us_cities')

from timeit import default_timer as timer
start = timer()

for i in range(1):
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id='us_cities')
    print("-- TESTING...")
    print("-- Test with temperature of 1.2 --")
    print(m.generate(30, temperature=1.2, seq_seed=seed))
    print("-- Test with temperature of 1.0 --")
    print(m.generate(30, temperature=1.0, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(m.generate(30, temperature=0.5, seq_seed=seed))

end = timer()

print("Tempo: %.2f segundos" % (end - start))

# Teste 1
# Tempo para uma interacao do loop: 909.68 segundos (15 min.) - 20.580 cidades
# Dois processadores e um core por processador
# resultado

#-- Test with temperature of 1.2 --
#
#Rives Junction
#Riveslaorn
#Ofel
#Fyphra Timtn
#Hoto 
#-- Test with temperature of 1.0 --
#
#Rives Junction
#Riveg
#ramheetSoili
#aa coleos
#Toehn
#-- Test with temperature of 0.5 --
#
#Rives Junction
#Riveha
#Bamlet
#Lhete
#Bortotae
#Radli
#Tempo: 909.68 segundos

# Teste 2
# Tempo para uma interacao do loop: 526.82 segundos - 8.7 min.
# Quatro processadores e dois cores por processador
# resultado

#-- Test with temperature of 1.2 --
#ddo Mills
#Caddo ValloLi
#Vlensrwlcln
#Polsnr
#Sieils
#
#-- Test with temperature of 1.0 --
#ddo Mills
#Caddo Vallnlrba
#Lirland Pevvrnaulost
#Jak
#-- Test with temperature of 0.5 --
#ddo Mills
#Caddo Valle
#Surisdoa
#Peete
#Hinre Fiinn
#S
#Tempo: 526.82 segundos
