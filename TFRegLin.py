import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parametros gerais
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Dados de treinamento (conjunto de treinamento)
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

#Nro de amostras
n_samples = train_X.shape[0]

# Colocando o  placeholder para os dados
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Variaveis com os pesos e bias ()
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construindo o modelo linear
pred = tf.add(tf.mul(X, W), b)

# Nossa m'etrica de custo: erro quadratico medio
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Funcao de otimizacao: Gradient descent 
# com parametros de taxa de aprendizado e custo
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Inicializando as variaveis
init = tf.initialize_all_variables()

# Disparando o grafico
with tf.Session() as sess:
    sess.run(init)

    # Treinando o modelo em training_epochs epocas
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Mostranddo o erro em cada epooca
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Terminou o treinamento")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Custo final=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Montando um grafico com os pontos e a linha que
    # representa o nosso modelo
    plt.plot(train_X, train_Y, 'ro', label='Dados originais')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Linha do modelo')
    plt.legend()
    plt.show()

    # Fazendo um teste com novos valores ((conjunto de teste)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testando com novos dados... (Comparacao de erro medio quadratico)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    
    print("Custo dos dados de teste=", testing_cost)
    print("Diferenca entre os custos :", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Dados de teste')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Linha gerada pelo modelo')
    plt.legend()
    plt.show()
