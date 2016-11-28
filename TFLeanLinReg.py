import tflearn

# Dados X e Y para treinar o modelo
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

# Montando o grafico da regreessao
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)

# Indicando os parametros
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)

#Montando o modeo
m = tflearn.DNN(regression)

#Treinando o modelo. Note que fit() ja imprime na tela o resultado
# a cada epoca
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)


print("\Resultado final da regressao:")
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nPrevendo o valor Y para os dados x = 3.2, x =3.3, x =3.4:")
print(m.predict([3.2, 3.3, 3.4]))

