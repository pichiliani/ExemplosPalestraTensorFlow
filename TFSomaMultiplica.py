import tensorflow as tf

# Duas constantes
a = tf.constant(2)
b = tf.constant(3)

# Iniciando o grafo 
with tf.Session() as sess:
    print("a=2, b=3")
    print("Adi��o: %i" % sess.run(a+b))
    print("Multiplica��o: %i" % sess.run(a*b))

# Uso de Placeholders e tipos de dados
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Definindo as opera��es usando add() e mul()
add = tf.add(a, b)
mul = tf.mul(a, b)

# Iniciando o grafo 
with tf.Session() as sess:
    # Passando os valores na hora da chamada!
    print("Adi��o com vari�veis: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multi com vari�veis: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


# Agora com matrizes!
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

# Definindo a opera��o usando o matmul()
product = tf.matmul(matrix1, matrix2)

#Rodando e obtendo o resultado
with tf.Session() as sess:
    result = sess.run(product)
    print(result)   # Resultado deve ser algo como [[ 12.]]
