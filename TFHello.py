import tensorflow as tf

# Cria uma constrante no TensorFlow
hello = tf.constant('Hello, TensorFlow!')

# Inicia a sess�o
sess = tf.Session()

# Executa a Operacao
print(sess.run(hello))