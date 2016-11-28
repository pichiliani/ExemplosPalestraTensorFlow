import tensorflow as tf

# Cria uma constrante no TensorFlow
hello = tf.constant('Hello, TensorFlow!')

# Inicia a sessão
sess = tf.Session()

# Executa a Operacao
print(sess.run(hello))