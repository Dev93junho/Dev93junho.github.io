import tensorflow as tf

class DNN:
    def __init__(self, sess):
        self.sess = sess
        self.network()
        
    def network(self):
        self.X = tf.placeholder(tf.float32, [None,784])
        self.Y = tf.placeholder(tf.float32, [None,10])
        
        L1 = tf.layers.dense(self.X, 256, activation=tf.nn.relu)
        L2 = tf.layers.dense(L1, 256, activation=tf.nn.relu)
        self.logits = tf.layers.dense(L2, 10, activation=None)
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, label=self.Y))
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)
        
        self.predicted = tf.argmax(self.logits,1)
        correction = tf.equal(self.predicted, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))
        
    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X:x_data, self.Y:y_data})
    
    def predict(self, x_data):
        return self.sess.run(self.predicted, feed_dict={self.X:x_data})
    
    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test})