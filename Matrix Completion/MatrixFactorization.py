from tqdm import tqdm
import tensorflow as tf

class MatrixFactorization:
    def __init__(self, R, k=3, learning_rate=0.005, l1=0.0, l2=0.01):
        self.R = tf.convert_to_tensor(R, dtype=tf.float32)
        self.nonzero = tf.not_equal(self.R, 0)  # Mask for non-zero entries
        self.m, self.n = R.shape
        self.learning_rate = learning_rate
        self.k = k
        self.l1 = l1  # L1 regularization parameter (Lasso)
        self.l2 = l2  # L2 regularization parameter (Ridge)
        self.U = tf.Variable(tf.random.uniform((self.m, self.k)))
        self.V = tf.Variable(tf.random.uniform((self.n, self.k)))
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

    def calc_loss(self):
        # Compute reconstructed matrix
        reconstructed = tf.matmul(self.U, self.V, transpose_b=True)
        
        # Calculate squared difference between non-zero entries of original matrix and reconstructed matrix
        E = tf.square(tf.boolean_mask(self.R, self.nonzero) - tf.boolean_mask(reconstructed, self.nonzero))
        
        # Compute the loss with L1 and L2 regularization
        l1_norm = tf.reduce_sum(tf.abs(self.U)) + tf.reduce_sum(tf.abs(self.V))
        l2_norm = tf.reduce_sum(self.U ** 2) + tf.reduce_sum(self.V ** 2)
        loss = tf.reduce_sum(E) + self.l1 * l1_norm + self.l2 * l2_norm

        print()
        print(loss.numpy())
        return loss

    def train(self, n_iterations=10000):
        for iterations in tqdm(range(n_iterations)):
            with tf.GradientTape() as tape:
                current_loss = self.calc_loss()
            gradients = tape.gradient(current_loss, [self.U, self.V])
            
            # Update U and V using optimizer and gradients
            self.optimizer.apply_gradients(zip(gradients, [self.U, self.V]))


    def get_v(self):
        return self.V
    
    def get_u(self):
        return self.U

    def get_matrix(self):
        return tf.matmul(self.U, self.V, transpose_b=True)

