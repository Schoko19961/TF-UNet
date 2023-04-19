import tensorflow as tf

class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, filters: int):
        super(DoubleConv, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), kernel_initializer="he_normal", strides=(1,1), padding="same", activation="relu")
        self.c2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), kernel_initializer="he_normal", strides=(1,1), padding="same", activation="relu")
        self.drop = tf.keras.layers.SpatialDropout2D(rate=.1)

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.drop(x)
        return x
    

class DownSample(tf.keras.layers.Layer):
    def __init__(self, filters: int):
        super(DownSample, self).__init__()
        self.c1 = DoubleConv(filters=filters)
        self.p1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

    def call(self, x):
        skip_connection = self.c1(x)
        x = self.p1(skip_connection)
        return skip_connection, x
    

class UpSample(tf.keras.layers.Layer):
    def __init__(self, filters: int):
        super(UpSample, self).__init__()
        self.t1 = tf.keras.layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="same")
        self.c1 = DoubleConv(filters=filters)

    def call(self, x, skip_connection):
        x = self.t1(x)
        x = tf.keras.layers.concatenate([x, skip_connection])
        x = self.c1(x)
        return x
