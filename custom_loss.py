
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):

    def loss(y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        focal = (
            alpha *
            tf.pow(
                1 - y_pred,
                gamma
            ) *
            cross_entropy
        )

        return tf.reduce_mean(
            tf.reduce_sum(
                focal,
                axis=1
            )
        )

    return loss
