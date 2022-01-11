import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Dense

K.set_image_data_format("channels_first")


def resBlock(input_l, feature_size, kernel_size, scale=0.1):
    """Definition of Residual Block to be repeated in body of network."""
    tmp = Conv2D(
        feature_size, kernel_size, kernel_initializer="he_uniform", padding="same"
    )(input_l)
    tmp = Activation("relu")(tmp)
    tmp = Conv2D(
        feature_size, kernel_size, kernel_initializer="he_uniform", padding="same"
    )(tmp)

    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([input_l, tmp])


def DSen2CR_model(
    input_shape,
    batch_per_gpu=2,
    num_layers=32,
    feature_size=256,
    use_cloud_mask=True,
    include_sar_input=True,
):
    """Definition of network structure."""

    global shape_n

    # define dimensions
    input_opt = tf.keras.Input(shape=input_shape[0])
    input_opt = tf.transpose(input_opt, [0, 2, 3, 1])  # transpose to channels_last

    input_sar = tf.keras.Input(shape=input_shape[1])
    input_sar = tf.transpose(input_sar, [0, 2, 3, 1])  # transpose to channels_last

    first = None
    if include_sar_input:
        first = Concatenate(axis=-1)([input_opt, input_sar])
    else:
        first = input_opt

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer="he_uniform", padding="same")(first)
    x = Activation("relu")(x)

    # main body of network as succession of resblocks
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3])

    x = Conv2D(input_shape[0][1], (3, 3), kernel_initializer="he_uniform", padding="same")(x)
    x = Activation("relu")(x)

    # # transpose to channel first
    # x = tf.transpose(x, [0, 3, 1, 2])

    # # One more convolution
    # x = Conv2D(input_shape[0][0], (3, 3), kernel_initializer="he_uniform", padding="same")(x)

    # # transpose to channel last
    # x = tf.transpose(x, [0, 2, 3, 1])

    # Add first layer (long skip connection)
    x = Add()([x, first])

    if use_cloud_mask:
        # the hacky trick with global variables and with lambda functions is needed to avoid errors when
        # pickle saving the model. Tensors are not pickable.
        # This way, the Lambda function has no special arguments and is "encapsulated"

        shape_n = tf.shape(input_opt)

        def concatenate_array(x):
            global shape_n
            return tf.concat([x, tf.zeros((batch_per_gpu, 1, shape_n[1], shape_n[2]))], axis=1)
            # return K.concatenate(
            #     [x, K.zeros(shape=(batch_per_gpu, 1, shape_n[2], shape_n[3]))], axis=1
            # )

        x = Concatenate(axis=1)([x, first])

        # x = tf.keras.(concatenate_array)(x)
        x = Concatenate(axis=1)([x, tf.zeros((batch_per_gpu, 1, shape_n[1], shape_n[2]))])

    model = Model(inputs=[input_opt, input_sar], outputs=x)

    return model, shape_n
