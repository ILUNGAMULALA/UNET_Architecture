import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation,MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
#This block is the building block of UNet and will be reused multiple times

    x = Conv2D(num_filters, 3, padding = "same")(inputs) # 3x3 Convolution
    x = BatchNormalization()(x) # Normalize activations
    x = Activation("relu")(x)  # ReLU Activation

    x = Conv2D(num_filters, 3, padding="same")(x) # Another 3x3 Convolution
    x = BatchNormalization()(x)
    x = Activation("relu")(x)   # ReLU Activation

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)  # Feature Extraction
    p= MaxPool2D((2, 2))(x)  # Reduce spatial size by half while keeping important information
    return x, p    # Return both extracted features and pooled output, X is needed for skip connections and p is used as input for the next encoder layer

def decoder_block(inputs,skip_features, num_filters):
    x=Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)  # Here it is for the Upsampling
    x = Concatenate()([x, skip_features])    # Combine with encoder feature map
    x = conv_block(x, num_filters)   # Apply convolutions
    return x

#The skip connection restores fine details lost during downsampling

def build_unet(input_shape):
    inputs = Input(input_shape)    # Define input shape

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    #print(s1.shape, s2.shape, s3.shape, s4.shape)
    #print(p1.shape, p2.shape, p3.shape, p4.shape)

    # Bottleneck (Deepest layer) for deep feature extraction
    b1= conv_block(p4, 1024)

    # Decoder
    d1= decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output Layer (Final segmentation mask). The use of sigmoid activation is for binary segmentation (1output channel, values between 0 and 1)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)


    # Build model
    model = Model(inputs, outputs, name="UNET")
    return model

if __name__ == '__main__':
    input_shape = (256, 256, 3)  # Image size 256x256 with 3 channels (RGB)
    model = build_unet(input_shape)  # Create UNet model
    model.summary()    # Display model architecture
