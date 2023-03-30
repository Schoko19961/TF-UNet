from typing import List, Tuple
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from .unet_modules import DownSample, DoubleConv, UpSample

class Unet(Model):
    def __init__(self, img_shape: Tuple[int, int, int], levels: List[int]):

        inputs = layers.Input(shape=img_shape)
        model = inputs

        skip_connections = []
        for filter in levels:
            skip_connection, model = DownSample(filter)(model)
            skip_connections.append(skip_connection)
        
        # Bottleneck
        model = DoubleConv(levels[-1] * 2)(model)

        for index, filter in enumerate(reversed(levels)):
            model = UpSample(filter)(model, skip_connections[-(index + 1)])
        # self.model = model
        outputs = layers.Conv2D(1, (1,1), activation="sigmoid")(model)
        
        super(Unet, self).__init__(inputs = [inputs], outputs = [outputs])

