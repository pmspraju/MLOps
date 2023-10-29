import time
import os

import tensorflow as tf
from tensorflow import keras
import keras_cv
import matplotlib.pyplot as plt
from tokenizer import SimpleTokenizerLocal
from diffusion import StableDiffusionLocal

print("================================================================")
print("----TensorFlow version:", tf.__version__)
print("----Keras version:", keras.__version__)
print("----Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())
print("================================================================")

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["tf_gpu_allocator"]="cuda_malloc_async"
tf.keras.mixed_precision.set_global_policy("mixed_float16")

MAX_PROMPT_LENGTH = 77

class StableDiffusionTest(keras_cv.models.StableDiffusion):
#class StableDiffusionTest(StableDiffusionLocal):

    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=False,
    ):
        super().__init__(img_height, img_width, jit_compile)
        print('In local class')

    def tokenizer(self):
        """tokenizer returns the tokenizer used for text inputs.
        Can be overriden for tasks like textual inversion where the tokenizer needs to be modified.
        """

        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizerLocal()
        
        return self._tokenizer
    
    def encode_text(self, prompt):
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer().encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)

        context = self.text_encoder.predict_on_batch(
            [phrase, self._get_pos_ids()]
        )

        return context


#model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
model = StableDiffusionTest(img_width=512, img_height=512)

images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


#plot_images(images)

# class GrandParent(object):
#     def __init__(self):
#         self.value = 5

#     @property
#     def get_value(self):
#         print('Grandparent')
#         return self._value
    
#     def call_get_value(self):
#         return self.get_value()

# class Parent(GrandParent):

#     def __init__(self):
#         super().__init__()
    
#     def get_value_parent(self):
#         print('Parent')
#         return self.value
    
#     def tst_order_override(self):
#         return self.get_value_parent()

# class Child(Parent):
#     def __init__(self):
#         super().__init__()

#     def get_value(self):
#         print('overriden')
#         return self.value + 2
    
# c = Child()
# print('test overriding')
# print(c.get_value())
# print(c.call_get_value())