from Classifier.Models.NetworkModels import ClassifierModel
import keras
from keras.models import Model
from keras.optimizers import *
from keras.callbacks import *
import keras
from pickle import load


model_name = '417be_ClassifierModel_e003'
res = 128

main_path = 'C:/Users/jakub/Desktop/Inżynierka/'
models_dir = 'Models/'
model_path = main_path +  models_dir + model_name
model_class = ClassifierModel
tensors_dir = 'Tensors/'
tensor_path = main_path + tensors_dir + 'images.tsr'

if __name__ == "__main__":
    print(keras.backend.backend())
    model = keras.models.load_model(model_path)
    with open(tensor_path, 'rb') as images_file:
        images_tensor = load(images_file)

    sample_tensor = images_tensor.tensor[50:]
    prediction = model.predict(sample_tensor, None)
    print(prediction)
    print("trt")