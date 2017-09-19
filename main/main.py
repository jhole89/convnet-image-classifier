from main.utils.logger import Logger
from main.FileSystemManager import FileSystemManager
from main.cnn_model import train, predict
from main.utils.sys_utils import graceful_exit


Logger(log_level='INFO')

image_directory = 'images'
model_directory = 'tmp'
clean_run = 'y'


if clean_run.upper() == 'Y':

    file_manager = FileSystemManager(image_directory, model_directory)
    file_manager.clean_run()

    file_manager.data_science_fs(category0='benign', category1='malignant')
    # file_manager.organise_files(extract_dir, category_rules={'benign': 'SOB_B_.*.png', 'malignant': 'SOB_M_.*.png'})

elif clean_run.upper() == 'N':
    pass

else:
    graceful_exit()

train(image_directory, model_directory)

prediction, ground_truth = predict(image_directory, model_directory)

