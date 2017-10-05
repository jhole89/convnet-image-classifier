from main.utils.logger import Logger
from main.utils.main_parser import MainParser
from main.FileSystemManager import FileSystemManager
from main.cnn_model import train, predict

if __name__ == '__main__':
    Logger(log_level='INFO')

    main_parser = MainParser().register_args().parse_args()

    image_directory = 'images'
    model_directory = 'tmp'

    if main_parser.clean:

        file_manager = FileSystemManager(main_parser.img_dir, main_parser.model_dir)
        file_manager.clean_run()

        file_manager.data_science_fs(category0='benign', category1='malignant')
        # file_manager.organise_files(extract_dir, category_rules={'benign': 'SOB_B_.*.png', 'malignant': 'SOB_M_.*.png'})

    train(image_directory, model_directory)

    prediction, ground_truth = predict(image_directory, model_directory)

