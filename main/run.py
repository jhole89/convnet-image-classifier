from main.utils.logger import Logger
from main.utils.main_parser import MainParser
from main.directory import Directory
from main.cnn_model import train, predict

if __name__ == '__main__':
    Logger(log_level='INFO')

    main_parser = MainParser().register_args()
    args = main_parser.parse_args()

    model_dir = Directory(args.model_dir)

    if args.clean:
        model_dir.remove()

    model_dir.create()

    # file_manager.organise_files(extract_dir, category_rules={'benign': 'SOB_B_.*.png', 'malignant': 'SOB_M_.*.png'})

    train(args.img_dir, args.model_dir)

    prediction, ground_truth = predict(args.img_dir, args.model_dir)
