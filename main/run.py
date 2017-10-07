from main.utils.logger import Logger
from main.utils.main_parser import MainParser
from main.directory import Directory
from main.cnn_model import ConvNet

if __name__ == '__main__':

    main_parser = MainParser().register_args()
    args = main_parser.parse_args()

    Logger(log_level=args.verbosity)

    model_dir = Directory(args.model_dir)
    image_dir = Directory(args.img_dir)

    if args.clean:
        model_dir.remove()

    model_dir.create()

    model = ConvNet(model_dir, image_dir, img_size=64, channels=3, batch_size=128)

    model.train(training_epochs=50)

    prediction, ground_truth = model.predict(args.img_dir, args.model_dir)
