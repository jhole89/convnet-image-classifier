from main.utils.logger import Logger
from main.utils.main_parser import MainParser
from main.directory import Directory
from main.convnet import ConvNet

if __name__ == '__main__':

    main_parser = MainParser().register_args()
    args = main_parser.parse_args()

    logger = Logger(log_level=args.verbosity)
    logger.set_log_level()

    model_dir = Directory(args.model_dir)
    image_dir = Directory(args.img_dir)

    img_size = 64
    channels = 3
    filter_size = 3

    if args.mode == 'train':

        if args.clean:
            model_dir.remove()

        model_dir.create()

        model = ConvNet(model_dir.path, image_dir.path, img_size, channels, filter_size, batch_size=1)

        model.train(training_epochs=args.iterations)

    else:
        model = ConvNet(model_dir.path, image_dir.path, img_size, channels, filter_size, batch_size=1)
        model.predict()
    # prediction, ground_truth = model.predict(args.img_dir, args.model_dir)
