from main.utils.main_parser import MainParser


def test_main_parser():

    parser = MainParser()
    parser.register_args()

    args_img = ['-d', 'test_img_dir']
    args_model = ['-m', 'test_model_dir']
    args_iterations = ['-i', '2000']
    args_options = ['-c']
    opts = parser.parse_args(args_img + args_model + args_iterations + args_options)

    assert opts.img_dir == args_img[1]
    assert opts.model_dir == args_model[1]
    assert opts.iterations == int(args_iterations[1])
    assert opts.clean is True
