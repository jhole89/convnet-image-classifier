from main.utils.main_parser import MainParser


def test_main_parser():

    parser = MainParser()
    parser.register_args()

    default_args = parser.parse_args(['/my/test/directory'])

    assert default_args.img_dir == '/my/test/directory'
    assert default_args.model_dir == 'tmp'
    assert default_args.iterations == 2000
    assert default_args.clean is False

    args_img = 'test_img_dir'
    args_model = ['-m', 'test_model_dir']
    args_iterations = ['-i', '500']
    args_options = ['-c']
    parsed_args = parser.parse_args([args_img] + args_model + args_iterations + args_options)

    assert parsed_args.img_dir == args_img
    assert parsed_args.model_dir == args_model[1]
    assert parsed_args.iterations == int(args_iterations[1])
    assert parsed_args.clean is True
