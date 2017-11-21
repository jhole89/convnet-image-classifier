import os

from main.utils.main_parser import MainParser


def test_main_parser(tmpdir):

    parser = MainParser()
    parser.register_args()

    args_img = str(tmpdir)
    args_mode = ['-m', 'train']
    args_model = ['-d', 'test/resources/model']
    args_iterations = ['-i', '500']
    args_batch = ['-b', '128']
    args_verbosity = ['-v', 'debug']
    args_options = ['-c']

    default_args = parser.parse_args([args_img])

    assert default_args.img_dir == tmpdir
    assert default_args.mode == 'predict'
    assert default_args.model_dir == os.path.abspath('tmp')
    assert default_args.iterations == 50
    assert default_args.batch_size == 1
    assert default_args.verbosity == 'INFO'
    assert default_args.clean is False

    parsed_args = parser.parse_args(
        [args_img] + args_mode + args_model + args_iterations + args_batch + args_verbosity + args_options
    )

    assert parsed_args.img_dir == tmpdir
    assert parsed_args.mode == args_mode[1]
    assert parsed_args.model_dir == args_model[1]
    assert parsed_args.iterations == int(args_iterations[1])
    assert parsed_args.batch_size == int(args_batch[1])
    assert parsed_args.verbosity == args_verbosity[1].upper()
    assert parsed_args.clean is True
