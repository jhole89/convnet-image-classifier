import os
from main.directory import Directory


def test_directory():

    test_path = os.path.join('test', 'resources', 'test_dir')

    model_dir = Directory(test_path)

    model_dir.create()
    assert os.path.exists(test_path)

    model_dir.remove()
    assert os.path.exists(test_path) is False
