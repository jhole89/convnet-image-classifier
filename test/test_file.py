from main.file import File
import tempfile
import os


def test_file():

    file_path = tempfile.NamedTemporaryFile().name

    file = File(file_path)

    assert file.path == os.path.abspath(file_path)

    file.remove()
    assert os.path.exists(file.path) is False
