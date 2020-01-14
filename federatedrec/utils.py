import os
import io
import zipfile


def zip_dir_as_bytes(path):
    with io.BytesIO() as io_bytes:
        with zipfile.ZipFile(io_bytes, 'w', zipfile.ZIP_DEFLATED) as zipper:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    full_path = os.path.join(root, name)
                    relative_path = os.path.relpath(full_path, path)
                    zipper.write(filename=full_path, arcname=relative_path)
                for name in dirs:
                    full_path = os.path.join(root, name)
                    relative_path = os.path.relpath(full_path, path)
                    zipper.write(filename=full_path, arcname=relative_path)
        zip_bytes = io_bytes.getvalue()
    return zip_bytes
