import platform, logging
from flask import Flask, send_file, send_from_directory, jsonify
from logging.handlers import RotatingFileHandler

DEFAULT_DATA_FOLDER = "/data/projects/fate/python/fml_agent/data"
handler = RotatingFileHandler('/data/projects/fate/python/fml_agent/fml_agent.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
api = Flask(__name__)
api.logger.addHandler(handler)

@api.route("/api/version")
def version():
    api.logger.info("Checking versiong...")
    return "v1"

@api.route("/api/download/<file_name>")
def get_file(file_name):
    """Download a file."""
    
    api.logger.info("Download file: %s" % file_name)
    if platform.system() == "Windows":
        print("Test case: %s" % path)
        return send_file(file_name, as_attachment=True)

    return send_from_directory(DEFAULT_DATA_FOLDER, file_name, as_attachment=True)

if __name__ == '__main__':
    api.run(host="0.0.0.0", port=8484)