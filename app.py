from flask import render_template
#from textdetection import read_image
import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from obj_detection import video_upload

cwd = os.getcwd()

UPLOAD_FOLDER = os.path.join(cwd,'static')
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

## To do
## Add a reset link, this will reset all the variable values in the html file.
## Add CSS in the webpage.

@app.route('/', methods=['POST', "GET"])
def image_read():
    if request.method == 'POST':
        # check if the post request has the file part
        print("Here ....")
        print(request.files['file'])
        if 'file' not in request.files:
            # flash('No file part')
            print("File not found!!")
            return render_template('index.html')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print("File name empty!")
            # flash('No selected file')
            return render_template('index.html')
        if file and allowed_file(file.filename):
            print("File found", file.filename)
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            path, res = video_upload(file_path)

            return render_template('index.html', result = res)
        return render_template('index.html')
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(debug=True)