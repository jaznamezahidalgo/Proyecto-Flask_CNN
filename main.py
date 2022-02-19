from flask import Flask, request, make_response, redirect, render_template, session, url_for

from flask_bootstrap import Bootstrap

from flask_wtf import FlaskForm
from wtforms.fields import FileField, SubmitField

from wtforms.validators import DataRequired 

from werkzeug.utils import secure_filename

import os

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


class UploadForm(FlaskForm):
	image = FileField('Archivo de la imagen', validators=[DataRequired()])
	submit = SubmitField("Reconocer")

def upload(request):
    form = UploadForm(request.POST)
    if form.image.data:
        image_data = request.FILES[form.image.name].read()
        open(os.path.join(UPLOAD_PATH, form.image.data), 'w').write(image_data)

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['SECRET_KEY'] = 'SUPER SECRETO'

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CNN_FOLDER'] = 'cnn_files'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/test', methods=['GET', 'POST'])
def test():	
	user_ip = session.get('user_ip')
	upload_form = UploadForm()
	image = session.get('image')
	
	context = {
		'user_ip' : user_ip,
		'upload_form' : upload_form,
		'image' : image
	}

	if upload_form.validate_on_submit():
		file = request.files['image']
		if file and allowed_file(file.filename):					
			filename = secure_filename(file.filename)
			session['image'] = filename
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('recognize'))
	return render_template('test.html', **context)	

@app.route('/respuesta')
def execute():
	files = os.listdir(app.config['UPLOAD_FOLDER'])
	return render_template('respuesta.html', files=files, 
			file = session['image'])

@app.route('/')
def index():
	user_ip = request.remote_addr
	response = make_response(redirect('/test'))
	session['user_ip'] = user_ip

	return response	

@app.route('/recognize')
def recognize():
	path_cnn = os.path.join(app.config['CNN_FOLDER'], 'RedEjemplo.h5')
	modelo = keras.models.load_model(path_cnn)
	name_file = session['image'] # 'img_b.jpeg'
	img_path = os.path.join(app.config['UPLOAD_FOLDER'], name_file)
	img = image.load_img(img_path, target_size=(150, 150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = modelo.predict(x)
	preds = preds.reshape(2)
	lpred = np.argmax(preds)

	answer = ['Ant' if lpred==0 else 'Bee']
	session['answer'] = answer
	return render_template('respuesta.html',answer = session['answer'], file = name_file)
