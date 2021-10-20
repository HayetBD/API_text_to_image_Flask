from flask import Flask, render_template, request
import numpy as np
import skipthoughts
import utils
import tensorflow as tf
import random
import tensorlayer as tl
from models_resnet import generator_txt2img_resnet

app = Flask(__name__)

#app.config.from_object(os.environ["APP_SETTINGS"])
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#db = SQLAlchemy(app)

Generator = generator_txt2img_resnet([None, 356], is_train=False)

tl.files.load_hdf5_to_weights('G.h5', Generator)
Generator.eval()
embedded_model = skipthoughts.load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method =='POST':
        data = dict(request.form.items())
        description = [data["description"]]

        caption_vectors = skipthoughts.encode(embedded_model, description)
        tensor_text_embedding = tf.convert_to_tensor(caption_vectors)
        reduced_text_embedding = utils.lrelu(utils.linear(tensor_text_embedding, 256))

        z = np.random.normal(loc=0.0, scale=1.0, size=[1, 100]).astype(np.float32)
        z = tf.convert_to_tensor(z)

        z_text_concat = tf.concat([z, reduced_text_embedding], 1)
        img = Generator(z_text_concat)
        img = img + 1
        img = img * 127.5

        img = img.numpy().squeeze().astype(np.uint8)
        i = random.uniform(0.5, 10.5)
        imgname = 'imgtest'+str(i)+'.jpg'
        #tl.visualize.save_image(img, 'static/imgtest'+str(i)+'.jpg')
        tl.visualize.save_image(img, 'static/'+imgname)
        i=i+1

    return render_template("predicted.html", imgname=imgname)

if __name__ == '__main__':
    app.run(debug=True)