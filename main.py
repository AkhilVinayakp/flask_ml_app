from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# loading the model
with open('model.pkl', 'rb') as fp:
    model = pickle.load(fp)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods= ['POST'])
def predict():
    try:
        sl = float(request.values['SL'])
        sw = float(request.values['SW'])
        pl = float(request.values['PL'])
        pw = float(request.values['PW'])
    except Exception:
        render_template('home.html')
    input_ = [sl, sw, pl, pw]
    input_ = np.reshape(input_, (-1,4))
    scaler = StandardScaler()
    input_ = scaler.fit_transform(input_)
    output_ = model.predict(input_)
    return render_template('result.html', data = output_.item())


if __name__ == '__main__':
    app.run(port=8000)
