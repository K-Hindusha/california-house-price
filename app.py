import pickle
from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd

app=Flask(__name__)
#load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    print("Received data:", data)
    new_data = np.array(list(data.values())).reshape(1, -1)
    prediction = regmodel.predict(new_data)[0]

    return jsonify({"prediction": float(prediction)})
@app.route('/pred',methods=['POST'])
def pred():
    data=[float(x) for x in request.form.values()]
    final_data=np.array(data).reshape(1,-1)
    output=float(regmodel.predict(final_data)[0])
    price = f"${output * 100000:,.2f}"

    return render_template(
        "home.html",
        prediction=f"The predicted house price is {price}"
    )


if __name__=="__main__":
    app.run(debug=True)

