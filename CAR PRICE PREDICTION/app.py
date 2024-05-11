from flask import Flask, render_template,request
import pickle
import numpy as np

with open('Models/car_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# first route 
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def prediction():
    vehicle_name = int(request.args.get('vehicle_name'))
    car_brand = int(request.args.get('car_brand'))
    fuel_type = int(request.args.get('fuel_type'))
    kms_driven = int(request.args.get('kms_driven'))
    manufactured_year = int(request.args.get('manufactured_year'))
    diesel_binary = 0
    laxuary_company = 0
    new_car = 0
    laxuary_car_company = [23,"BMW","Mini",30,"Jaguar","Mitsubishi","Volvo"]

    if fuel_type == 2:
        diesel_binary = 1
    if car_brand in laxuary_car_company:
        laxuary_company = 1
    if int(kms_driven) <= 20000:
        new_car = 1
    [predicted_rate] = model.predict([[vehicle_name,car_brand,manufactured_year,kms_driven,diesel_binary,laxuary_company,new_car]])
    return render_template('index.html',predicted_rate=int(predicted_rate))



if __name__ == "__main__":
    app.run(debug=True)