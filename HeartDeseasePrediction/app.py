from flask import Flask , render_template,request
import pickle 

with open('picklefiles/hd_model.pkl','rb') as file:
    model = pickle.load(file)
with open('picklefiles/hd_scaler_model.pkl','rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html') 

@app.route('/predict',methods=['GET'])
def prediction():
    age = request.args.get('age')
    sex = int(request.args.get('sex'))
    chest_pain = int(request.args.get('chest_pain'))
    bp = int(request.args.get('bp'))
    cholesterol = int(request.args.get('cholesterol'))
    fasting_blood_sugar = int(request.args.get('fasting_blood_sugar'))
    resting_ecg = int(request.args.get('resting_ecg'))
    exercise_angina = int(request.args.get('exercise_angina'))
    st_depression = float(request.args.get('st_depression'))
    num_vessels = int(request.args.get('num_vessels'))
    thalium_stress_test = int(request.args.get('thalium_stress_test'))
    [[age,bp,cholesterol,st_depression]] = scaler.transform([[age,bp,cholesterol,st_depression]])
    prediction = model.predict([[age,sex,chest_pain,bp,cholesterol,fasting_blood_sugar,
                                resting_ecg,exercise_angina,num_vessels,thalium_stress_test]])
    if prediction == 0:
        text = "Absence of Heart Desease"
    elif prediction == 1:
        text = 'Presence of Heart Disease'
    return render_template('prediction.html',prediction_text = text)



if __name__ == '__main__':
    app.run(debug=True)