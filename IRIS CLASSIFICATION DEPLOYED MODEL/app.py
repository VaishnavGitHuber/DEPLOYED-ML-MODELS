from flask import Flask,render_template,request
import pickle
# saving the model 
with open('pickle_files/min_max_scaler.pkl','rb') as file:
    scaler = pickle.load(file)
with open('pickle_files/model.pkl','rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict',methods=["GET"])
def predict_page():
    petal_width = float(request.args.get('petal_width'))
    values = [[petal_width]]
    values = scaler.transform(values)
    predicted_value = model.predict(values)
    if predicted_value == 1:
        predicted_text = "IRIS SETOSA"
    elif predicted_value == 2:
        predicted_text = "IRIS VERSICOLOR"
    else:
        predicted_text = 'IRIS VERGINICA'

    return render_template('index.html',predicted_text=predicted_text)

if __name__ == "__main__":
    app.run(debug=True)