from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

# Load the pre-trained model from a pickle file
with open('Models/genderclassification.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict_gender', methods=['GET'])
def predict_gender():
    long_hair = int(request.args.get('long_hair'))
    fore_head_width = float(request.args.get('fore_head_width'))
    fore_head_length = float(request.args.get('fore_head_length'))
    nose_wide = int(request.args.get('nose_wide'))
    nose_long = int(request.args.get('nose_long'))
    lips_thin = int(request.args.get('lips_thin'))
    distance_nose_to_lip_long = int(request.args.get('distance_nose_to_lip_long'))
    
    y_label = model.predict([[long_hair,fore_head_width,fore_head_length,nose_wide,nose_long,
                   lips_thin,distance_nose_to_lip_long]])
    
    
    if y_label == 0:
        prediction_text = "Male"
    else:
        prediction_text = "Female"

    return render_template("prediction.html",prediction_text = prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
