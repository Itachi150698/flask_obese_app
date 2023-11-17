import joblib
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np



app = Flask(__name__)

# Load the model using joblib
model = pickle.load(open('cancer.pkl', 'rb'))
model1 = joblib.load(open('heart.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/help')
def help():
    return render_template("help.html")


@app.route('/terms')
def terms():
    return render_template("tc.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/disindex")
def disindex():
    return render_template("disindex.html")


@app.route("/cancer")
def cancer():
    return render_template("cancer.html")





@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")


############################################################################################################################################################

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    result = None  # Set a default value for result
    if size == 7:
        loaded_model = joblib.load('kidney.pkl')
        result = loaded_model.predict(to_predict)
    return result[0] if result is not None else None

@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    result = None  # Set a default value for result
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)

    if result is not None and int(result) == 1:
        prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"

    # Generate a pandas report
    report_data = {'Age': to_predict_list[0],
                   'Blood Pressure': to_predict_list[1],
                   'Specific Gravity': to_predict_list[2],
                   'Albumin': to_predict_list[3],
                   'Blood Sugar Level': to_predict_list[4],
                   'Red Blood Cells Count': to_predict_list[5],
                   'Pus Cell Count': to_predict_list[6],
                   'Pus Cell Clumps': to_predict_list[7],
                   'Prediction': prediction}

    report_df = pd.DataFrame([report_data])

    # Save the report to a file (you can choose a format that suits your needs)
    report_df.to_csv('prediction_report.csv', index=False)

    # Pass the prediction and report to the template
    return render_template("kidney_result.html", prediction_text=prediction, report=report_df.to_html())

############################################################################################################################################################

def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    result = None  # Initialize with a default value
    if size == 7:
        loaded_model = joblib.load('liver.pkl')
        result = loaded_model.predict(to_predict)
    return result[0] if result is not None else None

@app.route('/predictliver', methods=["POST"])
def predictliver():
    result = None  # Initialize with a default value
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

    if result is not None and int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Liver Disease"

    # Generate a pandas report
    report_data = {'Age': to_predict_list[0],
                   'Gender': to_predict_list[1],
                   'Total Bilirubin': to_predict_list[2],
                   'Direct Bilirubin': to_predict_list[3],
                   'Alkaline Phosphotase': to_predict_list[4],
                   'Alamine Aminotransferase': to_predict_list[5],
                   'Aspartate Aminotransferase': to_predict_list[6],
                   'Total Protiens': to_predict_list[7],
                   'Albumin': to_predict_list[8],
                   'Albumin and Globulin Ratio': to_predict_list[9],
                   'Prediction': prediction}

    report_df = pd.DataFrame([report_data])

    # Save the report to a file (you can choose a format that suits your needs)
    report_df.to_csv('liver_prediction_report.csv', index=False)

    # Pass the prediction and report to the template
    return render_template("liver_result.html", prediction_text=prediction, report=report_df.to_html())

####################################################################################################################################################


@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    input_features = [int(x) for x in request.form.values()]

    # Create a DataFrame for the input features
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)

    # Predict using the loaded model
    output = model.predict(df)

    # Create a pandas report
    report_data = {'Feature Name': features_name, 'Feature Value': input_features}
    report_df = pd.DataFrame(report_data)

    # Save the report to a CSV file (you can choose a format that suits your needs)
    report_df.to_csv('breast_cancer_prediction_report.csv', index=False)

    # Interpret the prediction result
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    # Pass the prediction and report to the template
    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val), report=report_df.to_html(index=False))


#########################################################################################################################################################################

@app.route('/predictheart', methods=['POST'])
def predictheart():
    # Get input features from the form
    input_features = [float(x) for x in request.form.values()]

    # Create a DataFrame from the input features
    features_name = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak",
                     "slope", "ca", "thal"
                     ]
    df = pd.DataFrame([input_features], columns=features_name)

    # Predict using the loaded model
    output = model1.predict(df)

    # Create a pandas report
    report_data = {'Feature Name': features_name, 'Feature Value': input_features}
    report_df = pd.DataFrame(report_data)

    # Save the report to a CSV file (you can choose a format that suits your needs)
    report_df.to_csv('heart_disease_prediction_report.csv', index=False)

    # Interpret the prediction result
    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    # Pass the prediction and report to the template
    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val), report=report_df.to_html(index=False))



#########################################################################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)


