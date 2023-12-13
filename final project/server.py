from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
with open("D:\\final project\\deployment\\heartrate\\Scripts\\model.pkl", 'rb') as model_file:
    pipeline = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        MEAN_RR = float(request.form['MEAN_RR'])
        MEDIAN_RR = float(request.form['MEDIAN_RR'])
        SDRR = float(request.form['SDRR'])
        RMSSD = float(request.form['RMSSD'])
        SDSD = float(request.form['SDSD'])
        SDRR_RMSSD = float(request.form['SDRR_RMSSD'])
        pNN25 = float(request.form['pNN25'])
        pNN50 = float(request.form['pNN50'])
        KURT = float(request.form['KURT'])
        SKEW = float(request.form['SKEW'])
        MEAN_REL_RR = float(request.form['MEAN_REL_RR'])
        MEDIAN_REL_RR = float(request.form['MEDIAN_REL_RR'])
        SDRR_REL_RR = float(request.form['SDRR_REL_RR'])
        RMSSD_REL_RR = float(request.form['RMSSD_REL_RR'])
        SDSD_REL_RR = float(request.form['SDSD_REL_RR'])
        SDRR_RMSSD_REL_RR = float(request.form['SDRR_RMSSD_REL_RR'])
        KURT_REL_RR = float(request.form['KURT_REL_RR'])
        SKEW_REL_RR = float(request.form['SKEW_REL_RR'])
        VLF = float(request.form['VLF'])
        VLF_PCT = float(request.form['VLF_PCT'])
        LF = float(request.form['LF'])
        LF_PCT = float(request.form['LF_PCT'])
        LF_NU = float(request.form['LF_NU'])
        HF = float(request.form['HF'])
        HF_PCT = float(request.form['HF_PCT'])
        HF_NU = float(request.form['HF_NU'])
        TP = float(request.form['TP'])
        LF_HF = float(request.form['LF_HF'])
        HF_LF = float(request.form['HF_LF'])
        SD1 = float(request.form['SD1'])
        SD2 = float(request.form['SD2'])
        sampen = float(request.form['sampen'])
        higuci = float(request.form['higuci'])
        condition = float(request.form['condition'])

        # Create a DataFrame from user-entered features
        user_features = pd.DataFrame({
            'MEAN_RR': [MEAN_RR],
            'MEDIAN_RR': [MEDIAN_RR],
            'SDRR': [SDRR],
            'RMSSD': [RMSSD],
            'SDSD': [SDSD],
            'SDRR_RMSSD': [SDRR_RMSSD],
            'pNN25': [pNN25],
            'pNN50': [pNN50],
            'KURT': [KURT],
            'SKEW': [SKEW],
            'MEAN_REL_RR': [MEAN_REL_RR],
            'MEDIAN_REL_RR': [MEDIAN_REL_RR],
            'SDRR_REL_RR': [SDRR_REL_RR],
            'RMSSD_REL_RR': [RMSSD_REL_RR],
            'SDSD_REL_RR': [SDSD_REL_RR],
            'SDRR_RMSSD_REL_RR': [SDRR_RMSSD_REL_RR],
            'KURT_REL_RR': [KURT_REL_RR],
            'SKEW_REL_RR': [SKEW_REL_RR],
            'VLF': [VLF],
            'VLF_PCT': [VLF_PCT],
            'LF': [LF],
            'LF_PCT': [LF_PCT],
            'LF_NU': [LF_NU],
            'HF': [HF],
            'HF_PCT': [HF_PCT],
            'HF_NU': [HF_NU],
            'TP': [TP],
            'LF_HF': [LF_HF],
            'HF_LF': [HF_LF],
            'SD1': [SD1],
            'SD2': [SD2],
            'sampen': [sampen],
            'higuci': [higuci],
            'condition': [condition]
        })

        # Use the trained pipeline to make predictions
        prediction = pipeline.predict(user_features)

        # Redirect to the result page with the prediction value
        return render_template('index.html', prediction=f'Predicted HR: {prediction[0]:.2f}')
    except ValueError as e:
        error_message = "An error occurred: Please enter a valid number."
        return render_template('index.html', error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
