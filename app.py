import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os


app = Flask(__name__)

# Load models
model_names = ['support_vector_machine', 'k-nearest_neighbors', 'decision_tree', 'logistic_regression', 'random_forest']
models = {name: pickle.load(open(f'models/{name}_model.pkl', 'rb')) for name in model_names}

# Load and scale data
df = pd.read_csv('cardio_train.csv', delimiter=';')
df['age'] = df['age'] // 365
X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

scaler = StandardScaler().fit(X)

def plot_roc_curves(X_test, y_test):
    valid_indices = ~np.isnan(y_test)
    y_test = y_test[valid_indices]
    X_test = X_test[valid_indices]

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name.title()} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('static', 'roc_curve.png'))
    plt.close()


def plot_confusion_matrix_custom(y_true, y_pred):
    # Filter out NaN values
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('static', 'confusion_matrix.png'))
    plt.close()




@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        form_data = request.form
        age = int(form_data['age']) * 365
        gender = int(form_data['gender'])
        height = float(form_data['height'])
        weight = float(form_data['weight'])
        ap_hi = int(form_data['ap_hi'])
        ap_lo = int(form_data['ap_lo'])
        cholesterol = int(form_data['cholesterol'])
        gluc = int(form_data['gluc'])
        smoke = int(form_data['smoke'])
        alco = int(form_data['alco'])
        active = int(form_data['active'])

        input_data = pd.DataFrame([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]],
                                  columns=X.columns)
        input_data_scaled = scaler.transform(input_data)

        results = {}
        for name, model in models.items():
            prediction_prob = model.predict_proba(input_data_scaled)[0][1]
            prediction = 1 if prediction_prob > 0.5 else 0
            result = "Positive" if prediction == 1 else "Negative"
            results[name.title()] = {'result': result, 'probability': prediction_prob}

        # Plot ROC curve and Confusion Matrix
        X_test = scaler.transform(X)
        plot_roc_curves(X_test, y)
        plot_confusion_matrix_custom(y, models['logistic_regression'].predict(X_test))

        return render_template('results.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
