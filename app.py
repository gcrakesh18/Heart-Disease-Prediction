from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import webbrowser
# Set Matplotlib to use a non-interactive backend before importing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - MUST be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import threading
import json
import io
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
import tensorflow as tf

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Path for the model
model_path = 'app/models/heart_disease_model.h5'
scaler_path = 'app/models/scaler.pkl'
data_path = 'app/data/heart.csv'
metrics_path = 'app/models/metrics.json'
training_status_path = 'app/models/training_status.json'

# Global variables to track training status
training_thread = None
training_progress = 0
training_status = "idle"
training_message = ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': 'Model not trained yet. Please train the model first.'
            })
        
        # Get form data and convert to numpy array
        data = []
        for field in request.form:
            data.append(float(request.form[field]))
        
        # Load model and scaler
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            import pickle
            scaler = pickle.load(f)
        
        # Scale data
        data = scaler.transform(np.array(data).reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(data)
        result = (prediction > 0.5).astype(int)[0][0]
        probability = float(prediction[0][0])
        
        # Return prediction
        return jsonify({
            'status': 'success',
            'prediction': int(result),
            'probability': float(probability),
            'message': 'Heart disease detected.' if result == 1 else 'No heart disease detected.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

def generate_plot_base64(fig):
    """Generate base64 encoded PNG from matplotlib figure"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Convert to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    # Close figure to prevent memory leaks
    plt.close(fig)
    
    return img_base64

def save_training_status(status, message, progress=0):
    """Save training status to file"""
    status_data = {
        'status': status,
        'message': message,
        'progress': progress
    }
    with open(training_status_path, 'w') as f:
        json.dump(status_data, f)

def train_model_process():
    """Process to train the model in a separate thread"""
    global training_progress, training_status, training_message
    
    try:
        save_training_status('running', 'Loading dataset...', 5)
        
        # Load and preprocess data
        data = pd.read_csv(data_path)
        
        save_training_status('running', 'Preprocessing data...', 10)
        
        # Check if 'target' column exists, otherwise assume it's the last column
        if 'target' in data.columns:
            X = data.drop('target', axis=1).values
            y = data['target'].values
        else:
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        save_training_status('running', 'Scaling features...', 20)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        save_training_status('running', 'Building model...', 30)
        
        # Build model
        model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        save_training_status('running', 'Training model...', 40)
        
        # Create a callback to track training progress
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                progress = 40 + int((epoch / 50) * 40)  # 40% to 80% during training
                save_training_status('running', f'Training epoch {epoch+1}/50...', progress)
                
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[ProgressCallback()]
        )
        
        save_training_status('running', 'Evaluating model...', 80)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        
        # Save model
        model.save(model_path)
        
        save_training_status('running', 'Generating performance metrics...', 85)
        
        # Generate confusion matrix
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        cm = confusion_matrix(y_test, y_pred)
        
        # Create confusion matrix plot
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Confusion Matrix')
        confusion_matrix_image = generate_plot_base64(fig_cm)
        
        # Generate ROC curve
        save_training_status('running', 'Generating ROC curve...', 90)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc='lower right')
        roc_curve_image = generate_plot_base64(fig_roc)
        
        # Generate learning curve
        save_training_status('running', 'Generating learning curve...', 92)
        fig_lc, ax_lc = plt.subplots(figsize=(10, 6))
        ax_lc.plot(history.history['accuracy'], label='Training Accuracy')
        ax_lc.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax_lc.set_xlabel('Epoch')
        ax_lc.set_ylabel('Accuracy')
        ax_lc.set_title('Learning Curve')
        ax_lc.legend()
        learning_curve_image = generate_plot_base64(fig_lc)
        
        # Generate feature correlation plot
        save_training_status('running', 'Generating feature analysis...', 95)
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title('Feature Correlation Matrix')
        feature_correlation_image = generate_plot_base64(fig_corr)
        
        # Generate feature distributions
        fig_dist, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(data.columns):
            if i < len(axes):
                if col != 'target':
                    sns.histplot(data=data, x=col, hue='target' if 'target' in data.columns else data.columns[-1], 
                               kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
        
        # Hide any unused axes
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        feature_distributions_image = generate_plot_base64(fig_dist)
        
        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Prepare metrics
        metrics = {
            'train_accuracy': float(train_acc * 100),
            'test_accuracy': float(test_acc * 100),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'plots': {
                'confusion_matrix_image': confusion_matrix_image,
                'roc_curve_image': roc_curve_image,
                'learning_curve_image': learning_curve_image,
                'feature_correlation_image': feature_correlation_image,
                'feature_distributions_image': feature_distributions_image
            }
        }
        
        # Save metrics to file
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        # Update status to completed
        save_training_status('completed', 'Training complete', 100)
        
        # Add accuracy to status
        with open(training_status_path, 'r') as f:
            status_data = json.load(f)
        
        status_data['test_accuracy'] = f"{test_acc * 100:.2f}"
        
        with open(training_status_path, 'w') as f:
            json.dump(status_data, f)
            
    except Exception as e:
        # Update status to error
        save_training_status('error', f'Error during training: {str(e)}', 0)
        print(f"Training error: {str(e)}")

@app.route('/train', methods=['GET'])
def train():
    global training_thread
    
    try:
        # Check if training is already in progress
        if training_thread and training_thread.is_alive():
            return jsonify({
                'status': 'error',
                'message': 'Training is already in progress.'
            })
        
        # Initialize training status
        save_training_status('starting', 'Initializing training...', 0)
        
        # Start training in a separate thread
        training_thread = threading.Thread(target=train_model_process)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Model training started.'
        })
    except Exception as e:
        save_training_status('error', f'Error starting training: {str(e)}', 0)
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/training_status', methods=['GET'])
def get_training_status():
    try:
        if os.path.exists(training_status_path):
            with open(training_status_path, 'r') as f:
                status_data = json.load(f)
            return jsonify(status_data)
        else:
            return jsonify({
                'status': 'idle',
                'message': 'No training has been initiated.',
                'progress': 0
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error checking training status: {str(e)}',
            'progress': 0
        })

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        # Check if metrics file exists
        if not os.path.exists(metrics_path):
            return render_template('metrics.html', metrics=None, error="No metrics available. Please train the model first.")
        
        # Load metrics from file
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return render_template('metrics.html', metrics=metrics)
    except Exception as e:
        return render_template('metrics.html', metrics=None, error=str(e))

@app.route('/reset_metrics', methods=['GET'])
def reset_metrics():
    try:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
            return jsonify({
                'status': 'success',
                'message': 'Metrics file deleted successfully.'
            })
        else:
            return jsonify({
                'status': 'info',
                'message': 'No metrics file found.'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error deleting metrics file: {str(e)}'
        })

if __name__ == '__main__':
    # Create training status file if it doesn't exist
    save_training_status('idle', 'No training initiated.', 0)
    
    # Run Flask app
    app.run(debug=True) 