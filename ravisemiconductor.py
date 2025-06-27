# Write a python program to develop and evaluate a hybrid AI system using Random Forest and LSTM models for real-time classification and prediction of power quality disturbances (such as voltage sags and swells), including model interpretation with SHAP analysis and performance benchmarking through execution time and model persistence.
# ===============================
# Install Required Libraries
# ===============================
!pip install imbalanced-learn shap joblib keras --quiet

# ===============================
# Import Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical

# ===============================
# Step 1: Simulated PQ Dataset
# ===============================
data = {
    'Voltage': [220, 230, 210, 225, 235, 240, 220, 215, 230, 240, 245, 210, 220, 225, 235, 240, 230, 220, 225],
    'Current': [5.2, 5.5, 5.0, 5.3, 5.7, 6.0, 5.2, 5.1, 5.5, 5.9, 6.1, 4.9, 5.2, 5.3, 5.6, 5.8, 5.7, 5.3, 5.4],
    'Harmonics': [3.5, 2.8, 4.2, 3.0, 2.5, 1.8, 3.6, 4.0, 2.9, 2.1, 1.5, 4.5, 3.4, 3.1, 2.7, 2.0, 2.6, 3.3, 3.1],
    'Frequency': [50, 49.8, 50.2, 49.9, 50.1, 50, 49.7, 50.3, 49.9, 50.1, 50.2, 49.6, 50, 49.8, 50, 50.1, 49.8, 50, 49.9],
    'Power Factor': [0.95, 0.96, 0.92, 0.94, 0.97, 0.95, 0.95, 0.93, 0.96, 0.99, 1.0, 0.91, 0.94, 0.95, 0.98, 0.99, 0.95, 0.96, 0.97],
    'Temperature': [25, 27, 24, 26, 28, 29, 25, 26, 27, 30, 31, 23, 25, 26, 28, 29, 27, 26, 27],
    'Transient': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    'Class': ['Normal', 'Sag', 'Normal', 'Normal', 'Normal', 'Normal', 'Sag', 'Normal', 'Normal', 'Swell',
              'Normal', 'Sag', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Swell']
}
df = pd.DataFrame(data)
df['Class'] = df['Class'].map({'Normal': 0, 'Sag': 1, 'Swell': 2})

# ===============================
# Step 2: Preprocessing
# ===============================
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine safe k_neighbors
class_counts = Counter(y)
min_class_count = min(class_counts.values())
safe_k = min(5, min_class_count - 1)

smote = SMOTE(random_state=42, k_neighbors=safe_k)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ===============================
# Step 3: Random Forest + Time
# ===============================
start_rf = time.time()
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_rf

y_pred_rf = rf_model.predict(X_test)
print("\n=== Random Forest Results ===")
print(classification_report(y_test, y_pred_rf))
print(f"RF Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"RF Execution Time: {rf_time:.4f} seconds")

# ===============================
# Step 4: LSTM + Time
# ===============================
X_lstm = np.reshape(X_resampled, (X_resampled.shape[0], 1, X_resampled.shape[1]))
y_lstm = to_categorical(y_resampled)

lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(1, X_resampled.shape[1])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(3, activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_lstm = time.time()
history = lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=4, verbose=0)
lstm_time = time.time() - start_lstm

print("\n=== LSTM Results ===")
print(f"LSTM Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"LSTM Execution Time: {lstm_time:.2f} seconds")

# ===============================
# Step 5: SHAP Analysis
# ===============================
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar")

# ===============================
# Step 6: Visualizations
# ===============================
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='LSTM Accuracy')
plt.title('LSTM Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Sag', 'Swell'], yticklabels=['Normal', 'Sag', 'Swell'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ===============================
# Step 7: Save Models
# ===============================
joblib.dump(rf_model, 'random_forest_model.pkl')
lstm_model.save('lstm_model.h5')
print("\nModels saved as 'random_forest_model.pkl' and 'lstm_model.h5'")