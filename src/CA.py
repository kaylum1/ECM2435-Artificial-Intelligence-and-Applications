#final CA coursework

#all imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Load and preprocess the dataset

#imported needed for the file path of excel sheet
file_path = 'CCD.xls'
#disragarding the first row as its not relevent for AI
data = pd.read_excel(file_path, skiprows=1)

#Separating the features and target variable from the dataset
X = data.drop(columns=['default payment next month']).values
Y = data['default payment next month'].values

# Normalize input features so can standardize the dataset 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets with a test size of 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#deep neural network model, defining activation functions and dropouts
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
])

# Compiling the model and paramters needed
model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping and pactience set
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model and hyperparemters
history = model.fit(X_train, Y_train, epochs=50, batch_size=256, validation_split=0.2, verbose=1, callbacks=[early_stopping])




# to evaluate the model on the test set for loss and accuracy 
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# can use use plt.ylimit to make the graph smaller


# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Generate Confusion Matrix for the cassification performance
Y_pred = model.predict(X_test)
Y_pred_classes = (Y_pred > 0.5).astype("int32")
conf_matrix = confusion_matrix(Y_test, Y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Generate Classification Report
print("Classification Report:")
print(classification_report(Y_test, Y_pred_classes))

# Generate ROC Curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Generate Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()