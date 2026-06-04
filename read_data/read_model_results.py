import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# 1. הגדרת הנתיב וטעינת המודל השמור
model_dir = r"/Users/deviceone/Documents/d_detection/models"
model_out_path = os.path.join(model_dir, "2s_model_omesi.joblib")

loaded_model = joblib.load(model_out_path)
print("המודל נטען בהצלחה מתוך קובץ ה-Joblib!")

# 2. שחזור נתוני הבדיקה (ודא ששלב זה רץ על אותם נתונים בדיוק)
# X, y = prepare_data(BASE_DATA_DIR, binary_map)
# _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def reconstruct_precision_recall(model, X_test, y_test, target_class_index=1):
    # הפקת ההסתברויות מהמודל הנטען
    y_probs = model.predict_proba(X_test)[:, target_class_index]
    
    # חישוב ה-Precision וה-Recall עבור ספים (thresholds) שונים
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    
    # ציור הגרף מחדש
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall (Detection Rate)')
    plt.ylabel('Precision (1 - False Alarm Rate)')
    plt.title('Reconstructed Precision-Recall Curve')
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.show()
    
    return precision, recall, thresholds

# 3. הפעלת הפונקציה המשחזרת
precision, recall, thresholds = reconstruct_precision_recall(loaded_model, X_test, y_test)