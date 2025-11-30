import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Cargar y preprocesar datos ---
def load_and_preprocess_data(path='train.csv'):
    df = pd.read_csv(path)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y, le_sex, le_embarked

# --- Entrenar y evaluar Árbol de Decisión ---
def train_and_evaluate_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'criterion': ['gini', 'entropy'],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy'
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results = {
        'model': best_model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'report': report,
        'best_params': grid.best_params_,
    }
    return results

# --- Explicación local del árbol de decisión ---
def explain_decision_tree(model, feature_names, x):
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    node_indicator = model.decision_path(x)
    node_index = node_indicator.indices
    explanation = []
    for node_id in node_index:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            break
        fname = feature_names[feature[node_id]]
        thresh = threshold[node_id]
        val = x[0, feature[node_id]]
        if val <= thresh:
            explanation.append(f"{fname} <= {thresh:.2f} (valor: {val:.2f}) → Sí")
        else:
            explanation.append(f"{fname} <= {thresh:.2f} (valor: {val:.2f}) → No")
    leaf_id = model.apply(x)[0]
    value = tree.value[leaf_id]
    pred_class = np.argmax(value)
    explanation.append(f"Clase predicha: {pred_class} ('SOBREVIVIÓ' si es 1, 'NO SOBREVIVIÓ' si es 0)")
    return "\n".join(explanation)

# --- Interfaz gráfica solo para Árbol de Decisión ---
class TitanicDecisionTreeApp:
    def __init__(self, root, results, le_sex, le_embarked, feature_names):
        self.root = root
        self.results = results
        self.le_sex = le_sex
        self.le_embarked = le_embarked
        self.feature_names = feature_names
        self.root.title("Titanic - Árbol de Decisión")

        # TAB MÉTRICAS
        metrics_frame = ttk.LabelFrame(root, text="Métricas - Árbol de Decisión")
        metrics_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        # Texto para mostrar métricas
        self.metrics_text = tk.Text(metrics_frame, width=75, height=12, font=('Consolas', 10))
        self.metrics_text.grid(row=0, column=0, padx=5, pady=5)
        self.metrics_text.insert(tk.END, "Mejores hiperparámetros encontrados:\n")
        self.metrics_text.insert(tk.END, f"{self.results['best_params']}\n\n")
        self.metrics_text.insert(tk.END, "=== Reporte de métricas ===\n")
        self.metrics_text.insert(tk.END, f"Accuracy: {self.results['accuracy']:.4f}\n")
        self.metrics_text.insert(tk.END, f"Precisión: {self.results['precision']:.4f}\n")
        self.metrics_text.insert(tk.END, f"Recall: {self.results['recall']:.4f}\n")
        self.metrics_text.insert(tk.END, f"F1-Score: {self.results['f1_score']:.4f}\n\n")
        self.metrics_text.insert(tk.END, self.results['report'])

        # TAB DE PREDICCIÓN
        predict_frame = ttk.LabelFrame(root, text="Predicción y Explicación Local")
        predict_frame.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

        # Variables del formulario
        self.pclass_var = tk.IntVar(value=3)
        self.sex_var = tk.StringVar(value='male')
        self.age_var = tk.DoubleVar(value=30)
        self.sibsp_var = tk.IntVar(value=0)
        self.parch_var = tk.IntVar(value=0)
        self.fare_var = tk.DoubleVar(value=32.2)
        self.embarked_var = tk.StringVar(value='S')

        labels = [
            'Tipo de Ticket (1,2,3):', 'Sexo (hombre/mujer):', 'Edad:', 'N° de Hermanos/Pareja a bordo:', 
            'N° de Padres/Hijos a bordo:', 'Tarifa pagada:', 'Embarcado en (Cherbourg/Queenstown/Southampton):'
        ]
        vars_ = [
            self.pclass_var, self.sex_var, self.age_var, self.sibsp_var,
            self.parch_var, self.fare_var, self.embarked_var
        ]
        for i, (label, var) in enumerate(zip(labels, vars_)):
            ttk.Label(predict_frame, text=label).grid(row=i, column=0, sticky='e')
            if isinstance(var, tk.StringVar):
                if label.startswith('Sex'):
                    entry = ttk.Combobox(
                        predict_frame, textvariable=var,
                        values=['male', 'female'], state='readonly'
                    )
                elif label.startswith('Embarked'):
                    entry = ttk.Combobox(
                        predict_frame, textvariable=var,
                        values=['C', 'Q', 'S'], state='readonly'
                    )
                else:
                    entry = ttk.Entry(predict_frame, textvariable=var)
            else:
                entry = ttk.Entry(predict_frame, textvariable=var)
            entry.grid(row=i, column=1)

        ttk.Button(
            predict_frame, text="Predecir y Explicar",
            command=self.predict_case
        ).grid(row=len(labels), column=0, columnspan=2, pady=10)
        self.prediction_text = scrolledtext.ScrolledText(
            predict_frame, width=80, height=15, font=('Consolas', 10)
        )
        self.prediction_text.grid(row=len(labels)+1, column=0, columnspan=2, padx=5, pady=5)

    def predict_case(self):
        try:
            pclass = int(self.pclass_var.get())
            sex = self.sex_var.get()
            age = float(self.age_var.get())
            sibsp = int(self.sibsp_var.get())
            parch = int(self.parch_var.get())
            fare = float(self.fare_var.get())
            embarked = self.embarked_var.get()
        except Exception:
            messagebox.showerror("Error", "Verifica que los datos sean válidos.")
            return

        # Codifica sex y embarked igual que en el preprocesamiento
        try:
            sex_enc = self.le_sex.transform([sex])[0]
        except Exception:
            messagebox.showerror("Error", "Valor de 'Sex' no reconocido.")
            return
        try:
            embarked_enc = self.le_embarked.transform([embarked])[0]
        except Exception:
            messagebox.showerror("Error", "Valor de 'Embarked' no reconocido.")
            return

        X_input = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])

        self.prediction_text.delete(1.0, tk.END)
        model = self.results['model']
        explanation = explain_decision_tree(model, self.feature_names, X_input)
        pred = model.predict(X_input)[0]
        pred_str = "SOBREVIVIÓ" if pred == 1 else "NO SOBREVIVIÓ"
        self.prediction_text.insert(tk.END, f"Predicción: {pred_str}\n")
        self.prediction_text.insert(tk.END, "Explicación local de la predicción:\n")
        self.prediction_text.insert(tk.END, explanation + '\n')

def main():
    X, y, le_sex, le_embarked = load_and_preprocess_data('train.csv')
    feature_names = X.columns.tolist()
    results = train_and_evaluate_decision_tree(X, y)
    root = tk.Tk()
    app = TitanicDecisionTreeApp(root, results, le_sex, le_embarked, feature_names)
    root.mainloop()

if __name__ == "__main__":
    main()