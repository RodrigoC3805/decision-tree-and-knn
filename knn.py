import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar datos Iris (solo las dos primeras variables)
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
feature_names = iris.feature_names[:2]
class_names = iris.target_names

# Partición train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class IrisSepalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KNN Iris (Sepal length & width)")
        self.entries = {}
        for i, name in enumerate(feature_names):
            tk.Label(self, text=name).grid(row=i, column=0)
            self.entries[name] = tk.Entry(self)
            self.entries[name].grid(row=i, column=1)

        # Campo para elegir K
        tk.Label(self, text="Valor de K (vecinos):").grid(row=len(feature_names), column=0)
        self.k_entry = tk.Spinbox(self, from_=1, to=30, width=5)
        self.k_entry.grid(row=len(feature_names), column=1)

        tk.Button(self, text="Predecir", command=self.predecir).grid(row=len(feature_names)+1, columnspan=2)
        self.resultado = tk.Label(self, text="")
        self.resultado.grid(row=len(feature_names)+2, columnspan=2)
        self.tabla = tk.Frame(self)
        self.tabla.grid(row=len(feature_names)+3, columnspan=2)

    def predecir(self):
        try:
            X_new = [float(self.entries[n].get()) for n in feature_names]
            K = int(self.k_entry.get())
            if K < 1 or K > len(X_train):
                raise ValueError(f"K debe estar entre 1 y {len(X_train)}")
            X_new_scaled = scaler.transform(np.array(X_new).reshape(1, -1))

            modelo = KNeighborsClassifier(n_neighbors=K, metric='euclidean', weights='uniform')
            modelo.fit(X_train_scaled, y_train)

            pred = modelo.predict(X_new_scaled)[0]
            prob = modelo.predict_proba(X_new_scaled)[0][pred]
            self.resultado.config(text=f"Predicción: {class_names[pred]} con confianza {prob:.2f} (K={K})")
            self.visualizar_vecinos(modelo, X_new_scaled, X_new, pred)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def visualizar_vecinos(self, modelo, X_new_scaled, X_new_raw, pred_clase):
        # Limpia tabla previa
        for widget in self.tabla.winfo_children():
            widget.destroy()
        idxs = modelo.kneighbors(X_new_scaled, return_distance=False)[0]
        tk.Label(self.tabla, text="K vecinos más cercanos:").grid(row=0, columnspan=3)
        for j, name in enumerate(feature_names+['Clase']):
            tk.Label(self.tabla, text=name, bg="lightblue").grid(row=1, column=j)
        for i, idx in enumerate(idxs):
            for j, name in enumerate(feature_names):
                val = X_train[idx][j]
                tk.Label(self.tabla, text=f"{val:.2f}", relief="solid").grid(row=i+2, column=j)
            tk.Label(self.tabla, text=class_names[y_train[idx]], relief="solid").grid(row=i+2, column=len(feature_names))
        # Visualización gráfica 2D
        plt.figure(figsize=(6, 5))
        markers = ["o", "s", "D"]
        colors = ["royalblue", "orange", "limegreen"]
        for clase in np.unique(y_train):
            plt.scatter(X_train[y_train==clase][:,0], X_train[y_train==clase][:,1], 
                        label=class_names[clase], alpha=0.6, marker=markers[clase], color=colors[clase])
        # Vecinos
        plt.scatter([X_train[idx][0] for idx in idxs], [X_train[idx][1] for idx in idxs], 
                    c='red', marker='x', s=100, label='Vecinos')
        # Punto de predicción
        plt.scatter(X_new_raw[0], X_new_raw[1], c='black', marker='*', s=180, label='Predicción')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f"Vecinos y predicción ({class_names[pred_clase]})")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    app = IrisSepalApp()
    app.mainloop()