import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# CSV-bestand inlezen
data = pd.read_csv('Ocean_Health_Index_2018_global_scores.csv')

# Beschikbare features
features = ['CS', 'HAB', 'BD', 'ICO', 'CP', 'AO', 'FP', 'TR', 'MAR', 'FIS', 'NP', 'LIV', 'CW']

# Tkinter venster
root = tk.Tk()
root.title("Attribuut Voorspeller")
root.geometry("1200x800")

# Functie om model te trainen en plots te tonen
def run_model():
    target = target_var.get()
    X_features = [f for f in features if f != target]
    df = data.dropna(subset=X_features + [target])
    X = df[X_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    score_label.config(text=f"R² score voor {target}: {score:.3f}")

    voorspellingen = model.predict(X_test)
    X_test = X_test.copy()
    X_test[f'Predicted_{target}'] = voorspellingen
    X_test[f'Echte_{target}'] = y_test
    #if 'rgn_nam' in X_test.columns:
     #   X_test['Region Name'] = X_test['rgn_nam']
    #else:
     #   X_test['Region Name'] = 'Onbekend'

    # Voeg echte regio namen toe vanuit de originele dataset
    X_test = X_test.copy()
    X_test[f'Predicted_{target}'] = voorspellingen
    X_test[f'Echte_{target}'] = y_test

    # Gebruik de originele data voor de regiënamen
    X_test['Region Name'] = data.loc[X_test.index, 'rgn_nam']

    # Clear previous frames
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Scatter plot
    fig1, ax1 = plt.subplots(figsize=(6,4))


    #ax1.scatter(X_test[f'Echte_{target}'], X_test[f'Predicted_{target}'], alpha=0.6)
    # Filter rijen waarbij de echte waarde niet 0 is
    plot_data = X_test[X_test[f'Echte_{target}'] != 0]

    ax1.scatter(plot_data[f'Echte_{target}'], plot_data[f'Predicted_{target}'], alpha=0.6)

    # Bereken afwijking op gefilterde data
    plot_data['afwijking'] = np.abs(plot_data[f'Echte_{target}'] - plot_data[f'Predicted_{target}'])
    top_afwijking = plot_data.nlargest(10, 'afwijking')
    for i, row in top_afwijking.iterrows():
        ax1.text(row[f'Echte_{target}']+0.5, row[f'Predicted_{target}']+0.5, row['Region Name'], fontsize=8)



    X_test['afwijking'] = np.abs(X_test[f'Echte_{target}'] - X_test[f'Predicted_{target}'])
    top_afwijking = X_test.nlargest(10, 'afwijking')
    for i, row in top_afwijking.iterrows():
        ax1.text(row[f'Echte_{target}']+0.5, row[f'Predicted_{target}']+0.5, row['Region Name'], fontsize=8)
    ax1.set_xlabel(f"Echte {target} Score")
    ax1.set_ylabel(f"Voorspelde {target} Score")
    ax1.set_title(f"Voorspelde vs Echte {target} Score")
    ax1.plot([0,100],[0,100], color='red', linestyle='--')

    canvas1 = FigureCanvasTkAgg(fig1, master=plot_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Feature importances
    importances = model.feature_importances_
    fig2, ax2 = plt.subplots(figsize=(6,4))
    #ax2.bar(X_features, importances)
    # Genereer een unieke kleur voor elke feature
    colors = plt.cm.tab20.colors  # een colormap met 20 verschillende kleuren
    # Herhaal de kleuren als er meer features zijn dan kleuren
    colors = colors * ((len(X_features) // len(colors)) + 1)
    ax2.bar(X_features, importances, color=colors[:len(X_features)])



    ax2.set_title(f"Belangrijkste factoren voor {target}")
    ax2.set_ylabel("Feature Importance")
    ax2.set_xticklabels(X_features, rotation=90, fontsize=8)

    canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


# Dropdownmenu
tk.Label(root, text="Selecteer attribuut om te voorspellen:").pack(pady=10)
target_var = tk.StringVar(value=features[0])
dropdown = ttk.Combobox(root, textvariable=target_var, values=features, state="readonly", width=50)
dropdown.pack(pady=5)

# Run knop
tk.Button(root, text="Voorspel", command=run_model).pack(pady=10)

# Label voor model score
score_label = tk.Label(root, text="")
score_label.pack(pady=5)

# Frame voor plots
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()