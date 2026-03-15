from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Chargement des données
iris = load_iris()
X, y = iris.data, iris.target

# Séparation en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Évaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Précision du modèle : {accuracy:.2%}")

# Sauvegarde du modèle dans un fichier
joblib.dump(model, "model.pkl")
print("Modèle sauvegardé : model.pkl")
