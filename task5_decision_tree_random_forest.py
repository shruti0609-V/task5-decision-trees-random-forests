import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Load dataset
df = pd.read_csv("data/heart.csv")

# Preprocessing
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

# Visualize tree
dot_data = export_graphviz(dtree, out_file=None, feature_names=X.columns,
                           class_names=["No Disease", "Disease"],
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("images/decision_tree")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, dtree.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# Feature importance
importances = rf.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("images/feature_importances.png")
