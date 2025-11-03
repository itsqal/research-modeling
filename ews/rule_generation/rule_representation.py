import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from sklearn import tree

def extract_rules_from_tree(clf, feature_names):
    tree_ = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Kiri dan kanan split
            recurse(tree_.children_left[node], current_rule + [f"{name} ≤ {threshold:.3f}"])
            recurse(tree_.children_right[node], current_rule + [f"{name} > {threshold:.3f}"])
        else:
            # Leaf node → class prediksi
            predicted_class = clf.classes_[np.argmax(tree_.value[node])]
            rules.append({"rule": current_rule, "predicted_class": predicted_class})

    recurse(0, [])
    return rules


def visualize_rule_network_topN(clf, feature_names=None, top_n=5, figsize=(10, 6)):
    if feature_names is None:
        feature_names = getattr(clf, "feature_names_in_", None)
        if feature_names is None:
            raise ValueError("Feature names must be provided or available in clf.feature_names_in_")

    # pastikan panjangnya sama
    if len(feature_names) != len(clf.feature_importances_):
        print("[Warning] Length mismatch: adjusting feature names to match model features")
        feature_names = feature_names[: len(clf.feature_importances_)]

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    top_features = importances.head(top_n)["feature"].tolist()
    rules = extract_rules_from_tree(clf, feature_names)

    # Build network
    G = nx.DiGraph()
    for i, r in enumerate(rules):
        rule_name = f"Rule_{i+1}"
        for cond in r["rule"]:
            if any(f in cond for f in top_features):
                G.add_edge(cond, rule_name)
        G.add_edge(rule_name, r["predicted_class"])

    pos = nx.spring_layout(G, k=0.8, seed=42)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color="#8BD3E6")
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    plt.title(f"Rule Network (Top {top_n} Features Only)")
    plt.axis("off")
    plt.show()

    return G

def visualize_decision_tree_graph(clf, feature_names, class_names=None, figsize=(20, 10), dpi=200):
    """
    Visualisasi pohon keputusan dalam bentuk diagram hierarkis (tree graph).

    Parameters
    ----------
    clf : DecisionTreeClassifier
        Model decision tree yang sudah dilatih.
    feature_names : list
        Nama-nama fitur yang digunakan untuk training.
    class_names : list, optional
        Nama kelas target (jika tidak, diambil dari clf.classes_).
    figsize : tuple
        Ukuran figure.
    dpi : int
        Resolusi figure untuk tampilan lebih tajam.
    """

    if class_names is None:
        class_names = [str(c) for c in clf.classes_]

    plt.figure(figsize=figsize, dpi=dpi)
    tree.plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=10
    )
    plt.title("Decision Tree Visualization for Disaster Risk Prediction", fontsize=14, fontweight="bold")
    plt.show()
