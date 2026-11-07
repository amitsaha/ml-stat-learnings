from ucimlrepo import fetch_ucirepo 
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# https://fairlearn.org/v0.14/user_guide/datasets/diabetes_hospital_data.html#footcite-strack2014diabetes
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

print(X.shape)


#(Pdb) y.value_counts()
#readmitted
#NO            54864
#>30           35545
#<30           11357

# 1 if readmitted - binary classification
y = y['readmitted'] != 'NO'

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#(Pdb) X.columns
#Index(['race', 'gender', 'age', 'weight', 'admission_type_id',
#       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
#       'payer_code', 'medical_specialty', 'num_lab_procedures',
#       'num_procedures', 'num_medications', 'number_outpatient',
#       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
#       'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
#       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
#       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
#       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
#       'tolazamide', 'examide', 'citoglipton', 'insulin',
#       'glyburide-metformin', 'glipizide-metformin',
#       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
#       'metformin-pioglitazone', 'change', 'diabetesMed'],
#      dtype='str')
#import pdb; pdb.set_trace()
#

  
# metadata 
#print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 
  
# variable information 
#print(diabetes_130_us_hospitals_for_years_1999_2008.variables) 



# reparse the string data as categorical data
enc = OneHotEncoder(drop='if_binary')
encoder = enc.fit(X)

X_train_transformed = encoder.transform(X_train)
X_test_transformed = encoder.transform(X_test)

# attempt some feature selection
#X_new = SelectKBest(f_classif, k=5).fit_transform(X_transformed, y)




#clf = tree.DecisionTreeClassifier(max_depth=3, random_state=0)
clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train_transformed, y_train)

# visualization

plot_tree(clf, filled=True, proportion=True)
plt.savefig('decision_tree.png', dpi=300)
#plt.show()

# Tree stats

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold
values = clf.tree_.value

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
)

print(f"Depth of tree: {max(node_depth)}")

#for i in range(n_nodes):
#    if is_leaves[i]:
#        print(
#            "{space}node={node} is a leaf node with value={value}.".format(
#                space=node_depth[i] * "\t", node=i, value=np.around(values[i], 3)
#            )
#        )
#    else:
#        print(
#            "{space}node={node} is a split node with value={value}: "
#            "go to node {left} if X[:, {feature}] <= {threshold} "
#            "else to node {right}.".format(
#                space=node_depth[i] * "\t",
#                node=i,
#                left=children_left[i],
#                feature=feature[i],
#                threshold=threshold[i],
#                right=children_right[i],
#                value=np.around(values[i], 3),
#            )
#        )

# Decision path

node_indicator = clf.decision_path(X_test_transformed)
leaf_id = clf.apply(X_test_transformed)

sample_id = 0
# obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
node_index = node_indicator.indices[
    node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
]

print("Rules used to predict sample {id}:\n".format(id=sample_id))
for node_id in node_index:
    # continue to the next node if it is a leaf node
    if leaf_id[sample_id] == node_id:
        continue

    # check if value of the split feature for sample 0 is below threshold
    if X_test_transformed[sample_id, feature[node_id]] <= threshold[node_id]:
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print(
        "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
        "{inequality} {threshold})".format(
            node=node_id,
            sample=sample_id,
            feature=feature[node_id],
            value=X_test_transformed[sample_id, feature[node_id]],
            inequality=threshold_sign,
            threshold=threshold[node_id],
        )
    )
