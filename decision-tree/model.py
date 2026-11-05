from ucimlrepo import fetch_ucirepo 
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
  
# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
# data (as pandas dataframes) 
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
  
# metadata 
#print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 
  
# variable information 
print(diabetes_130_us_hospitals_for_years_1999_2008.variables) 



# reparse the string data as categorical data
enc = OneHotEncoder(drop='if_binary')
X_transformed = enc.fit(X).transform(X)


clf = tree.DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
clf = clf.fit(X_transformed, y)

plot_tree(clf, filled=True)

import pdb; pdb.set_trace()

plt.savefig('decision_tree.png')

plt.show()
