from sklearn import tree

# Data source to learn from. First number in the set is the age, the second is a binary representation of if they have wrinkles or not (1 = yes, 0 = no)
features = [[69, 1], [78, 1], [20, 0], [21, 1]]
# Let's map those to real the outcomes (eg. The first person is 69 and has wrinkles. Therefore, they are Old)
labels = ["Old", "Old", "Young", "Young"]
# Instantiate the classifier
clf = tree.DecisionTreeClassifier()
# Call the learning function...
clf = clf.fit(features, labels)

age = input("Enter your age: ")
wrinkles = input("Do you have wrinkles? (y/n): ")
if wrinkles.lower() == "y":
    wrinkles_state = 1
else:
    wrinkles_state = 0

result = clf.predict([[age, wrinkles_state]])
print(f"You are {str(result[0])}")
