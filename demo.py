import numpy
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image as PImage
import pydotplus
import pandas as pd
from sklearn import tree
import subprocess


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

full_data = [train, test]

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

features = ['Has_Cabin']
x = train[features]
y = train[['Survived']]

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=1)
Survival_Model = tree.DecisionTreeClassifier()
Survival_Model.fit(train_x,train_y)
Val_prediction = Survival_Model.predict(val_x)
print (mean_absolute_error(val_y,Val_prediction))

with open("tree1.dot", 'w') as f:
    f = tree.export_graphviz(Survival_Model,
                             out_file=f,
                             class_names=['Died','Survived'])

# Convert .dot to .png to allow display in web notebook
check_call(['dot', '-Tpng', 'tree1.dot', '-o', 'tree1.png'])

from subprocess import call
call(['dot', '-Tpng', 'tree1.dot', '-o', 'tree.png', '-Gdpi=600'])


# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0),  # Drawing offset (position)
          '"Title <= 1.5" corresponds to "Mr." title',  # Text to draw
          (0, 0, 255),  # RGB desired color
          font=font)  # ImageFont object with desired font
img.save('sample-out.png')
PImage("sample-out.png")


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


visualize_tree(Survival_Model, features)


from sklearn import datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()

