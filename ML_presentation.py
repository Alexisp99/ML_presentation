from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_nested_layout
from sklearn.neighbors import KNeighborsClassifier

from PIL import Image

st.set_page_config(page_title = "Iris",
                   layout = "wide", 
                   )


iris = load_iris(as_frame=True)
df_iris = pd.DataFrame(iris.frame)


y = df_iris["target"]
y = y.values.reshape((y.shape[0],1))
X = df_iris.drop("target",axis=1)


def df_iris_names():
    for i in range(3):
        df_iris["target"].loc[df_iris["target"]==i] = iris.target_names[i]
    st.dataframe(df_iris)
    

def petal_input(model):
    x = pd.DataFrame(data=[[s_length],[s_width],[p_length],[p_width]]).T

    fig,ax= plt.subplots()
    ax = sns.scatterplot(data=df_iris, x="petal length (cm)", y="petal width (cm)", hue="target", palette="tab10")
    
    plt.scatter(x=p_length, y=p_width, color="r")
    st.pyplot(fig)
    
    st.write(model.predict_proba(x))
    prediction = model.predict(x)[0]
    if prediction == 0:
        st.write("Your plant is a Setosa")
    elif prediction == 1:
        st.write("Your plant is a Versicolor")
    elif prediction == 2:
        st.write("Your plant is a Virginica")
           
           
def sepal_input(model):
    x = pd.DataFrame(data=[[s_length],[s_width],[p_length],[p_width]]).T
    


    fig,ax= plt.subplots()
    ax = sns.scatterplot(data=df_iris, x="sepal length (cm)", y="sepal width (cm)", hue="target", palette="tab10")

    plt.scatter(x=s_length, y=s_width, color="r")
    st.pyplot(fig)

    

head = st.container()
dataset = st.container()
display_algorithm = st.container()

linearR = Image.open("image/linear_regression.png")
knn = Image.open("image/KNN.png")
flower = Image.open("image/flower.png")
algorithm = Image.open("image/algorithm.png")
algorithm = algorithm.resize((700,500))


with head:
    st.title("Machine learning")
    st.markdown("This is the presentation of my different subjects of machine learning  \n"
                "We're gonna use different type of algorithm :  \n"
                )
    col1,col2,col3,col4 = st.columns([0.1,1,1,0.1])
    with col2:
        st.header("Linear Regression")
        st.image(linearR)
    with col3:
        st.header("KNN")
        st.image(knn)
    
    
with dataset:
    col1,col2,col3 = st.columns([0.1,1,0.7])
    col2.header("Iris project")
    col2.markdown("Here we use the dataset Iris of scikit-learn who allowing to learn more about the KNN algorithm  \n"
                "In the dataset we have multiple parameters: the name of the flowers(Setosa, versicolor,viriginica), the lenght and width of the petal and sepal  \n"
                "The target here is to predict which category of plants do these flowers belong depending of the lenght and width of petal and sepal"
                )
    col1,col2,col3,col4 = st.columns([0.1,1,1,0.1])
    with col2:
        st.image(flower)
    
    
    with col3:
        df_iris_names()
    
with display_algorithm:  
    col1,col2,col3 = st.columns([0.1,1,0.7])
    col2.header("Display and choose alogithm")
    col2.markdown("Now, we started by display the different features and choose which graphic is relevant or not")
    col1,col2,col3,col4= st.columns([0.1,1.2,1,0.1])
    
    with col2:
        fig = sns.pairplot(df_iris, hue = "target", palette= "tab10")
        st.pyplot(fig)
        
    with col3:
        st.markdown("We can clearly see the link between each features(lenght,width of sepal and petal), and see three distinc group :  \n"
                    "- Blue for Setosa,  "
                    "Orange for Versicolor,  "
                    "Green for Virginica  \n"
                    "Like we sayd before the main goal is to predict if my flowers is an Setosa, Versicolor or Virginica.  \n"
                    "To do that, we need to identify which category of algotihm fit our problem. It is a problem of Classification, Regression, Clustering, or dimensionality reduction ?  \n"
                    "If you don't know which category is appropriate in this situation, we can refer to the graphic of sickit-learn."
                    )
        st.image(algorithm)
        st.markdown("So, we start, we got more than 50 samples, we need to predicting a category, our data is labeled. We can conclude it's a Classification problem  \n"
                    "Now we need to determin which alogithm fit our problem. We don't have more than 100K samples, got it, Linear SVC. But for simplified we will use the KNN alogithm who is more simpler and easier to understand. ")
        
        
experimentation = st.container()

with experimentation:
    col1,col2,col3 = st.columns([0.1,1.5,0.2])
    
    with col2:
        st.header("Train our AI")
        st.markdown("To train our model, we're gonna use Scikit learn who is a powerfull library.  \n"
                    "We import the module KNeighborsClassifier.  \n"
                    "To train our model, we need, a variable X who is the mutliple values of lenght and width, and a variable y who is our target.")
        st.code("""y = df_iris["target"]
(150,)
y = y.values.reshape((y.shape[0],1))
(150,1)
X = df_iris.drop("target",axis=1)""")
        st.markdown("Now we have our values for X and y so we can train our model.  \n"
                    "The KNN algorithm can work with certain parameters like the number of neighbors, the weight(if each point have the same value or if may vary with the distance), the metrics, etc..  \n"
                    "For the moment we just use the number of neighbors and train our model.")
        st.code("""model = KNeighborsClassifier(n_neighbors= 3)
model.fit(X,y)
model.score(X)
0.96""")
        st.markdown("model.score(X) reveal if our model is reliable or no. However, the score of 0.96 here is not reliable, we just set the paramters in the simple way, we will see later why.  \n"
                    )
    
        st.subheader("Try the model")
        st.markdown("Now insert a new value of a plant to see what our model can predict")
        
        
        col1,col2,col3,col4 = st.columns([1,1,1,1])
        model = KNeighborsClassifier(n_neighbors= 3)
        model.fit(X,y)
        
        s_length=col1.slider("sepal length",0,9,3) 
        s_width=col2.slider("sepal_width",0,5,2)
        p_length=col3.slider("Petal length",0,8,4) 
        p_width=col4.slider("Petal width",0,3,1)
        
        
        col1,col2,col3 = st.columns([0.1,1,1])
        with col2:
            petal_input(model)
        with col3:
            sepal_input(model)
            
        
        
        
    
    

