import matplotlib.pyplot as plt
import pylab as pl
import streamlit as st
from PIL import Image
import base64
from rough_copy import create_class_date
import random
import numpy as np

st.markdown(
    '# K Means Visualization App'
)

st.markdown(
    'K Means is an unsupervised machine learning Algorithm. When your data is not classified into different categories '
    'then Kmeans is used to categorize your data into clusters to gain some kind of classification of the data. '
    'Without any prior knowledge the algorithm Classifies the data based on features that group together'
)

st.markdown(
    '## But How?'
)

st.markdown(
    'Good question. Your question hurt me. Jokes apart, while this looks like magic the real maths behind the curtain '
    'is much simpler. The algorithm decides random points in respect to the features called centroids, then it iterates through all '
    'the points to see if the .....'
)

st.markdown(
    '## You Lost me at maths'
)

st.markdown(
    'Okay! The data science lingo might make it look like a little more difficult than it actually is, the real '
    'algorithm works in a fairly simple way. And if reading all the big terms like centroids, iteration, maths, keyboard '
    'haunts you then do not worry you are not the only one. \n'
    'The best way to learn is more often than not is a visual one, so lets take a deep dive on how does computer does magic, '
)

file_ = open("kmean-vis.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="kmean_gif">',
    unsafe_allow_html=True,
)

st.markdown(
    '## Lets Start from beginning'
)

st.markdown(
    'Lets say We have data of a class where we store weight and height of the students. Now as we know kids come in all'
    'shape and sizes. Lets have a look at this data. ')

class_data = create_class_date()

st.write(class_data)

st.markdown('')
st.markdown(
    'But we do not have any priori categorization of these kids. If we manually go through the data then'
    'we can classify them on the basis of what we want. We can classify them as athletic or not athleitc, or good for '
    'basketball or not good for basketball, or any other category we want to assign them. But for 100 records or million records'
    ' we might not have resources or time to categorize them all. So we can let machine learning algorithm see what kind '
    'of groups can we form and then get properties of these group to see what kind of groups we have.'
)

st.markdown(
    '## Sounds Interesting but HOW?'
)

st.markdown(
    'To see how the computer does it lets first plot the data of our students on a graph'
)

fig, ax = plt.subplots()

ax.scatter(class_data['Weight'],class_data['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')

st.pyplot(fig)

st.markdown(
    "## Step One"
)
st.markdown(
    'So just by looking at it seems like the class has 2 groups of students. One who have weight between 80-120 and '
    'height under 160. The second group has weight between 120-160 and height above 150. But if we have a very scattered '
    'with many such features we cant view them and visualize them. But our good silicon friends can. \n'
    'For that first we tell the algorithm how many categories we want. But how do we know how many categories we want '
    'when we do not know how many categories are there? Fair question, we will cover that too. But for now lets say we '
    'want 2 categories. \n'
    'The KMean Algorithm will select 2 data points at random'
)

record1 = random.randrange(0,100)
record2 = random.randrange(0,100)

fig, ax = plt.subplots()

ax.scatter(class_data['Weight'],class_data['Height'])
ax.scatter(class_data.iloc[record1,0],class_data.iloc[record1,1],color = 'red',s=150)
ax.scatter(class_data.iloc[record2,0],class_data.iloc[record2,1], color='green',s=150)
plt.xlabel('Weight')
plt.ylabel('Height')

st.pyplot(fig)

st.markdown(
    'These are our first centers. Now we calculate the distance of all the points from the centers. If a point is closer'
    ' green center we color it green, if it is closer to red center we color it red. \n'
    'Note- The distance is calculated using Euclidean Distance'
)

center_red = np.array([class_data.iloc[record1,0],class_data.iloc[record1,1]])
center_green = np.array([class_data.iloc[record2,0],class_data.iloc[record2,1]])

fig, ax = plt.subplots()

ax.scatter(class_data['Weight'],class_data['Height'])
ax.scatter(class_data.iloc[record1,0],class_data.iloc[record1,1],color = 'red',s=150)
ax.scatter(class_data.iloc[record2,0],class_data.iloc[record2,1], color='green',s=150)
plt.xlabel('Weight')
plt.ylabel('Height')

#st.pyplot(fig)

red_pt = [[],[]]
green_pt = [[],[]]

pl = st.empty()
for i in range(100):
    curr_point = np.array([class_data.iloc[i,0],class_data.iloc[i,1]])

    dist_red = np.linalg.norm(center_red - curr_point)
    dist_green = np.linalg.norm(center_green - curr_point)


    if dist_red<dist_green:

        ax.scatter(class_data.iloc[i,0],class_data.iloc[i,1], color='red')

        plt.xlabel('Weight')
        plt.ylabel('Height')
        red_pt[0].append(curr_point.tolist()[0])
        red_pt[1].append(curr_point.tolist()[1])


    else:
        ax.scatter(class_data.iloc[i, 0], class_data.iloc[i, 1], color='green')

        plt.xlabel('Weight')
        plt.ylabel('Height')
        green_pt.append(curr_point)
        green_pt[0].append(curr_point.tolist()[0])
        green_pt[1].append(curr_point.tolist()[1])

    pl.pyplot(fig)

st.markdown(
    '## Step Two'
)

st.markdown(
    'So these are our 2 group. We got the classification, alright! Bye guys, good work. Except this cluster does not '
    'look so perfect. The first random points can be anything, so our clustering is as good as a coin toss. And we Data '
    'Scientist are not Witchers so thats not good for us.'
)



fig, ax = plt.subplots()

#ax.scatter(class_data['Weight'],class_data['Height'])
ax.scatter(red_pt[0],red_pt[1],color = 'red')
ax.scatter(green_pt[0],green_pt[1],color = 'green')
plt.xlabel('Weight')
plt.ylabel('Height')

st.pyplot(fig)
st.markdown(
    'Now we have our initial groups, red and green. Now we can find out the centers of these new groups and pin them '
    'down'
)


center_red = (sum(red_pt[0]) / len(red_pt[0]), sum(red_pt[1]) / len(red_pt[1]))
#print(center_red)
center_green = (sum(green_pt[0]) / len(green_pt[0]), sum(green_pt[1]) / len(green_pt[1]))
#print(center_green)

fig, ax = plt.subplots()

ax.scatter(class_data['Weight'],class_data['Height'])
ax.scatter(center_red[0],center_red[1],color = 'red',s=150)
ax.scatter(center_green[0],center_green[1], color='green',s=150)
plt.xlabel('Weight')
plt.ylabel('Height')

st.pyplot(fig)

st.markdown(
    'So now these new center might be looking better than the ones you had before. These new center might look they are '
    'better around the clusters that we can visualize. It might not look really big improvement or might look like a huge improvement. '
    'Because the initial centers are taken at random, your results may differ every time you run the program. That is '
    'how KMeans approach a problem. And since its not a creep it does not scare it away. Lets not get personal here'
)


st.markdown(
    'Since these new points are better than points selected at random, we can now use them to group our students '
    'together. So lets run that simulation of classifying the points again'
)


center_red = (sum(red_pt[0]) / len(red_pt[0]), sum(red_pt[1]) / len(red_pt[1]))
#print(center_red)
center_green = (sum(green_pt[0]) / len(green_pt[0]), sum(green_pt[1]) / len(green_pt[1]))
#print(center_green)


st.markdown(
    'Since we have better center points now lets group our points based on these new centers'
)

red_pt = []
green_pt = []

pl = st.empty()
for i in range(100):
    curr_point = np.array([class_data.iloc[i,0],class_data.iloc[i,1]])

    dist_red = np.linalg.norm(center_red - curr_point)
    dist_green = np.linalg.norm(center_green - curr_point)


    if dist_red<dist_green:

        ax.scatter(class_data.iloc[i,0],class_data.iloc[i,1], color='red')

        plt.xlabel('Weight')
        plt.ylabel('Height')
        red_pt.append(curr_point.tolist())


    else:
        ax.scatter(class_data.iloc[i, 0], class_data.iloc[i, 1], color='green')

        plt.xlabel('Weight')
        plt.ylabel('Height')
        green_pt.append(curr_point)

    pl.pyplot(fig)

st.markdown(
    'Now our new data points are looking even better put together as a cluster. What we visually figured out can be '
    'seen by iteration of this process as well'
)

fig, ax = plt.subplots()

#ax.scatter(class_data['Weight'],class_data['Height'])
ax.scatter(red_pt[0],red_pt[1],color = 'red')
ax.scatter(green_pt[0],green_pt[1],color = 'green')
plt.xlabel('Weight')
plt.ylabel('Height')

st.pyplot(fig)

st.markdown(
    'Since these points are even better looking cluster the algorithm can go one more time. It will repeat the same steps'
    '1. Make the centers'
    '2. Classify points according to the new centers based on Eclidean Distance'
    '3. Identify new center'
    '4. Repeat'
)

st.markdown(
    '## When will the process of pain end?'
)

st.markdown(
    'While I cannot say about your life, I can tell you when the KMean will stop doing these steps over and over again'
    ' so it does not go insane. If the center points do not change on the next iteration then we know there will be no '
    'change in the clustering in the next round. So KMeans will stop. In this case we have found our groups. \n'
    'In case KMeans did not reach such a condition, then it has predefined number of iterations, that is the number of '
    'time it will perform this cycle. Usually the number is predefined to be 300. So after 300 cycles if the Center '
    'points are still changing then KMeans will stop and give you the last result. In this case the clustering might '
    'have not be optimal. So you can change the number of iterations manually. In theory it might look like a great '
    'idea to let the algorithm run for 10,000 steps so it comes to a stable center point, it is not the best approach. '
    'Later we will cover why this is the case.'
)

st.markdown(
    '## Conclusion'
)

st.markdown(
    'You might still have many questions, like how do we do we select the number of clusters, why we should not run '
    'the algorithm for many iterations, how to know if we have correct groups when we cannot visually see them, was I a '
    'good father? And we will cover most of them going ahead. But for now you got a good understanding of how KMeans '
    'work. \n'
    'And if you think about it, you not only know how KMean works, but you also know how computers use maths to think '
    'and see things the way humans do. You now know how a computer thinks, which is part of Artificial Intelligence. '
    'This might not look like terminator but trust me even most advance AI also work on similar mathematics of trial and'
    ' error. If you can understand this than nothing is stopping you from taking a deeper dive in the world of Data '
    'Science. And in case you did not understand this then refresh button is next to you ;) '
)

st.markdown(
    '## Going Ahead \n'
    'I will be updating this module soon to cover many unanswered question. But if you are impatient you can take a dive'
    ' into the world of machine learning yourself. \n'
    ''
)