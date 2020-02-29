"""catplot, stripplot, swarmplot"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


tips = sns.load_dataset("tips")

"""
The approach used by stripplot(), which is the default “kind” in catplot() is to adjust 
the positions of points on the categorical axis with a small amount of random “jitter”:
"""
# sns.catplot(x="day", y="total_bill", data=tips)
# plt.savefig("catplot_scatterplots_images\image1.png")
"""
The jitter parameter controls the magnitude of jitter or disables it altogether:
"""
# sns.catplot(x="day", y="total_bill", jitter=False, data=tips)
# plt.savefig("catplot_scatterplots_images\image2.png")
"""
The second approach adjusts the points along the categorical axis using an algorithm 
that prevents them from overlapping. It can give a better representation of the 
distribution of observations, although it only works well for relatively small datasets. 
This kind of plot is sometimes called a “beeswarm” and is drawn in seaborn by swarmplot(), 
which is activated by setting kind="swarm" in catplot():
"""
# sns.catplot(x="day", y="total_bill", kind="swarm", data=tips)
# plt.savefig("catplot_scatterplots_images\image3.png")
"""
Similar to the relational plots, it’s possible to add another dimension to a categorical 
plot by using a hue semantic. (The categorical plots do not currently support size or style semantics). 
Each different categorical plotting function handles the hue semantic differently. 
For the scatter plots, it is only necessary to change the color of the points:
"""
# sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips)
# plt.savefig("catplot_scatterplots_images\image4.png")
"""
Unlike with numerical data, it is not always obvious how to order the levels of the categorical
variable along its axis. In general, the seaborn categorical plotting functions try to infer the
order of categories from the data. If your data have a pandas Categorical datatype, then the default
order of the categories can be set there. If the variable passed to the categorical axis looks numerical, 
the levels will be sorted. But the data are still treated as categorical and drawn at ordinal positions 
on the categorical axes (specifically, at 0, 1, …) even when numbers are used to label them:
"""
# sns.catplot(x="size", y="total_bill", kind="swarm", data=tips.query("size != 3"))
# plt.savefig("catplot_scatterplots_images\image5.png")
"""
The other option for chosing a default ordering is to take the levels of the category as they 
appear in the dataset. The ordering can also be controlled on a plot-specific basis using the 
order parameter. This can be important when drawing multiple categorical plots in the same figure,
which we’ll see more of below:
"""
# sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips)
# plt.savefig("catplot_scatterplots_images\image6.png")
"""
We’ve referred to the idea of “categorical axis”. In these examples, that’s always corresponded 
to the horizontal axis. But it’s often helpful to put the categorical variable on the vertical 
axis (particularly when the category names are relatively long or there are many categories). 
To do this, swap the assignment of variables to axes:
"""
# sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips)
# plt.savefig("catplot_scatterplots_images\image7.png")
"""
Draw outlines around the points:
"""
# sns.stripplot(x="total_bill", y="day", data=tips, jitter=True, linewidth=1)
# plt.savefig("catplot_scatterplots_images\image8.png")
"""
Draw each level of the hue variable at different locations on the major categorical axis:
"""
# sns.stripplot(x="day",
#               y="total_bill",
#               hue="smoker",
#               data=tips,
#               jitter=True,
#               palette="Set2",
#               dodge=True)
# plt.savefig("catplot_scatterplots_images\image9.png")
"""
Draw strips with large points and different aesthetics:
"""
# sns.stripplot("day",
#               "total_bill",
#               "smoker",
#               data=tips,
#               palette="Set2",
#               size=20,
#               marker="D",
#               edgecolor="gray",
#               alpha=.25)
# plt.savefig("catplot_scatterplots_images\image10.png")
"""
Use catplot() to combine a stripplot() and a FacetGrid. This allows grouping within additional 
categorical variables. Using catplot() is safer than using FacetGrid directly, as it 
ensures synchronization of variable order across facets:
"""
# sns.catplot(x="sex",
#             y="total_bill",
#             hue="smoker",
#             col="time",
#             data=tips,
#             kind="strip",
#             jitter=True,
#             height=4,
#             aspect=.7)
# plt.savefig("catplot_scatterplots_images\image11.png")

# plt.show()
