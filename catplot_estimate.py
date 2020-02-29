"""catplot, pointplot, barplot, countplot"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()


titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")

"""
A familiar style of plot that accomplishes this goal is a bar plot. In seaborn, the barplot()
function operates on a full dataset and applies a function to obtain the estimate
(taking the mean by default). When there are multiple observations in each category,
it also uses bootstrapping to compute a confidence interval around the estimate and plots that using error bars:
"""
# sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)
# plt.savefig("catplot_estimate_images\image1.png")
"""
Control bar order by passing an explicit order:
"""
# sns.barplot(x="time", y="tip", data=tips, order=["Dinner", "Lunch"])
# plt.savefig("catplot_estimate_images\image2.png")
"""
Use median as the estimate of central tendency:
"""
# sns.barplot(x="day", y="tip", data=tips, estimator=np.median)
# plt.savefig("catplot_estimate_images\image3.png")
"""
Show the standard error of the mean with the error bars:
"""
# sns.barplot(x="day", y="tip", data=tips, ci=68)
# plt.savefig("catplot_estimate_images\image4.png")
"""
Show standard deviation of observations instead of a confidence interval:
"""
# sns.barplot(x="day", y="tip", data=tips, ci="sd")
# plt.savefig("catplot_estimate_images\image5.png")
"""
Add “caps” to the error bars:
"""
# sns.barplot(x="day", y="tip", data=tips, capsize=.2)
# plt.savefig("catplot_estimate_images\image6.png")
"""
Use a different color palette for the bars:
"""
# sns.barplot("size", y="total_bill", data=tips, palette="Blues_d")
# plt.savefig("catplot_estimate_images\image7.png")
"""
Use hue without changing bar position or width:
"""
# tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
# sns.barplot(x="day", y="total_bill", hue="weekend", data=tips, dodge=False)
# plt.savefig("catplot_estimate_images\image8.png")
"""
Plot all bars in a single color:
"""
# sns.barplot("size", y="total_bill", data=tips, color="salmon", saturation=.5)
# plt.savefig("catplot_estimate_images\image9.png")
"""
Use plt.bar keyword arguments to further change the aesthetic:
"""
# sns.barplot("day",
#             "total_bill",
#             data=tips,
#             linewidth=2.5,
#             facecolor=(1, 1, 1, 0),
#             errcolor=".2",
#             edgecolor=".2")
# plt.savefig("catplot_estimate_images\image10.png")
"""
Use catplot() to combine a barplot() and a FacetGrid. This allows grouping within
additional categorical variables. Using catplot() is safer than using FacetGrid 
directly, as it ensures synchronization of variable order across facets:
"""
# sns.catplot(x="sex",
#             y="total_bill",
#             hue="smoker",
#             col="time",
#             data=tips,
#             kind="bar",
#             height=4,
#             aspect=.7)
# plt.savefig("catplot_estimate_images\image11.png")
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
Show value counts for a single categorical variable:
"""
# sns.countplot(x="class", data=titanic)
# plt.savefig("catplot_estimate_images\image12.png")
"""
A special case for the bar plot is when you want to show the number of observations in each 
category rather than computing a statistic for a second variable. This is similar to a 
histogram over a categorical, rather than quantitative, variable. In seaborn, 
it’s easy to do so with the countplot() function:
"""
# sns.catplot(x="deck", kind="count", palette="ch:.25", data=titanic)
# plt.savefig("catplot_estimate_images\image13.png")
"""
Both barplot() and countplot() can be invoked with all of the options discussed above, 
along with others that are demonstrated in the detailed documentation for each function:
"""
# sns.catplot(y="deck",
#             hue="class",
#             kind="count",
#             palette="pastel",
#             edgecolor=".6",
#             data=titanic)
# plt.savefig("catplot_estimate_images\image14.png")
"""
Use plt.bar keyword arguments for a different look:
"""
# sns.countplot(x="who",
#               data=titanic,
#               facecolor=(0, 0, 0, 0),
#               linewidth=5,
#               edgecolor=sns.color_palette("dark", 3))
# plt.savefig("catplot_estimate_images\image15.png")
"""
Use catplot() to combine a countplot() and a FacetGrid. This allows grouping within 
additional categorical variables. Using catplot() is safer than using FacetGrid 
directly, as it ensures synchronization of variable order across facets:
"""
# sns.catplot(x="class",
#             hue="who",
#             col="survived",
#             data=titanic,
#             kind="count",
#             height=4,
#             aspect=.7)
# plt.savefig("catplot_estimate_images\image16.png")
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
An alternative style for visualizing the same information is offered by the pointplot() function.
This function also encodes the value of the estimate with height on the other axis, but rather 
than showing a full bar, it plots the point estimate and confidence interval. Additionally,
pointplot() connects points from the same hue category. This makes it easy to see how the 
main relationship is changing as a function of the hue semantic, because your eyes are 
quite good at picking up on differences of slopes:
"""
# sns.catplot(x="sex", y="survived", hue="class", kind="point", data=titanic)
# plt.savefig("catplot_estimate_images\image17.png")
"""
When the categorical functions lack the style semantic of the relational functions, 
it can still be a good idea to vary the marker and/or linestyle along with the hue 
to make figures that are maximally accessible and reproduce well in black and white:
"""
# sns.catplot(x="class",
#             y="survived",
#             hue="sex",
#             palette={"male": "g", "female": "m"},
#             markers=["^", "o"],
#             linestyles=["-", "--"],
#             kind="point",
#             data=titanic)
# plt.savefig("catplot_estimate_images\image18.png")

# plt.show()
