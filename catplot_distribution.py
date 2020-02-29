"""catplot, boxplot, violinplot, boxenplot"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


tips = sns.load_dataset("tips")
planets = sns.load_dataset("planets")
iris = sns.load_dataset("iris")
diamonds = sns.load_dataset("diamonds")

"""
The first is the familiar boxplot(). This kind of plot shows the three quartile values of the
distribution along with extreme values. The “whiskers” extend to points that lie within 1.5 IQRs
of the lower and upper quartile, and then observations that fall outside this range are displayed
independently. This means that each value in the boxplot corresponds to an actual observation in the data.
"""
# sns.catplot(x="day", y="total_bill", kind="box", data=tips)
# plt.savefig("catplot_distribution_images\image1.png")
"""
When adding a hue semantic, the box for each level of the semantic variable
is moved along the categorical axis so they don’t overlap:
"""
# sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips, palette="Set3")
# plt.savefig("catplot_distribution_images\image2.png")

"""
This behavior is called “dodging” and is turned on by default because it is assumed that 
the semantic variable is nested within the main categorical variable. 
If that’s not the case, you can disable the dodging:
"""
# tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
# sns.catplot(x="day", y="total_bill", hue="weekend", kind="box", dodge=False, data=tips)
# plt.savefig("catplot_distribution_images\image3.png")
"""
Draw a boxplot with nested grouping when some bins are empty:
"""
# sns.boxplot(x="day", y="total_bill", hue="time", data=tips, linewidth=2.5)
# plt.savefig("catplot_distribution_images\image4.png")
"""
Control box order by passing an explicit order:
"""
# sns.boxplot(x="time", y="tip", data=tips, order=["Dinner", "Lunch"])
# plt.savefig("catplot_distribution_images\image5.png")
"""
Draw a boxplot for each numeric variable in a DataFrame:
"""
# sns.boxplot(data=iris, orient="h", palette="Set2")
# plt.savefig("catplot_distribution_images\image6.png")
"""
Use catplot() to combine a pointplot() and a FacetGrid. This allows grouping within
additional categorical variables. Using catplot() is safer than using FacetGrid 
directly, as it ensures synchronization of variable order across facets:
"""
# sns.catplot(x="sex",
#             y="total_bill",
#             hue="smoker",
#             col="time",
#             data=tips,
#             kind="box",
#             height=4,
#             aspect=.7)
# plt.savefig("catplot_distribution_images\image7.png")
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
A related function, boxenplot(), draws a plot that is similar to a box plot but optimized 
for showing more information about the shape of the distribution. It is best suited for larger datasets:
"""
# sns.catplot(x="color", y="price", kind="boxen", data=diamonds.sort_values("color"))
# plt.savefig("catplot_distribution_images\image8.png")
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
A different approach is a violinplot(), which combines a boxplot with the kernel 
density estimation procedure described in the distributions tutorial:
"""
# sns.catplot(x="total_bill", y="day", hue="time", kind="violin", data=tips)
# plt.savefig("catplot_distribution_images\image9.png")
"""
This approach uses the kernel density estimate to provide a richer description of the 
distribution of values. Additionally, the quartile and whikser values from the boxplot 
are shown inside the violin. The downside is that, because the violinplot uses a KDE, 
there are some other parameters that may need tweaking, adding some complexity 
relative to the straightforward boxplot:
"""
# sns.catplot(x="total_bill", y="day", hue="time", kind="violin", bw=.15, cut=0, data=tips)
# plt.savefig("catplot_distribution_images\image10.png")
"""
It’s also possible to “split” the violins when the hue parameter has only 
two levels, which can allow for a more efficient use of space:
"""
# sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", split=True, data=tips)
# plt.savefig("catplot_distribution_images\image11.png")
"""
Finally, there are several options for the plot that is drawn on the 
interior of the violins, including ways to show each individual 
observation instead of the summary boxplot values:
"""
# sns.catplot(x="day",
#             y="total_bill",
#             hue="sex",
#             kind="violin",
#             inner="stick",  # {“box”, “quartile”, “point”, “stick”, None}
#             split=True,
#             palette="pastel",
#             data=tips)
# plt.savefig("catplot_distribution_images\image12.png")
"""
It can also be useful to combine swarmplot() or striplot() with a
box plot or violin plot to show each observation along with 
a summary of the distribution:
"""
# g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips)
# sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax)
# plt.savefig("catplot_distribution_images\image13.png")
"""
Scale the violin width by the number of observations in each bin:
"""
# sns.violinplot(x="day",
#                y="total_bill",
#                hue="sex",
#                data=tips,
#                palette="Set2",
#                split=True,
#                scale="count",)  # {“area”, “count”, “width”},
# plt.savefig("catplot_distribution_images\image14.png")
"""
Scale the density relative to the counts across all bins:
"""
# sns.violinplot(x="day",
#                y="total_bill",
#                hue="sex",
#                data=tips,
#                palette="Set2",
#                split=True,
#                scale="count",
#                inner="stick",
#                scale_hue=False)
# plt.savefig("catplot_distribution_images\image15.png")
"""
Use a narrow bandwidth to reduce the amount of smoothing:
"""
# sns.violinplot(x="day",
#                y="total_bill",
#                hue="sex",
#                data=tips,
#                palette="Set2",
#                split=True,
#                scale="count",
#                inner="stick",
#                scale_hue=False,
#                bw=.2)
# plt.savefig("catplot_distribution_images\image16.png")
"""
Draw horizontal violins:
"""
# sns.violinplot(x="orbital_period",
#                y="method",
#                data=planets[planets.orbital_period < 1000],
#                scale="width",
#                palette="Set3")
# plt.savefig("catplot_distribution_images\image17.png")
"""
Use catplot() to combine a violinplot() and a FacetGrid. This allows grouping within 
additional categorical variables. Using catplot() is safer than using FacetGrid 
directly, as it ensures synchronization of variable order across facets:
"""
# sns.catplot(x="sex",
#             y="total_bill",
#             hue="smoker",
#             col="time",
#             data=tips,
#             kind="violin",
#             split=True,
#             height=4,
#             aspect=.7)
# plt.savefig("catplot_distribution_images\image18.png")

# plt.show()

