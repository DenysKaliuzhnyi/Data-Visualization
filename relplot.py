"""relplot, lineplot, scatterplot"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
fmri = sns.load_dataset("fmri")
dots = sns.load_dataset("dots").query("align == 'dots'")

pd.plotting.register_matplotlib_converters()
index = pd.date_range("1 1 2000", periods=100, freq="m", name="date")
data = np.random.randn(100, 4).cumsum(axis=0)
wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])

"""
The scatterplot() is the default kind in relplot() (it can also be forced by setting kind="scatter"):
"""
# sns.relplot(x="total_bill", y="tip", data=tips)
# plt.savefig("relplot_images\image1.png")
"""
While the points are plotted in two dimensions, another dimension can be added to the plot by coloring the points 
according to a third variable. In seaborn, this is referred to as using a “hue semantic”, 
because the color of the point gains meaning:
"""
# sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips)
# plt.savefig("relplot_images\image2.png")
"""
To emphasize the difference between the classes, and to improve accessibility, 
you can use a different marker style for each class:
"""
# sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker", data=tips)
# plt.savefig("relplot_images\image3.png")
"""
It’s also possible to represent four variables by changing the hue and style of each point independently.
 But this should be done carefully, because the eye is much less sensitive to shape than to color:
"""
# sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips)
# plt.savefig("relplot_images\image4.png")
"""
If the hue semantic is numeric (specifically, if it can be cast to float), 
the default coloring switches to a sequential palette:
"""
# sns.relplot(x="total_bill", y="tip", hue="size", data=tips)
# plt.savefig("relplot_images\image5.png")
"""
In both cases, you can customize the color palette. There are many options for doing so.
Here, we customize a sequential palette using the string interface to cubehelix_palette():
"""
# sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=tips)
# plt.savefig("relplot_images\image6.png")
"""
The third kind of semantic variable changes the size of each point:
"""
# sns.relplot(x="total_bill", y="tip", size="size", data=tips)
# plt.savefig("relplot_images\image7.png")
"""
Unlike with matplotlib.pyplot.scatter(), the literal value of the variable is not used to pick the area of the point. 
Instead, the range of values in data units is normalized into a range in area units. This range can be customized:
"""
# sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips)
# plt.savefig("relplot_images\image8.png")
"""
Also show the quantitative variable by also using continuous colors:
"""
# sns.scatterplot(x="total_bill", y="tip", hue="size", size="size", data=tips)
# plt.savefig("relplot_images\image9.png")
"""
Use a different continuous color map:
"""
# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
# sns.scatterplot(x="total_bill",
#                 y="tip",
#                 hue="size",
#                 size="size",
#                 palette=cmap,
#                 data=tips)
# plt.savefig("relplot_images\image10.png")
"""
Change the minimum and maximum point size and show all sizes in legend:
"""
# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
# sns.scatterplot(x="total_bill",
#                 y="tip",
#                 hue="size",
#                 size="size",
#                 sizes=(20, 200),
#                 palette=cmap,
#                 legend="full",
#                 data=tips)
# plt.savefig("relplot_images\image11.png")
"""
Use a narrower range of color map intensities:
"""
# sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
# sns.scatterplot(x="total_bill",
#                 y="tip",
#                 hue="size",
#                 size="size",
#                 sizes=(20, 200),
#                 hue_norm=(0, 10),
#                 legend="full",
#                 data=tips)
# plt.savefig("relplot_images\image12.png")
"""
Vary the size with a categorical variable, and use a different palette:
"""
# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
# sns.scatterplot(x="total_bill",
#                 y="tip",
#                 hue="day",
#                 size="smoker",
#                 palette="Set2",
#                 data=tips)
# plt.savefig("relplot_images\image13.png")
"""
Use a specific set of markers:
"""
# markers = {"Lunch": "s", "Dinner": "X"}
# sns.scatterplot(x="total_bill", y="tip", style="time", markers=markers, data=tips)
# plt.savefig("relplot_images\image14.png")
"""
Control plot attributes using matplotlib parameters:
"""
# sns.scatterplot(x="total_bill", y="tip", s=100, color=".2", marker="+", data=tips)
# plt.savefig("relplot_images\image15.png")
"""
Pass data vectors instead of names in a data frame:
"""
# sns.scatterplot(x=iris.sepal_length, y=iris.sepal_width, hue=iris.species, style=iris.species)
# plt.savefig("relplot_images\image16.png")
"""
Pass a wide-form dataset and plot against its index:
"""
# index = pd.date_range("1 1 2000", periods=100, freq="m", name="date")
# data = np.random.randn(100, 4).cumsum(axis=0)
# wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
# ax = sns.scatterplot(data=wide_df)
# plt.savefig("relplot_images\image17.png")
"""
Facet on the columns with another variable:
"""
# sns.relplot(x="total_bill", y="tip", hue="day", col="time", data=tips)
# plt.savefig("relplot_images\image18.png")
"""
Facet on the columns and rows:
"""
# sns.relplot(x="total_bill", y="tip", hue="day", col="time", row="sex", data=tips)
# plt.savefig("relplot_images\image19.png")
"""
“Wrap” many column facets into multiple rows:
"""
# sns.relplot(x="total_bill", y="tip", hue="time", col="day", col_wrap=2, data=tips)
# plt.savefig("relplot_images\image20.png")
"""
Use multiple semantic variables on each facet with specified attributes:
"""
# sns.relplot(x="total_bill",
#             y="tip",
#             hue="time",
#             size="size",
#             palette=["b", "r"],
#             sizes=(10, 100),
#             col="time",
#             data=tips)
# plt.savefig("relplot_images\image21.png")
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
With some datasets, you may want to understand changes in one variable as a function of time, 
or a similarly continuous variable. In this situation, a good choice is to draw a line plot.
In seaborn, this can be accomplished by the lineplot() function, either directly or
with relplot() by setting kind="line":
"""
# df = pd.DataFrame(dict(time=np.arange(500), value=np.random.randn(500).cumsum()))
# g = sns.relplot(x="time", y="value", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.savefig("relplot_images\image22.png")
"""
Because lineplot() assumes that you are most often trying to draw y as a function of x, the default 
behavior is to sort the data by the x values before plotting. However, this can be disabled:
"""
# df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
# sns.relplot(x="x", y="y", sort=False, kind="line", data=df)
# plt.savefig("relplot_images\image23.png")
"""
More complex datasets will have multiple measurements for the same value of the x variable. 
The default behavior in seaborn is to aggregate the multiple measurements at each x value 
by plotting the mean and the 95% confidence interval around the mean:
"""
# fig, ax = plt.subplots()
# sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, ax=ax)
# sns.relplot(x="timepoint", y="signal", data=fmri, ax=ax)
# fig.savefig("relplot_images\image24.png")
"""
The confidence intervals are computed using bootstrapping, which can be 
time-intensive for larger datasets. It’s therefore possible to disable them:
"""
# sns.relplot(x="timepoint", y="signal", ci=None, kind="line", data=fmri)
# plt.savefig("relplot_images\image25.png")
"""
Another good option, especially with larger data, is to represent the spread of the distribution 
at each timepoint by plotting the standard deviation instead of a confidence interval:
"""
# fig, ax = plt.subplots()
# sns.relplot(x="timepoint", y="signal", kind="line", ci="sd", data=fmri, ax=ax)
# sns.relplot(x="timepoint", y="signal", data=fmri, ax=ax)
# fig.savefig("relplot_images\image26.png")
"""
To turn off aggregation altogether, set the estimator parameter to None 
This might produce a strange effect when the data have multiple observations at each point.
"""
# sns.relplot(x="timepoint", y="signal", estimator=None, kind="line", data=fmri)
# plt.savefig("relplot_images\image27.png")
"""
Using semantics in lineplot() will also determine how the data get aggregated. For example, 
adding a hue semantic with two levels splits the plot into two lines and error bands, 
coloring each to indicate which subset of the data they correspond to.
"""
# sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri)
# plt.savefig("relplot_images\image28.png")
"""
Adding a style semantic to a line plot changes the pattern of dashes in the line by default:
"""
# sns.relplot(x="timepoint", y="signal", hue="region", style="event", kind="line", data=fmri)
# plt.savefig("relplot_images\image29.png")
"""
But you can identify subsets by the markers used at each observation, 
either together with the dashes or instead of them:
"""
# sns.relplot(x="timepoint",
#             y="signal",
#             hue="region",
#             style="event",
#             dashes=False,
#             markers=True,
#             kind="line",
#             data=fmri)
# plt.savefig("relplot_images\image30.png")
"""
As with scatter plots, be cautious about making line plots using multiple semantics. 
While sometimes informative, they can also be difficult to parse and interpret. 
But even when you are only examining changes across one additional variable, 
it can be useful to alter both the color and style of the lines. 
This can make the plot more accessible when printed to black-and-white 
or viewed by someone with color blindness:
"""
# sns.relplot(x="timepoint", y="signal", hue="event", style="event", kind="line", data=fmri)
# plt.savefig("relplot_images\image31.png")
"""
When you are working with repeated measures data (that is, you have units that were sampled multiple times), 
you can also plot each sampling unit separately without distinguishing them through semantics. 
This avoids cluttering the legend:
"""
# sns.relplot(x="timepoint",
#             y="signal",
#             hue="region",
#             units="subject",
#             estimator=None,
#             kind="line",
#             data=fmri.query("event == 'stim'"))
# plt.savefig("relplot_images\image32.png")
"""
The default colormap and handling of the legend in lineplot() also depends on
whether the hue semantic is categorical or numeric:
"""
# sns.relplot(x="time",
#             y="firing_rate",
#             hue="coherence",
#             style="choice",
#             kind="line",
#             data=dots)
# plt.savefig("relplot_images\image33.png")
"""
It may happen that, even though the hue variable is numeric, it is poorly represented by 
a linear color scale. That’s the case here, where the levels of the hue variable are 
logarithmically scaled. You can provide specific color values for each line by passing a list or dictionary:
"""
# palette = sns.cubehelix_palette(light=.8, n_colors=6)
# sns.relplot(x="time",
#             y="firing_rate",
#             hue="coherence",
#             style="choice",
#             palette=palette,
#             kind="line",
#             data=dots)
# plt.savefig("relplot_images\image34.png")
"""
Or you can alter how the colormap is normalized:
"""
# from matplotlib.colors import LogNorm
# palette = sns.cubehelix_palette(light=.7, n_colors=6)
# sns.relplot(x="time",
#             y="firing_rate",
#             hue="coherence",
#             style="choice",
#             hue_norm=LogNorm(),
#             kind="line",
#             data=dots)
# plt.savefig("relplot_images\image35.png")
"""
The third semantic, size, changes the width of the lines:
"""
# sns.relplot(x="time", y="firing_rate", size="coherence", style="choice", kind="line", data=dots)
# plt.savefig("relplot_images\image36.png")
"""
While the size variable will typically be numeric, it’s also possible to map a categorical
variable with the width of the lines. Be cautious when doing so, because it will be
difficult to distinguish much more than “thick” vs “thin” lines. However, dashes can 
be hard to perceive when lines have high-frequency variability, so using different
widths may be more effective in that case:
"""
# palette = sns.cubehelix_palette(light=.7, n_colors=6)
# sns.relplot(x="time",
#             y="firing_rate",
#             hue="coherence",
#             size="choice",
#             palette=palette,
#             kind="line", data=dots)
# plt.savefig("relplot_images\image37.png")
"""
Line plots are often used to visualize data associated with real dates and times. 
These functions pass the data down in their original format to the underlying matplotlib functions, 
and so they can take advantage of matplotlib’s ability to format dates in tick labels.
But all of that formatting will have to take place at the matplotlib layer, 
and you should refer to the matplotlib documentation to see how it works:
"""
# df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500), value=np.random.randn(500).cumsum()))
# g = sns.relplot(x="time", y="value", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.savefig("relplot_images\image38.png")
"""
You can also show the influence two variables this way: one by faceting on the columns 
and one by faceting on the rows. As you start adding more variables to the grid, 
you may want to decrease the figure size. Remember that the size FacetGrid is
parameterized by the height and aspect ratio of each facet:
"""
# sns.relplot(x="timepoint",
#             y="signal",
#             hue="subject",
#             col="region",
#             row="event",
#             height=3,
#             kind="line",
#             estimator=None,  # do nothing here
#             data=fmri)
# plt.savefig("relplot_images\image39.png")
"""
When you want to examine effects across many levels of a variable, it can be a good 
idea to facet that variable on the columns and then “wrap” the facets into the rows:
"""
# sns.relplot(x="timepoint",
#             y="signal",
#             hue="event",
#             style="event",
#             col="subject",
#             col_wrap=5,
#             height=3,
#             aspect=.75,
#             linewidth=2.5,
#             kind="line",
#             data=fmri.query("region == 'frontal'"))
# plt.savefig("relplot_images\image40.png")
"""
Show error bars instead of error bands and plot the standard error:
"""
# sns.lineplot(x="timepoint", y="signal", hue="event", err_style="bars", ci=68, data=fmri)
# plt.savefig("relplot_images\image41.png")
"""
Plot from a wide-form DataFrame:
"""
# sns.lineplot(data=wide_df)
# plt.savefig("relplot_images\image42.png")
"""
Plot from a list of Series:
"""
# list_data = [wide_df.loc[:"2005", "a"], wide_df.loc["2003":, "b"]]
# sns.lineplot(data=list_data)
# plt.savefig("relplot_images\image43.png")
"""
Plot a single Series, pass kwargs to plt.plot:
"""
# sns.lineplot(data=wide_df["a"], color="coral", label="line")
# plt.savefig("relplot_images\image44.png")


# plt.show()
