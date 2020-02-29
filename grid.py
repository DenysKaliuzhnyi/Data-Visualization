"""FacetGrid, PairGrid"""
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

sns.set()

tips = sns.load_dataset("tips")
attend = sns.load_dataset("attention").query("subject <= 12")
iris = sns.load_dataset("iris")


def quantile_plot(x, **kwargs):
    qntls, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, qntls, **kwargs)


def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)


"""
The class is used by initializing a FacetGrid object with a dataframe and the names
of the variables that will form the row, column, or hue dimensions of the grid. 
These variables should be categorical or discrete, and then the data at each level 
of the variable will be used for a facet along that axis. For example, say we wanted 
to examine differences between lunch and dinner in the tips dataset.
"""
# g = sns.FacetGrid(tips, col="time")
# plt.savefig('grid_images\\image1.png')
"""
Initializing the grid like this sets up the matplotlib figure and 
axes, but doesn’t draw anything on them.
The main approach for visualizing data on this grid is with the 
FacetGrid.map() method. Provide it with a plotting function and 
the name(s) of variable(s) in the dataframe to plot. Let’s look 
at the distribution of tips in each of these subsets, using a histogram.
"""
# g = sns.FacetGrid(tips, col="time")
# g.map(plt.hist, "tip")
# plt.savefig('grid_images\\image2.png')
"""
This function will draw the figure and annotate the axes, hopefully producing 
a finished plot in one step. To make a relational plot, just pass multiple 
variable names. You can also provide keyword arguments, which will be passed 
to the plotting function:
"""
# g = sns.FacetGrid(tips, col="sex", hue="smoker")
# g.map(plt.scatter, "total_bill", "tip", alpha=.7)
# g.add_legend()
# plt.savefig('grid_images\\image3.png')
"""
There are several options for controlling the look of the grid 
that can be passed to the class constructor.
"""
# g = sns.FacetGrid(tips,
#                   row="smoker",
#                   col="time",
#                   margin_titles=True)
# g.map(sns.regplot,
#       "size",
#       "total_bill",
#       color=".3",
#       fit_reg=False,
#       x_jitter=.1)
# plt.savefig('grid_images\\image4.png')
"""
Note that margin_titles isn’t formally supported by the matplotlib API, 
and may not work well in all cases. In particular, it currently can’t 
be used with a legend that lies outside of the plot.
The size of the figure is set by providing the height of each facet, 
along with the aspect ratio:
"""
# g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
# g.map(sns.barplot, "sex", "total_bill")
# plt.savefig('grid_images\\image5.png')
"""
The default ordering of the facets is derived from the information in the DataFrame. 
If the variable used to define facets has a categorical type, then the order of 
the categories is used. Otherwise, the facets will be in the order of appearance 
of the category levels. It is possible, however, to specify an ordering of any 
facet dimension with the appropriate *_order parameter:
"""
# ordered_days = tips.day.value_counts().index
# print(ordered_days)
# g = sns.FacetGrid(tips,
#                   row="day",
#                   row_order=ordered_days,
#                   height=1.7,
#                   aspect=4,)
# g.map(sns.distplot, "total_bill", hist=False, rug=True)
# plt.savefig('grid_images\\image6.png')
"""
Any seaborn color palette (i.e., something that can be passed to color_palette() 
can be provided. You can also use a dictionary that maps the names of values in 
the hue variable to valid matplotlib colors
"""
# pal = dict(Lunch="seagreen", Dinner="gray")
# g = sns.FacetGrid(tips, hue="time", palette=pal, height=5)
# g.map(plt.scatter,
#       "total_bill",
#       "tip",
#       s=50,
#       alpha=.7,
#       linewidth=.5,
#       edgecolor="white")
# g.add_legend()
# plt.savefig('grid_images\\image7.png')
"""
You can also let other aspects of the plot vary across levels of the hue variable, 
which can be helpful for making plots that will be more comprehensible when printed 
in black-and-white. To do this, pass a dictionary to hue_kws where keys are the names 
of plotting function keyword arguments and values are lists of keyword values, one 
for each level of the hue variable.
"""
# g = sns.FacetGrid(tips,
#                   hue="sex",
#                   palette="Set1",
#                   height=5,
#                   hue_kws={"marker": ["^", "v"]})
# g.map(plt.scatter,
#       "total_bill",
#       "tip",
#       s=100,
#       linewidth=.5,
#       edgecolor="white")
# g.add_legend()
# plt.savefig('grid_images\\image8.png')
"""
If you have many levels of one variable, you can plot it along the columns but “wrap” 
them so that they span multiple rows. When doing this, you cannot use a row variable.
"""
# g = sns.FacetGrid(attend,
#                   col="subject",
#                   col_wrap=4,
#                   height=2,
#                   ylim=(0, 10))
# g.map(sns.pointplot,
#       "solutions",
#       "score",
#       color=".3",
#       ci=None)
# plt.savefig('grid_images\\image9.png')
"""
Once you’ve drawn a plot using FacetGrid.map() (which can be called multiple times), 
you may want to adjust some aspects of the plot. There are also a number of methods 
on the FacetGrid object for manipulating the figure at a higher level of abstraction. 
The most general is FacetGrid.set(), and there are other more specialized methods 
like FacetGrid.set_axis_labels(), which respects the fact that interior facets do 
not have axis labels. For example:
"""
# with sns.axes_style("white"):
#     g = sns.FacetGrid(tips,
#                       row="sex",
#                       col="smoker",
#                       margin_titles=True,
#                       height=2.5)
# g.map(plt.scatter,
#       "total_bill",
#       "tip",
#       color="#334488",
#       edgecolor="white",
#       lw=.5)
# g.set_axis_labels("Total bill (US Dollars)", "Tip")
# g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])
# g.fig.subplots_adjust(wspace=.02, hspace=.02)
# plt.savefig('grid_images\\image10.png')
"""
For even more customization, you can work directly with the underling matplotlib 
Figure and Axes objects, which are stored as member attributes at fig and axes 
(a two-dimensional array), respectively. When making a figure without row or 
column faceting, you can also use the ax attribute to directly access the single axes.
"""
# g = sns.FacetGrid(tips,
#                   col="smoker",
#                   margin_titles=True,
#                   height=4)
# g.map(plt.scatter,
#       "total_bill",
#       "tip",
#       color="#338844",
#       edgecolor="white",
#       s=50,
#       lw=1)
# for ax in g.axes.flat:
#     ax.plot((0, 50), (0, .2 * 50), c=".2", ls="--")
# g.set(xlim=(0, 60), ylim=(0, 14))
# plt.savefig('grid_images\\image11.png')
"""
Specify the order for plot elements:
"""
# bins = np.arange(0, 65, 5)
# g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])
# g = g.map(plt.hist, "total_bill", bins=bins, color="m")
# plt.savefig('grid_images\\image12.png')
"""
Use a different color palette:
"""
# kws = dict(s=50, linewidth=.5, edgecolor="w")
# g = sns.FacetGrid(tips,
#                   col="sex",
#                   hue="time",
#                   palette="Set1",
#                   hue_order=["Dinner", "Lunch"])
# g = (g.map(plt.scatter, "total_bill", "tip", **kws).add_legend())
# plt.savefig('grid_images\\image13.png')
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
You’re not limited to existing matplotlib and seaborn functions when using FacetGrid. 
However, to work properly, any function you use must follow a few rules:
It must plot onto the “currently active” matplotlib Axes. This will be true of functions 
in the matplotlib.pyplot namespace, and you can call plt.gca to get a reference to the 
current Axes if you want to work directly with its methods.
It must accept the data that it plots in positional arguments. Internally, FacetGrid 
will pass a Series of data for each of the named positional arguments passed to FacetGrid.map().
It must be able to accept color and label keyword arguments, and, ideally, it will do 
something useful with them. In most cases, it’s easiest to catch a generic dictionary 
of **kwargs and pass it along to the underlying plotting function.
Let’s look at minimal example of a function you can plot with. This function will just 
take a single vector of data for each facet:
"""
# g = sns.FacetGrid(tips, col="sex", height=4)
# g.map(quantile_plot, "total_bill")
# plt.savefig('grid_images\\image14.png')
"""
If we want to make a bivariate plot, you should write the function so 
that it accepts the x-axis variable first and the y-axis variable second:
"""
# g = sns.FacetGrid(tips, col="smoker", height=4)
# g.map(qqplot, "total_bill", "tip")
# plt.savefig('grid_images\\image15.png')
"""
Because plt.scatter accepts color and label keyword arguments and does 
the right thing with them, we can add a hue facet without any difficult
"""
# g = sns.FacetGrid(tips, hue="time", col="sex", height=4)
# g.map(qqplot, "total_bill", "tip")
# g.add_legend()
# plt.savefig('grid_images\\image16.png')
"""
This approach also lets us use additional aesthetics to distinguish the 
levels of the hue variable, along with keyword arguments that won’t be 
dependent on the faceting variables:
"""
# g = sns.FacetGrid(tips,
#                   hue="time",
#                   col="sex",
#                   height=4,
#                   hue_kws={"marker": ["s", "D"]})
# g.map(qqplot,
#       "total_bill",
#       "tip",
#       s=40,
#       edgecolor="w")
# g.add_legend()
# plt.savefig('grid_images\\image17.png')
"""
Sometimes, though, you’ll want to map a function that doesn’t work the way you 
expect with the color and label keyword arguments. In this case, you’ll want 
to explicitly catch them and handle them in the logic of your custom function. 
For example, this approach will allow use to map plt.hexbin, which otherwise 
does not play well with the FacetGrid API:
"""
# with sns.axes_style("dark"):
#     g = sns.FacetGrid(tips, hue="time", col="time", height=4)
# g.map(hexbin, "total_bill", "tip", extent=[0, 50, 0, 10])
# plt.savefig('grid_images\\image18.png')
"""
Define a custom function that uses a DataFrame object and 
accepts column names as positional variables:
"""
# df = pd.DataFrame(
#     data=np.random.randn(90, 4),
#     columns=pd.Series(list("ABCD"), name="walk"),
#     index=pd.date_range("2015-01-01", "2015-03-31", name="date"))
# df = df.cumsum(axis=0).stack().reset_index(name="val")
#
# def dateplot(x, y, **kwargs):
#     ax = plt.gca()
#     data = kwargs.pop("data")
#     data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
#
# g = sns.FacetGrid(df, col="walk", col_wrap=2, height=3.5)
# g = g.map_dataframe(dateplot, "date", "val")
# plt.savefig('grid_images\\image19.png')
"""
Use a different template for the facet titles:
"""
# g = sns.FacetGrid(tips, col="size", col_wrap=3)
# g = (g.map(plt.hist, "tip", bins=np.arange(0, 13), color="c").set_titles("{col_name} diners"))
# plt.savefig('grid_images\\image20.png')
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
The basic usage of the class is very similar to FacetGrid. First you initialize 
the grid, then you pass plotting function to a map method and it will be called 
on each subplot. There is also a companion function, pairplot() that trades off 
some flexibility for faster plotting.
"""
# g = sns.PairGrid(iris)
# g.map(plt.scatter)
# plt.savefig('grid_images\\image21.png')
"""
It’s possible to plot a different function on the diagonal to show the univariate 
distribution of the variable in each column. Note that the axis ticks won’t
correspond to the count or density axis of this plot, though.
"""
# g = sns.PairGrid(iris)
# g.map_diag(plt.hist)
# g.map_offdiag(plt.scatter)
# plt.savefig('grid_images\\image22.png')
"""
A very common way to use this plot colors the observations by a separate categorical 
variable. For example, the iris dataset has four measurements for each of three 
different species of iris flowers so you can see how they differ.
"""
# g = sns.PairGrid(iris, hue="species",  hue_kws={"marker": ["o", "s", "D"]})
# g.map_diag(plt.hist)
# g.map_offdiag(plt.scatter, linewidths=1, edgecolor="w", s=40)
# g.add_legend()
# plt.savefig('grid_images\\image23.png')
"""
By default every numeric column in the dataset is used, but you 
can focus on particular relationships if you want.
"""
# g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")
# g.map(plt.scatter)
# plt.savefig('grid_images\\image24.png')
"""
It’s also possible to use a different function in the upper and lower 
triangles to emphasize different aspects of the relationship.
"""
# g = sns.PairGrid(iris)
# g.map_upper(plt.scatter)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot, lw=3, legend=False)
# plt.savefig('grid_images\\image25.png')
"""
The square grid with identity relationships on the diagonal is actually just a 
special case, and you can plot with different variables in the rows and columns.
"""
# g = sns.PairGrid(tips,
#                  y_vars=["tip"],
#                  x_vars=["total_bill", "size"],
#                  height=4)
# g.map(sns.regplot, color=".3")
# g.set(ylim=(-1, 11), yticks=[0, 5, 10])
# plt.savefig('grid_images\\image26.png')
"""
Of course, the aesthetic attributes are configurable. For instance, you can 
use a different palette (say, to show an ordering of the hue variable) and 
pass keyword arguments into the plotting functions.
"""
# g = sns.PairGrid(tips, hue="size", palette="GnBu_d")
# g.map(plt.scatter, s=50, edgecolor="white")
# g.add_legend()
# plt.savefig('grid_images\\image27.png')
"""
Pass additional keyword arguments to the functions
"""
# g = sns.PairGrid(iris)
# g = g.map_diag(plt.hist, histtype='step', edgecolor="m")
# g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)
# plt.savefig('grid_images\\image28.png')


plt.show()

