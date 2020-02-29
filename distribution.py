"""distplot, kdeplot, rugplot, jointplot, pairplot"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import trapz

sns.set()

x = np.random.normal(size=100)
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
x2, y2 = np.random.multivariate_normal(mean, cov, 1000).T
x3, y3 = np.random.randn(2, 300)


"""
The most convenient way to take a quick look at a univariate distribution in seaborn 
is the distplot() function. By default, this will draw a histogram and fit a kernel 
density estimate (KDE).
"""
# sns.distplot(x)
# plt.savefig("distribution_images\image1")
"""
Histograms are likely familiar, and a hist function already exists in matplotlib. 
A histogram represents the distribution of data by forming bins along the range 
of the data and then drawing bars to show the number of observations that fall in each bin.
To illustrate this, let’s remove the density curve and add a rug plot, which draws 
a small vertical tick at each observation. You can make the rug plot itself with 
the rugplot() function, but it is also available in distplot():
"""
# sns.distplot(x, kde=False, rug=True)
# plt.savefig("distribution_images\image2")
"""
When drawing histograms, the main choice you have is the number of bins to 
use and where to place them. distplot() uses a simple rule to make a good 
guess for what the right number is by default, but trying more or fewer 
bins might reveal other features in the data:
"""
# sns.distplot(x, bins=20, kde=False, rug=True)
# plt.savefig("distribution_images\image3")
"""
The kernel density estimate may be less familiar, but it can be a useful 
tool for plotting the shape of a distribution. Like the histogram, the KDE 
plots encode the density of observations on one axis with height along the other axis:
"""
# sns.distplot(x, hist=False, rug=True)
# plt.savefig("distribution_images\image4")
"""
Drawing a KDE is more computationally involved than drawing a histogram. 
What happens is that each observation is first replaced with a normal 
(Gaussian) curve centered at that value:
"""
# y = np.random.normal(0, 1, size=30)
# bandwidth = 1.06 * y.std() * y.size ** (-1 / 5.)
# support = np.linspace(-4, 4, 200)
# kernels = []
# for y_i in y:
#     kernel = stats.norm(y_i, bandwidth).pdf(support)
#     kernels.append(kernel)
#     plt.plot(support, kernel, color="r")
# sns.rugplot(y, color=".2", linewidth=3)
# plt.savefig("distribution_images\image5")
"""
Next, these curves are summed to compute the value of the density at each 
point in the support grid. The resulting curve is then normalized so that 
the area under it is equal to 1:
"""
# y = np.random.normal(0, 1, size=30)
# bandwidth = 1.06 * y.std() * y.size ** (-1 / 5.)
# support = np.linspace(-4, 4, 200)
# kernels = []
# for y_i in y:
#     kernel = stats.norm(y_i, bandwidth).pdf(support)
#     kernels.append(kernel)
# density = np.sum(kernels, axis=0)
# density /= trapz(density, support)
# sns.lineplot(support, density)
# plt.savefig("distribution_images\image6")
"""
We can see that if we use the kdeplot() function in seaborn, we get the same curve. 
This function is used by distplot(), but it provides a more direct interface with 
easier access to other options when you just want the density estimate:
"""
# sns.kdeplot(x, shade=True)
# plt.savefig("distribution_images\image7")
"""
The bandwidth (bw) parameter of the KDE controls how tightly the estimation is fit 
to the data, much like the bin size in a histogram. It corresponds to the width of 
the kernels we plotted above. The default behavior tries to guess a good value using 
a common reference rule, but it may be helpful to try larger or smaller values:
"""
# sns.kdeplot(x)
# sns.kdeplot(x, bw=.2, label="bw: 0.2")
# sns.kdeplot(x, bw=2, label="bw: 2")
# plt.legend()
# plt.savefig("distribution_images\image8")
"""
As you can see above, the nature of the Gaussian KDE process means that estimation 
extends past the largest and smallest values in the dataset. It’s possible to control 
how far past the extreme values the curve is drawn with the cut parameter; however, 
this only influences how the curve is drawn and not how it is fit:
"""
# sns.kdeplot(x, shade=True, cut=0)
# sns.rugplot(x)
# plt.savefig("distribution_images\image9")
"""
You can also use distplot() to fit a parametric distribution to a dataset and 
visually evaluate how closely it corresponds to the observed data:
"""
# z = np.random.gamma(6, size=200)
# sns.distplot(z, kde=False, fit=stats.gamma)
# plt.savefig("distribution_images\image10")
"""
Change the color of all the plot elements:
"""
# sns.distplot(x, kde_kws={'kernel': 'gau'}, color="y")
# plt.savefig("distribution_images\image11")
"""
Pass specific parameters to the underlying plot functions:
"""
# sns.distplot(x,
#              rug=True,
#              bins=6,
#              rug_kws={"color": "g"},
#              kde_kws={"color": "k",
#                       "lw": 3,
#                       "label": "KDE"},
#              hist_kws={"histtype": "step",
#                        "linewidth": 3,
#                        "alpha": 1,
#                        "color": "g",
#                        "rwidth": (-3, 3)})
# plt.savefig("distribution_images\image12")
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
The most familiar way to visualize a bivariate distribution is a scatterplot, 
where each observation is shown with point at the x and y values. This is 
analgous to a rug plot on two dimensions. You can draw a scatterplot with 
the matplotlib plt.scatter function, and it is also the default kind of plot 
shown by the jointplot() function:
"""
# sns.jointplot(x="x", y="y", data=df)
# plt.savefig("distribution_images\image13")
"""
Add regression and kernel density fits:
"""
# sns.jointplot("total_bill", "tip", data=tips, kind="reg")
# plt.savefig("distribution_images\image14")
"""
Draw a smaller figure with more space devoted to the marginal plots:
"""
# sns.jointplot("total_bill", "tip", data=tips, height=5, ratio=3, color="g")
# plt.savefig("distribution_images\image15")
"""
Pass keyword arguments down to the underlying plots:
"""
# sns.jointplot("petal_length",
#               "sepal_length",
#               data=iris,
#               marginal_kws=dict(bins=15, rug=True),
#               annot_kws=dict(stat="r"),
#               s=40,
#               edgecolor="w",
#               linewidth=1)
# plt.savefig("distribution_images\image16")
"""
The bivariate analogue of a histogram is known as a “hexbin” plot, because it shows 
the counts of observations that fall within hexagonal bins. This plot works best with 
relatively large datasets. It’s available through the matplotlib plt.hexbin function 
and as a style in jointplot(). It looks best with a white background:
"""
# with sns.axes_style("white"):
#     sns.jointplot(x=x2, y=y2, kind="hex", color="k")
# plt.savefig("distribution_images\image17")
"""
Pass vectors in directly without using Pandas, then name the axes:
"""
# with sns.axes_style("white"):
#     sns.jointplot(x3, y3, kind="hex").set_axis_labels("x", "y")
# plt.savefig("distribution_images\image18")
"""
It is also possible to use the kernel density estimation procedure described above to 
visualize a bivariate distribution. In seaborn, this kind of plot is shown with a 
contour plot and is available as a style in jointplot():
"""
# g = sns.jointplot(x="x", y="y", data=df, kind="kde")
# g.ax_joint.collections[0].set_alpha(0)
# plt.savefig("distribution_images\image19")
"""
You can also draw a two-dimensional kernel density plot with the kdeplot() function. 
This allows you to draw this kind of plot onto a specific (and possibly already existing) 
matplotlib axes, whereas the jointplot() function manages its own figure:
"""
# f, ax = plt.subplots(figsize=(6, 6))
# sns.kdeplot(df.x, df.y, ax=ax)
# sns.rugplot(df.x, color="g", ax=ax)
# sns.rugplot(df.y, vertical=True, ax=ax)
# plt.savefig("distribution_images\image20")
"""
If you wish to show the bivariate density more continuously, you can simply 
increase the number of contour levels:
"""
# f, ax = plt.subplots(figsize=(6, 6))
# cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
# sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=100, shade=True)
# plt.savefig("distribution_images\image21")
"""
The jointplot() function uses a JointGrid to manage the figure. For more flexibility, 
you may want to draw your figure by using JointGrid directly. jointplot() returns the 
JointGrid object after plotting, which you can use to add more layers or to tweak other
aspects of the visualization:
"""
# g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
# g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
# g.ax_joint.collections[0].set_alpha(0)
# g.set_axis_labels("$X$", "$Y$")
# plt.savefig("distribution_images\image22")
"""
Replace the scatterplots and histograms with density estimates and align the 
marginal Axes tightly with the joint Axes:
"""
# with sns.axes_style("white"):
#     g = sns.jointplot("sepal_width",
#                       "petal_length",
#                       data=iris,
#                       kind="kde",
#                       space=0,
#                       color="g")
#     g.ax_joint.collections[0].set_alpha(0)
# plt.savefig("distribution_images\image23")
"""
Draw a scatterplot, then add a joint density estimate:
"""
# with sns.axes_style("white"):
#     sns.jointplot("sepal_length",
#                   "sepal_width",
#                   data=iris,
#                   color="k").\
#         plot_joint(sns.kdeplot, zorder=0, n_levels=6)
# plt.savefig("distribution_images\image24")
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""
To plot multiple pairwise bivariate distributions in a dataset, you can use the pairplot() 
function. This creates a matrix of axes and shows the relationship for each pair of columns 
in a DataFrame. by default, it also draws the univariate distribution of each variable 
on the diagonal Axes:
"""
# sns.pairplot(iris)
# plt.savefig("distribution_images\image25")
"""
Much like the relationship between jointplot() and JointGrid, the pairplot() function 
is built on top of a PairGrid object, which can be used directly for more flexibility:
"""
# g = sns.PairGrid(iris)
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.kdeplot, n_levels=6)
# plt.savefig("distribution_images\image26")
"""
Show different levels of a categorical variable by the color of plot elements:
"""
# sns.pairplot(iris, hue="species")
# plt.savefig("distribution_images\image27")
"""
Use a different color palette:
"""
# sns.pairplot(iris, hue="species", palette="husl")
# plt.savefig("distribution_images\image28")
"""
Use different markers for each level of the hue variable:
"""
# sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
# plt.savefig("distribution_images\image29")
"""
Plot a subset of variables:
"""
# sns.pairplot(iris, vars=["sepal_width", "sepal_length"])
# plt.savefig("distribution_images\image30")
"""
Draw larger plots:
"""
# sns.pairplot(iris, height=3, vars=["sepal_width", "sepal_length"])
# plt.savefig("distribution_images\image31")
"""
Plot different variables in the rows and columns:
"""
# sns.pairplot(iris, x_vars=["sepal_width", "sepal_length"], y_vars=["petal_width", "petal_length"])
# plt.savefig("distribution_images\image32")
"""
Use kernel density estimates for univariate plots:
"""
# sns.pairplot(iris, diag_kind="kde")
# plt.savefig("distribution_images\image33")
"""
Pass keyword arguments down to the underlying functions (it may be easier to use PairGrid directly):
"""
# sns.pairplot(iris,
#              diag_kind="kde",
#              markers="+",
#              plot_kws=dict(s=50, edgecolor="p", linewidth=2),
#              diag_kws=dict(shade=False))
# plt.savefig("distribution_images\image34")


plt.show()
sns.scatterplot()