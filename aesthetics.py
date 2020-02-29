import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = np.random.normal(size=(20, 6)) + np.arange(6) / 2

"""
Let’s define a simple function to plot some offset sine waves, which will 
help us see the different stylistic parameters we can tweak.
"""
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
"""
This is what the plot looks like with matplotlib defaults:
"""
# sinplot()
# plt.savefig('aesthetics_images\\image1.png')
"""
To switch to seaborn defaults, simply call the set() function.
"""
# sns.set()
# sinplot()
# plt.savefig('aesthetics_images\\image2.png')
"""
Seaborn splits matplotlib parameters into two independent groups. The first group 
sets the aesthetic style of the plot, and the second scales various elements of 
the figure so that it can be easily incorporated into different contexts.
The interface for manipulating these parameters are two pairs of functions. 
To control the style, use the axes_style() and set_style() functions. To scale 
the plot, use the plotting_context() and set_context() functions. In both cases, 
the first function returns a dictionary of parameters and the second sets the matplotlib defaults.
"""

"""
There are five preset seaborn themes: darkgrid, whitegrid, dark, white, and ticks. 
They are each suited to different applications and personal preferences. The default
theme is darkgrid. As mentioned above, the grid helps the plot serve as a lookup
table for quantitative information, and the white-on grey helps to keep the grid
from competing with lines that represent data. The whitegrid theme is similar, but
it is better suited to plots with heavy data elements:
"""
# sns.set_style("whitegrid")
# sns.boxplot(data=data)
# plt.savefig('aesthetics_images\\image3.png')
"""
For many plots, (especially for settings like talks, where you primarily want to 
use figures to provide impressions of patterns in the data), the grid is less necessary.
"""
# sns.set_style("dark")
# sinplot()
# plt.savefig('aesthetics_images\\image4.png')

# sns.set_style("white")
# sinplot()
# plt.savefig('aesthetics_images\\image5.png')

# sns.set_style("ticks")
# sinplot()
# plt.savefig('aesthetics_images\\image6.png')
"""
Both the white and ticks styles can benefit from removing the top and right axes spines,
which are not needed. The seaborn function despine() can be called to remove them:
"""
# sinplot()
# sns.despine()
# plt.savefig('aesthetics_images\\image7.png')
"""
Some plots benefit from offsetting the spines away from the data, which can also be 
done when calling despine(). When the ticks don’t cover the whole range of the axis, 
the trim parameter will limit the range of the surviving spines.
"""
# f, ax = plt.subplots()
# sns.violinplot(data=data)
# sns.despine(offset=10, trim=True)
# plt.savefig('aesthetics_images\\image8.png')
"""
You can also control which spines are removed with additional arguments to despine():
"""
# sns.set_style("whitegrid")
# sns.boxplot(data=data, palette="deep")
# sns.despine(left=True)
# plt.savefig('aesthetics_images\\image9.png')
"""
Although it’s easy to switch back and forth, you can also use the axes_style() function 
in a with statement to temporarily set plot parameters. This also allows you to make 
figures with differently-styled axes:
"""
# f = plt.figure()
# with sns.axes_style("darkgrid"):
#     ax = f.add_subplot(1, 2, 1)
#     sinplot()
# ax = f.add_subplot(1, 2, 2)
# sinplot(-1)
# plt.savefig('aesthetics_images\\image10.png')
"""
If you want to customize the seaborn styles, you can pass a dictionary of parameters 
to the rc argument of axes_style() and set_style(). Note that you can only override 
the parameters that are part of the style definition through this method. (However, 
the higher-level set() function takes a dictionary of any matplotlib parameters).
If you want to see what parameters are included, you can just call the function with 
no arguments, which will return the current settings:
"""
# print(sns.axes_style())
"""
You can then set different versions of these parameters:
"""
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
# sinplot()
# plt.savefig('aesthetics_images\\image11.png')
"""
A separate set of parameters control the scale of plot elements, which should let
you use the same code to make plots that are suited for use in settings where
larger or smaller plots are appropriate.
First let’s reset the default parameters by calling set():
"""
# sns.set()
"""
The four preset contexts, in order of relative size, are paper, notebook, talk, 
and poster. The notebook style is the default, and was used in the plots above.
"""
# sns.set_context("paper")
# sinplot()
# plt.savefig('aesthetics_images\\image12.png')
#
# sns.set_context("talk")
# sinplot()
# plt.savefig('aesthetics_images\\image13.png')

# sns.set_context("poster")
# sinplot()
# plt.savefig('aesthetics_images\\image14.png')
"""
Most of what you now know about the style functions should transfer to the context functions.
You can call set_context() with one of these names to set the parameters, and you can override 
the parameters by providing a dictionary of parameter values.
You can also independently scale the size of the font elements when changing the context. 
(This option is also available through the top-level set() function).
"""
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
# sinplot()
# plt.savefig('aesthetics_images\\image15.png')





# plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(14)
y = np.sin(x / 2)

plt.step(x, y + 2, label='pre (default)')
plt.plot(x, y + 2, 'C0o', alpha=0.5)

plt.step(x, y + 1, where='mid', label='mid')
plt.plot(x, y + 1, 'C1o', alpha=0.5)

plt.step(x, y, where='post', label='post')
plt.plot(x, y, 'C2o', alpha=0.5)

plt.legend(title='Parameter where:')
plt.show()