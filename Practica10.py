#------------------------------Joshua Hernández 1930693----------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("DataFrame/titanic2.csv")
text = " ".join(description for description in df.description)
w_c = WordCloud(collocations = False, background_color = 'purple').generate(text)
plt.imshow(w_c, interpolation='bilinear')
plt.axis("off")
# plt.savefig('imgs/W_Cloud.png')
plt.show()