#!/usr/bin/env python
# coding: utf-8

# # 1. Matplotlib & PIL

# ### Import Libraries

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Use the PIL for the image
# 
# https:/pillow.readthedocs.io/en/stable/reference/Image.html

# In[4]:


from PIL import Image


# In[5]:


img=Image.open('images/rock.jpg')


# In[6]:


img


# ### Rotate the Image

# In[7]:


img.rotate(-90)


# ### Check the Type of the Image

# In[8]:


type(img)


# ### Turn the image into an array

# In[9]:


img_array=np.asarray(img)


# In[10]:


type(img_array)


# ### Get the Height, Width & Channels

# In[11]:


img_array.shape


# In[12]:


plt.imshow(img_array)


# ### R G B channels
# 
# <b>Red</b> channel is in position No <b>0</b>, <br>
# <b>Green</b> channel is in position No <b>1</b>, <br>
# <b>Blue</b> channel is in position No <b>2</b>.
# 
# The Colour Values are from <b>0</b> == <u>no colour </u>from the channel, to <b>255</b> == <u>full colour</u> from the channel.

# In[13]:


img_test=img_array.copy()


# ### Only Red channel

# In[14]:


plt.imshow(img_test[:, :, 0])


# ##### Scale Red channel to Gray

# In[15]:


plt.imshow(img_test[:, :, 0], cmap='gray')


# ### Only Green channel

# In[16]:


plt.imshow(img_test[:, :, 1])


# ##### Scale Green channel to Gray

# In[17]:


plt.imshow(img_test[:, :, 1], cmap='gray')


# ### Only Blue channel

# In[18]:


plt.imshow(img_test[:, :, 2])


# ##### Scale Blue channel to Gray

# In[19]:


plt.imshow(img_test[:, :, 2], cmap='gray')


# ### Remove Red Colour

# In[20]:


img_test[:, :, 0]=0
plt.imshow(img_test)


# ### Remove Green Colour

# In[21]:


img_test=img_array.copy()
img_test[:, :, 1]=0
plt.imshow(img_test)


# ### Remove Blue Colour

# In[22]:


img_test=img_array.copy()
img_test[:, :, 2]=0
plt.imshow(img_test)


# # 2. OpenCV

# ### Import Libraries

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import OpenCV

# In[24]:


import cv2


# ##### Get the image with the <b>imread</b>

# In[25]:


img=cv2.imread('images/rock.jpg')


# ### Always check the type

# In[26]:


type(img)


# ##### Get the image <b>shape</b>

# In[27]:


img.shape


# ##### Let's see the image with the <b>imshow</b>

# In[28]:


plt.imshow(img)


# ##### Until now we were working with <U>Matplotlib</u> and <b>RGB</b>. <br>
# ##### <u>OpenCV</u> is reading the channel as <b>BGR</b>. <br>
# ##### We will <b>convert</b> OpenCV to the channels of the photo.

# In[29]:


img_fix=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[30]:


plt.imshow(img_fix)


# ##### Scale it to <b>Gray</b> and check the <b>Shape</b>

# In[31]:


img_gray=cv2.imread('images/rock.jpg', cv2.IMREAD_GRAYSCALE)


# In[32]:


img_gray.shape


# In[33]:


plt.imshow(img_gray, cmap="gray")


# ### Resize the image
# 
# ##### To resize the image we change the Height Width --> Width Height

# In[34]:


img_new=cv2.resize(img_fix, (1000, 400))


# In[35]:


plt.imshow(img_new)


# ### Resize with Ratio

# In[36]:


width_ratio=0.5
height_ratio=0.5


# In[38]:


img2=cv2.resize(img_fix, (0,0), img_fix, width_ratio, height_ratio)


# In[39]:


plt.imshow(img2)


# In[40]:


img2.shape


# ### Flip on Horizontal Axis

# In[42]:


img_3=cv2.flip(img_fix, 0)
plt.imshow(img_3)


# ### Flip on Vertical Axis

# In[43]:


img_3=cv2.flip(img_fix, 1)
plt.imshow(img_3)


# ### Flip on Horizontal and on Vertical Axis

# In[44]:


img_3=cv2.flip(img_fix, -1)
plt.imshow(img_3)


# ### Change the size of our canva

# In[48]:


last_img=plt.figure(figsize=(10,7))
ilp=last_img.add_subplot(111)
ilp.imshow(img_fix)


# In[ ]:




