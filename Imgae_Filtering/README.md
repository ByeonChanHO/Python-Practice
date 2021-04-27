#Part 1 : Gaussian Filtering
##1. (10 points) We follow the convention that 2D filters always have an odd number of rows and columns (so that the center row/column of the filter is well-defined). As a simple warm-up exercise, write a Python function, ‘boxfilter(n)’, that returns a box filter of size n by n. You should check that n is odd, checking and signaling an error with an ‘assert’ statement. The filter should be a Numpy array. For example, your function should work as follows:
**Show the results of your boxfilter(n) function for the cases n=3, n=4, and n=7**

![image](https://user-images.githubusercontent.com/38696775/116166107-35f22180-a738-11eb-8805-7e01defbab00.png)


##2. (10 points) Write a Python function, ‘gauss1d(sigma)’, that returns a 1D Gaussian filter for a given value of sigma. The filter should be a 1D Numpy array with length 6 times sigma rounded up to the next odd integer. Each value of the filter can be computed from the Gaussian function, 𝑒𝑥𝑝(−𝑥^2/(2 ∗ 𝑠𝑖𝑔𝑚𝑎^2 )), where x is the distance of an array value from the center. This formula for the Gaussian ignores the constant factor. Therefore, you should normalize the values in the filter so that they sum to 1.
**Show the filter values produced for sigma values of 0.3, 0.5, 1, and 2**
 
![image](https://user-images.githubusercontent.com/38696775/116166113-3b4f6c00-a738-11eb-9d6f-b4a1d91fccc0.png)
![image](https://user-images.githubusercontent.com/38696775/116166118-3e4a5c80-a738-11eb-9e5f-4560da171157.png)






##3. (10 points) Create a Python function ‘gauss2d(sigma)’ that returns a 2D Gaussian filter for a given value of sigma. The filter should be a 2D Numpy array. Use ‘np.outer‘ with the 1D array from the function gauss1d(sigma). You also need to normalize the values in the filter so that they sum to 1. 
**Show the 2D Gaussian filter for sigma values of 0.5 and 1.**
![image](https://user-images.githubusercontent.com/38696775/116166169-63d76600-a738-11eb-897b-66df2dbc4e3f.png)


##4. (30 points) 
###(a) Write a function ‘convolve2d(array, filter)’ that takes in an image (stored in `array`) and a filter, and performs convolution to the image with zero paddings (thus, the image sizes of input and output are the same). 
Both input variables are in type `np.float32`. Note that for this implementation you should use two for-loops to iterate through each neighborhood.
![image](https://user-images.githubusercontent.com/38696775/116166191-6df96480-a738-11eb-9f4d-b7a3b0de902f.png)



###(b) Write a function ‘gaussconvolve2d(array,sigma)’ that applies Gaussian convolution to a 2D array for the given value of sigma. The result should be a 2D array. Do this by first generating a filter with your ‘gauss2d’, and then applying it to the array with ‘convolve2d(array, filter)’
 ![image](https://user-images.githubusercontent.com/38696775/116166197-72be1880-a738-11eb-8efc-0bc2b3beebcd.png)




###(c) Apply your ‘gaussconvolve2d’ with a sigma of 3 on the image of the dog (attached in PLATO). Load this image into Python, convert it to a greyscale, Numpy array and run your ‘gaussconvolve2d’ (with a sigma of 3). Note, as mentioned in class, for any image filtering or processing operations converting image to a double array format will make your life a lot easier and avoid various artifacts. Once all processing operations are done, you will need to covert the array back to unsigned integer format for storage and display
![image](https://user-images.githubusercontent.com/38696775/116166205-76ea3600-a738-11eb-9219-13daa18faf1d.png)
![image](https://user-images.githubusercontent.com/38696775/116166210-7b165380-a738-11eb-9517-a860014ec2ad.png)
![image](https://user-images.githubusercontent.com/38696775/116166216-7e114400-a738-11eb-95b8-255922689463.png)


###(d) Use PIL to show both the original and filtered images.
 ![image](https://user-images.githubusercontent.com/38696775/116166221-81a4cb00-a738-11eb-8e93-4aeb58d288bd.png)
![image](https://user-images.githubusercontent.com/38696775/116166223-84072500-a738-11eb-903e-810a60ab2b2b.png)











#Part 2: Hybrid Images
##1. (12 points) Choose an appropriate sigma and create a blurred version of the one of the paired images (choose one other pair, NOT the einstein/marilyn). For this to work you will need to choose a relatively large sigma and filter each of the three color channels (RGB) separately, then compose the channels back to the color image to display. Note, you should use the same sigma for all color channels.
![image](https://user-images.githubusercontent.com/38696775/116166232-879aac00-a738-11eb-867d-5dc923d45753.png)
![image](https://user-images.githubusercontent.com/38696775/116166238-89fd0600-a738-11eb-88c4-af43c724e757.png)
![image](https://user-images.githubusercontent.com/38696775/116166243-8cf7f680-a738-11eb-9b36-0a9e77d0d857.png)
![image](https://user-images.githubusercontent.com/38696775/116166266-9aad7c00-a738-11eb-84b7-b89cd762b7b2.png)




##2. (12 points) Choose an appropriate sigma (it is suggested to use the same as above) and create a high frequency version of the second from the two the paired images. Again you will operate on each of the color channels separately and use same sigma for all channels. High frequency filtered image is obtained by first computing a low frequency Gaussian filtered image and then subtracting it from the original. The high frequency image is actually zero-mean with negative values so it is visualized by adding 128 (if you re-scaled the original image to the range between 0 and 1, then add 0.5 for visualization). In the resulting visualization illustrated below, bright values are positive and dark values are negative.

![image](https://user-images.githubusercontent.com/38696775/116166273-9da86c80-a738-11eb-94bc-e51606442724.png)
![image](https://user-images.githubusercontent.com/38696775/116166278-a00ac680-a738-11eb-9992-86daf22890f1.png)
![image](https://user-images.githubusercontent.com/38696775/116166288-a5681100-a738-11eb-9aea-be087661927f.png)


##3. (16 points) Now simply add the low and high frequency images (per channel). Note, the high frequency image that you add, should be the originally computed high frequency image (without adding 128; this addition is only done for visualization in the part above). You may get something like the following as a result:
 
![image](https://user-images.githubusercontent.com/38696775/116166347-bf095880-a738-11eb-9387-ba870f09d4c0.png)
![image](https://user-images.githubusercontent.com/38696775/116166356-c3ce0c80-a738-11eb-9f8b-afc4bf4d057e.png)

