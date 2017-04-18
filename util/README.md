# Util Package

## Intro
   This package is designed for util toolbox of this repository, [image_util](image_util.py) provide 
   a handler for image format transforming, such as transform raw image into binary file. Just like [cifar dataset](http://www.cs.toronto.edu/~kriz/cifar.html).
   
    The bin file format is designed as follows:
      Exp : <1 x label><62280 x pixel>
            <1 x label><62280 x pixel>
            ...
            <1 x label><62280 x pixel>
            
            where 1 stands for the label byte(0, 1, 2, 3, 4),
            and 62280 stands for 144(width) * 144(height) * depth(channel)
      
   We design this format