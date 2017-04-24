# Util Package

## Introduction
   This package is designed for util toolbox of this repository, [image_util](image_util.py) provide 
   a handler for image format transforming, such as transform raw image into binary file. Just like [cifar dataset](http://www.cs.toronto.edu/~kriz/cifar.html).
   
    The bin file format is designed as follows:
      Exp : {
                "data": array(
                    [
                        [255, 214, ..., 30],
                        ...
                        [204, 254, ..., 69],
                    ], dtype=uint8
                ),
                "labels": [0, 2, 1, 1, 2, ..., 3],
                "batch_label": "training bat 1 of 1 of HZAU faces",
                "filenames": ["1.png", ..., "10086.png"]
            }      
__data__ stands for image r, g, b channel gray value, each line's size is 62280(width * height * depth).

__labels__ represents the face score:

    0 means "ugly",
    1 means "medium down",
    2 means "medium",
    3 means "medium up",
    4 means "beautiful"

__batch_label__ represents the description of this binary file.

__filenames__ represents all image files.