# Captcha Solver

Captcha solver using **YOLO v3**. Has an accuracy of **98.166%** on the test dataset.
 
* The trained weights and config file can be found **[here](https://drive.google.com/open?id=1_OH7LXR82GEeDKAa7ntEuI_HpG0e3WkH)**.

* The annotation of dataset is done manually using BBox Label Tool. We finf the cntre's x and y coordinates and normalized width and height.

## Example of data annotation

![00012.jpg](00012.txt)

| Object-d      | x-coordinate  | y-coordinate  | width     | height   |
| ------------- |:-------------:| ------------:	| ---------:|---------:|
| 0      	| 0.285185 	| 0.528571 	| 0.111111  | 0.600000 |
| 0      	| 0.396296      | 0.528571 	| 0.111111  | 0.542857 |
| 0 		| 0.500000      | 0.457143 	| 0.111111  | 0.571429 |
| 1		| 0.607407	| 0.514286	| 0.118519  | 0.571429 |
| 2		| 0.725926	| 0.557143	| 0.118519  | 0.600000 |

## How to run the code

```python

python3 captcha-single.py --image test.jpg # replace with your image

python3 captcha.py #for running on images in test/Solved-600 folder
```
