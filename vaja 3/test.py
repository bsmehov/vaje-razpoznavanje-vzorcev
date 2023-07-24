import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

if __name__ == "__main__":

 list1 = [1,2,3,4,5,6]
 list2 = [3, 5, 7, 9]
 list11 = set(list1)
 list21 = set(list2)
 list = list11.intersection(list21)
 print(list)
