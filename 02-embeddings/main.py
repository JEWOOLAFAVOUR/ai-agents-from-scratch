import numpy as np


cat = np.array([0.21, -0.44, 0.83, 0.17])

dog = np.array([0.19, -0.39, 0.79, 0.22])

car = np.array([-0.72, 0.11, -0.31, 0.92])


print("cat:", cat)
print("dog:", dog)
print("car:", car)

cat_dog_distance = np.linalg.norm(cat - dog)
cat_car_distance = np.linalg.norm(cat - car)

print("Distance between cat and dog:", cat_dog_distance)
print("Distance between cat and car:", cat_car_distance)