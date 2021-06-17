import csv
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sqrt, pow, exp
from random import randint, shuffle
from matplotlib import pyplot as plt
import datetime

weather_rows = []
f = open('./outage_weather_learning.csv', 'r')
reader = csv.reader(f, delimiter=',')
for row in reader:
  weather_rows.append(row)
weather_rows.pop(0)

dictionary = { }
for row in weather_rows:
  dictionary[str(row[0])] = [row]

outage_rows = []
f = open('./outage_learning.csv', 'r')
reader = csv.reader(f, delimiter=',')
for row in reader:
  outage_rows.append(row)
outage_rows.pop(0)

for row in outage_rows:
  dictionary[str(row[6])].append(row)

def normalize_zero_to_one(input_array):
  input_array_max = input_array.max()
  input_array_min = input_array.min()
  input_array -= input_array_min
  input_array /= (input_array_max - input_array_min)
  return torch.tensor(input_array).unsqueeze(1)

def normalize_negative_one_to_positive_one(input_array):
  input_array_max = input_array.max()
  input_array_min = input_array.min()
  middle = (input_array_max - input_array_min) / 2.0
  input_array = ((input_array - input_array_min) - middle) / middle
  return (torch.tensor(input_array).unsqueeze(1), input_array_min, middle)

def create_one_hot_zero_start(input_array):
  input_array_max = input_array.max() + 1
  one_hot = torch.zeros(len(input_array), input_array_max)
  for i in range(len(input_array)):
    one_hot[i][input_array[i]] += 1
  return one_hot

def create_one_hot_one_start(input_array):
  input_array_max = input_array.max()
  one_hot = torch.zeros(len(input_array), input_array_max)
  for i in range(len(input_array)):
    one_hot[i][input_array[i] - 1] += 1
  return one_hot

def euclidean_distance(x1, y1, x2, y2):
  return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

image_number_of_pixels = 40
number_of_input_pixels = image_number_of_pixels * image_number_of_pixels
minimum_latitude = 40.508274000000000
maximum_latitude = 40.900819000000000
latitude_difference = (maximum_latitude - minimum_latitude) / image_number_of_pixels
minimum_longitude = -74.243870000000000
maximum_longitude = -73.709560000000000
longitude_difference = (maximum_longitude - minimum_longitude) / image_number_of_pixels
normalizing_difference = sqrt(pow(latitude_difference, 2) + pow(longitude_difference, 2))

def euclidean_distance(x1, y1, x2, y2):
  return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)) / normalizing_difference

def create_period_image(period):
  image = np.zeros((image_number_of_pixels, image_number_of_pixels)).astype(np.float32)
  for row in period:
    for j in range(image_number_of_pixels):
      for k in range(image_number_of_pixels):
        current_latitude = minimum_latitude + float(latitude_difference * j)
        current_longitude = minimum_longitude + float(longitude_difference * k)
        distance = euclidean_distance(current_latitude, current_longitude, float(row[0]), float(row[1]))
        image[j][k] += (1.0 / (1.0 + exp(distance)))
  return image

data = []
images = []
i = 0
for row in dictionary.values():
  data.append(row[0])
  images.append(create_period_image(np.array(row[1:])))
  #plt.imshow(images[i])
  #plt.show()
  print(i)
  i += 1
  #if i == 30:
  #  break

images = np.array(images)
images_max = np.amax(images)

for image in images:
  for j in range(image_number_of_pixels):
    for k in range(image_number_of_pixels):
      if image[j][k] != 0.0:
        image[j][k] /= images_max

data = np.array(data)
month_of_year = create_one_hot_one_start(data[:,1].astype(np.int32))
week = create_one_hot_one_start(data[:,2].astype(np.int32))
day_of_week = create_one_hot_one_start(data[:,3].astype(np.int32))
hour_of_day = create_one_hot_one_start(data[:,4].astype(np.int32))
condition = create_one_hot_zero_start(data[:,5].astype(np.int32))
average_celsius = normalize_zero_to_one(data[:,6].astype(np.float32))
minimum_celsius = normalize_zero_to_one(data[:,7].astype(np.float32))
maximum_celsius = normalize_zero_to_one(data[:,8].astype(np.float32))
pressure = normalize_zero_to_one(data[:,9].astype(np.float32))
humidity = normalize_zero_to_one(data[:,10].astype(np.float32))
wind_speed = normalize_zero_to_one(data[:,11].astype(np.float32))
inputs = torch.cat((month_of_year, week, day_of_week, hour_of_day,
  condition, average_celsius, minimum_celsius, maximum_celsius, pressure,
  humidity, wind_speed), axis=1)

deep_net = []
combo_net = []
weather_net = []
results = []
for i in range(len(images) - 3):
  deep_net.append(torch.tensor([images[i], images[i + 1], images[i + 2]]).unsqueeze(0).unsqueeze(0))
  
  first_image_flattened = torch.tensor(images[i].flatten())
  second_image_flattened = torch.tensor(images[i + 1].flatten())
  third_image_flattened = torch.tensor(images[i + 2].flatten())
  fourth_image_flattened = torch.tensor(images[i + 3].flatten())
  first_weather_flattened = inputs[i]
  second_weather_flattened = inputs[i + 1]
  third_weather_flattened = inputs[i + 2]
  fourth_weather_flattened = inputs[i + 3]
  
  combo_net.append(torch.cat((first_image_flattened, second_image_flattened,
    third_image_flattened, first_weather_flattened, second_weather_flattened,
    third_weather_flattened, fourth_weather_flattened), axis=0))

  weather_net.append(torch.cat((first_weather_flattened, second_weather_flattened,
    third_weather_flattened, fourth_weather_flattened), axis=0))

  results.append(fourth_image_flattened)
deep_net = torch.stack(deep_net, axis=0)
print(deep_net.shape)
combo_net = torch.stack(combo_net, axis=0)
weather_net = torch.stack(weather_net, axis=0)
results = torch.stack(results, axis=0)

class DeepNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.convolution1 = nn.Conv3d(1, 20, kernel_size=3, padding=1)
    self.activation1 = nn.Tanh()
    self.max_pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 3, 3))
    self.convolution2 = nn.Conv3d(20, 10, kernel_size=3, padding=1)
    self.activation2 = nn.Tanh()
    self.max_pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 3, 3))
    self.fully_connected1 = nn.Linear(480, 3000)
    self.activation3 = nn.Tanh()
    self.fully_connected2 = nn.Linear(3000, number_of_input_pixels)

  def forward(self, x):
    output = self.max_pool1(self.activation1(self.convolution1(x)))
    output = self.max_pool2(self.activation2(self.convolution2(output)))
    output = output.view(-1, 480)
    output = self.activation3(self.fully_connected1(output))
    output = self.fully_connected2(output)
    return output

class ComboNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fully_connected1 = nn.Linear(3 * number_of_input_pixels + 456, 7500)
    self.activation1 = nn.Tanh()
    self.fully_connected2 = nn.Linear(7500, number_of_input_pixels)

  def forward(self, x):
    output = self.activation1(self.fully_connected1(x))
    output = self.fully_connected2(output)
    return output

class WeatherNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fully_connected1 = nn.Linear(456, 2000)
    self.activation1 = nn.Tanh()
    self.fully_connected2 = nn.Linear(2000, number_of_input_pixels)

  def forward(self, x):
    output = self.activation1(self.fully_connected1(x))
    output = self.fully_connected2(output)
    return output
    
class FullNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fully_connected1 = nn.Linear(2 * number_of_input_pixels, 2000)
    self.activation1 = nn.Tanh()
    self.fully_connected2 = nn.Linear(2000, number_of_input_pixels)

  def forward(self, x):
    output = self.activation1(self.fully_connected1(x))
    output = self.fully_connected2(output)
    return output

number_of_samples = int(0.8 * len(results))

indices = [i for i in range(len(results))]
print(indices)
training_indices = [indices.pop(randint(0, len(indices) - 1)) for i in range(number_of_samples)]
print(training_indices)
print(indices)

learning_rate = 1e-1

deep_net_model = DeepNet()
combo_net_model = ComboNet()
weather_net_model = WeatherNet()
full_net_model = FullNet()

deep_net_optimizer = optim.SGD(deep_net_model.parameters(), lr=learning_rate)
combo_net_optimizer = optim.SGD(combo_net_model.parameters(), lr=learning_rate)
weather_net_optimizer = optim.SGD(weather_net_model.parameters(), lr=learning_rate)
full_net_optimizer = optim.SGD(full_net_model.parameters(), lr=learning_rate)

deep_net_loss_fn = nn.MSELoss()
combo_net_loss_fn = nn.MSELoss()
weather_net_loss_fn = nn.MSELoss()
full_net_loss_fn = nn.MSELoss()

number_of_deep_net_epochs = 100
number_of_combo_net_epochs = 0 #100
number_of_weather_net_epochs = 200 #100
number_of_full_net_epochs = 200 #100

for epoch in range(number_of_deep_net_epochs):
  shuffle(training_indices)
  for i in range(len(training_indices)):
    deep_net_out = deep_net_model(deep_net[i]).squeeze()

    loss = deep_net_loss_fn(deep_net_out, results[i])
    deep_net_optimizer.zero_grad()
    loss.backward()
    deep_net_optimizer.step()
  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

with torch.no_grad():
  for i in range(len(indices)):
    output = deep_net_model(deep_net[i]).squeeze()
    output = np.reshape(output, (-1, image_number_of_pixels))
    image = np.reshape(results[i], (-1, image_number_of_pixels))
    plt.figure()
    f, arr = plt.subplots(2, 1)
    arr[0].imshow(output)
    arr[1].imshow(image)
    plt.show()

for epoch in range(number_of_combo_net_epochs):
  shuffle(training_indices)
  for i in range(len(training_indices)):
    combo_net_out = combo_net_model(combo_net[i])

    loss = combo_net_loss_fn(combo_net_out, results[i])
    combo_net_optimizer.zero_grad()
    loss.backward()
    combo_net_optimizer.step()
  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

for epoch in range(number_of_weather_net_epochs):
  shuffle(training_indices)
  for i in range(len(training_indices)):
    weather_net_out = weather_net_model(weather_net[i])

    loss = weather_net_loss_fn(weather_net_out, results[i])
    weather_net_optimizer.zero_grad()
    loss.backward()
    weather_net_optimizer.step()
  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

for epoch in range(number_of_full_net_epochs):
  shuffle(training_indices)
  for i in range(len(training_indices)):
    deep_net_output = deep_net_model(deep_net[i]).squeeze()
    #combo_net_output = combo_net_model(combo_net_training[i])
    weather_net_output = weather_net_model(weather_net[i])
    full_net_input = torch.cat((deep_net_output, weather_net_output), axis=0)

    full_net_out = full_net_model(full_net_input)

    loss = full_net_loss_fn(full_net_out, results[i])
    full_net_optimizer.zero_grad()
    loss.backward()
    full_net_optimizer.step()
  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

with torch.no_grad():
  for i in range(len(indices)):
    deep_net_output = deep_net_model(deep_net[i]).squeeze()
    weather_net_output = weather_net_model(weather_net[i])
    total_input = torch.cat((deep_net_output, weather_net_output), axis=0)
    output = full_net_model(total_input)
    output = np.reshape(output, (-1, image_number_of_pixels))
    image = np.reshape(results[i], (-1, image_number_of_pixels))
    plt.figure()
    f, arr = plt.subplots(2, 1)
    arr[0].imshow(output)
    arr[1].imshow(image)
    plt.show()