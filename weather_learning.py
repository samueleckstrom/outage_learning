import csv
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

def repair_time_prediction(output, min_repair_time, middle_repair_time):
  return ((output * middle_repair_time) + middle_repair_time) + min_repair_time

class net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fully_connected1 = nn.Linear(296, 148)
    self.activation1 = nn.Tanh()
    self.fully_connected2 = nn.Linear(148, 74)
    self.activation2 = nn.Tanh()
    self.fully_connected3 = nn.Linear(74, 37)
    self.activation3 = nn.Tanh()
    self.fully_connected4 = nn.Linear(37, 1)

  def forward(self, x):
    output = self.activation1(self.fully_connected1(x))
    output = self.activation2(self.fully_connected2(output))
    output = self.activation3(self.fully_connected3(output))
    output = self.fully_connected4(output)
    return output

def create_zip_code_one_hot(input_array):
  number_of_zip_codes = 241
  one_hot = torch.zeros(len(input_array), number_of_zip_codes)
  for i in range(len(input_array)):
    one_hot[i][input_array[i] - 1] += 1
  return one_hot

def create_problem_one_hot(input_array):
  number_of_problems = 10
  one_hot = torch.zeros(len(input_array), number_of_problems)
  for i in range(len(input_array)):
    one_hot[i][input_array[i]] += 1
  return one_hot

def create_crew_status_one_hot(input_array):
  number_of_crew_statuses = 6
  one_hot = torch.zeros(len(input_array), number_of_crew_statuses)
  for i in range(len(input_array)):
    one_hot[i][input_array[i]] += 1
  return one_hot

def create_cause_one_hot(input_array):
  number_of_causes = 7
  one_hot = torch.zeros(len(input_array), number_of_causes)
  for i in range(len(input_array)):
    one_hot[i][input_array[i]] += 1
  return one_hot

def create_day_of_week_one_hot(input_array):
  number_of_days = 7
  one_hot = torch.zeros(len(input_array), number_of_days)
  for i in range(len(input_array)):
    one_hot[i][input_array[i] - 1] += 1
  return one_hot

def create_condition_one_hot(input_array):
  number_of_days = 17
  one_hot = torch.zeros(len(input_array), number_of_days)
  for i in range(len(input_array)):
    one_hot[i][input_array[i]] += 1
  return one_hot

def training():
  rows = []
  f = open('./routing_learning.csv', 'r')
  reader = csv.reader(f, delimiter=',')
  for row in reader:
    rows.append(row)
  rows.pop(0)

  all_data = np.array(rows)
  results = all_data[:,28]
  inputs = np.concatenate((np.array(all_data[:,22:28]), np.array(all_data[:,29:])), axis=1)

  zip_code = create_zip_code_one_hot(inputs[:,0].astype(np.int64))
  number_of_outage_customers = normalize_zero_to_one(inputs[:,1].astype(np.float32))
  problem = create_problem_one_hot(inputs[:,2].astype(np.int64))
  crew_status = create_crew_status_one_hot(inputs[:,3].astype(np.int64))
  cause = create_cause_one_hot(inputs[:,4].astype(np.int64))
  day_of_week = create_day_of_week_one_hot(inputs[:,5].astype(np.int64))
  time_crawled_number_of_outages = normalize_zero_to_one(inputs[:,6].astype(np.float32))
  condition = create_condition_one_hot(inputs[:,7].astype(np.int64))
  average_celsius = normalize_zero_to_one(inputs[:,8].astype(np.float32))
  minimum_celsius = normalize_zero_to_one(inputs[:,9].astype(np.float32))
  maximum_celsius = normalize_zero_to_one(inputs[:,10].astype(np.float32))
  pressure = normalize_zero_to_one(inputs[:,11].astype(np.float32))
  humidity = normalize_zero_to_one(inputs[:,12].astype(np.float32))
  wind_speed = normalize_zero_to_one(inputs[:,13].astype(np.float32))

  (results, min_repair_time, middle_repair_time) = normalize_negative_one_to_positive_one(results.astype(np.float32))
  inputs = torch.cat((zip_code, number_of_outage_customers,
    problem, crew_status, cause, day_of_week, time_crawled_number_of_outages,
    condition, average_celsius, minimum_celsius, maximum_celsius, pressure,
    humidity, wind_speed), axis=1)

  number_of_samples = int(0.8 * len(results))
  validation_results = results[:number_of_samples,:]
  validation_inputs = inputs[:number_of_samples,:]
  test_results = results[number_of_samples:,:]
  test_inputs = inputs[number_of_samples:,:]

  model = net()

  learning_rate = 1e-4
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  loss_fn = nn.MSELoss()
  number_of_epochs = 50

  for epoch in range(number_of_epochs):
    for i in range(len(validation_results)):
      out = model(validation_inputs[i])
      loss = loss_fn(out, validation_results[i])
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

  torch.save(model.state_dict(), './weather_model.pt')
  model = net()
  model.load_state_dict(torch.load('./weather_model.pt'))
  model.eval()

  with torch.no_grad():
    for i in range(len(test_results)):
      out = model(test_inputs[i])
      print("Repair time prediction: " + str(repair_time_prediction(out, min_repair_time, middle_repair_time)[0])
        + ", Actual time: " + str(repair_time_prediction(test_results[i], min_repair_time, middle_repair_time)[0]))

def prediction(rows):
  all_data = []
  for row in rows:  # necessary to change this if training to simply all_data = np.array(rows)
    row_values = []
    for value in row.values():
      row_values.append(value)
    all_data.append(row_values)
  all_data = np.array(all_data)

  all_data = np.array(rows)
  results = all_data[:,28]
  inputs = np.concatenate((np.array(all_data[:,22:28]), np.array(all_data[:,29:])), axis=1)

  zip_code = create_zip_code_one_hot(inputs[:,0].astype(np.int64))
  number_of_outage_customers = normalize_zero_to_one(inputs[:,1].astype(np.float32))
  problem = create_problem_one_hot(inputs[:,2].astype(np.int64))
  crew_status = create_crew_status_one_hot(inputs[:,3].astype(np.int64))
  cause = create_cause_one_hot(inputs[:,4].astype(np.int64))
  day_of_week = create_day_of_week_one_hot(inputs[:,5].astype(np.int64))
  time_crawled_number_of_outages = normalize_zero_to_one(inputs[:,6].astype(np.float32))
  condition = create_condition_one_hot(inputs[:,7].astype(np.int64))
  average_celsius = normalize_zero_to_one(inputs[:,8].astype(np.float32))
  minimum_celsius = normalize_zero_to_one(inputs[:,9].astype(np.float32))
  maximum_celsius = normalize_zero_to_one(inputs[:,10].astype(np.float32))
  pressure = normalize_zero_to_one(inputs[:,11].astype(np.float32))
  humidity = normalize_zero_to_one(inputs[:,12].astype(np.float32))
  wind_speed = normalize_zero_to_one(inputs[:,13].astype(np.float32))

  (results, min_repair_time, middle_repair_time) = normalize_negative_one_to_positive_one(results.astype(np.float32))
  inputs = torch.cat((zip_code, number_of_outage_customers,
    problem, crew_status, cause, day_of_week, time_crawled_number_of_outages,
    condition, average_celsius, minimum_celsius, maximum_celsius, pressure,
    humidity, wind_speed), axis=1)
  
  model = net()
  model.load_state_dict(torch.load('./weather_model.pt'))
  model.eval()

  with torch.no_grad():
    for i in range(len(test_results)):
      out = model(test_inputs[i])
      print("Repair time prediction: " + str(repair_time_prediction(out, min_repair_time, middle_repair_time)[0])
        + ", Actual time: " + str(repair_time_prediction(test_results[i], min_repair_time, middle_repair_time)[0]))

def actual():
  rows = []
  f = open('./routing_learning.csv', 'r')
  reader = csv.reader(f, delimiter=',')
  for row in reader:
    rows.append(row)
  rows.pop(0)

  all_data = np.array(rows)
  results = all_data[:,28]
  inputs = np.concatenate((np.array(all_data[:,22:28]), np.array(all_data[:,29:])), axis=1)

  zip_code = create_zip_code_one_hot(inputs[:,0].astype(np.int64))
  number_of_outage_customers = normalize_zero_to_one(inputs[:,1].astype(np.float32))
  problem = create_problem_one_hot(inputs[:,2].astype(np.int64))
  crew_status = create_crew_status_one_hot(inputs[:,3].astype(np.int64))
  cause = create_cause_one_hot(inputs[:,4].astype(np.int64))
  day_of_week = create_day_of_week_one_hot(inputs[:,5].astype(np.int64))
  time_crawled_number_of_outages = normalize_zero_to_one(inputs[:,6].astype(np.float32))
  condition = create_condition_one_hot(inputs[:,7].astype(np.int64))
  average_celsius = normalize_zero_to_one(inputs[:,8].astype(np.float32))
  minimum_celsius = normalize_zero_to_one(inputs[:,9].astype(np.float32))
  maximum_celsius = normalize_zero_to_one(inputs[:,10].astype(np.float32))
  pressure = normalize_zero_to_one(inputs[:,11].astype(np.float32))
  humidity = normalize_zero_to_one(inputs[:,12].astype(np.float32))
  wind_speed = normalize_zero_to_one(inputs[:,13].astype(np.float32))

  (results, min_repair_time, middle_repair_time) = normalize_negative_one_to_positive_one(results.astype(np.float32))
  inputs = torch.cat((zip_code, number_of_outage_customers,
    problem, crew_status, cause, day_of_week, time_crawled_number_of_outages,
    condition, average_celsius, minimum_celsius, maximum_celsius, pressure,
    humidity, wind_speed), axis=1)
  
  model = net()
  model.load_state_dict(torch.load('./weather_model.pt'))
  model.eval()

  with torch.no_grad():
    for i in range(len(test_results)):
      out = model(test_inputs[i])
      print("Repair time prediction: " + str(repair_time_prediction(out, min_repair_time, middle_repair_time)[0])
        + ", Actual time: " + str(repair_time_prediction(test_results[i], min_repair_time, middle_repair_time)[0]))

def main(type_of_process, outages=None):
  if type_of_process == "training":
    training()
  elif type_of_process == "prediction":
    prediction()
  elif type_of_process == "actual":
    actual()

def prediction_time(type_of_process, outages):
  main(type_of_process, outages)