import ast

def prepare_batch(batch, device):
  image = [batch[i]["image"].to(device) for i in range(len(batch))]
  targets = [{"boxes":batch[i]["boxes"].to(device), "labels":batch[i]["labels"].to(device)} for i in range(len(batch))]
  return image, targets

def average_dicts(list_of_dicts):
  num_dicts = len(list_of_dicts)
  avg_dict = {}
  for d in list_of_dicts:
      for key, value in d.items():
          avg_dict[key] = avg_dict.get(key, 0) + value / num_dicts
  return avg_dict
  

def log_transforms(file_path, dataset_name):
    train_list, test_list = ["TRAIN:\n"],["TEST:\n"]
    with open(file_path, 'r') as file:
        lines = file.readlines()
    #get train and test sections from the file
    for line_idx in range(len(lines)):
      if "train_transforms" in lines[line_idx]:
        train_start = line_idx
      if "return" in lines[line_idx]:
        train_end = line_idx
        break
    for line_idx in range(train_end+1,len(lines)):
      if "test_transforms" in lines[line_idx]:
        test_start = line_idx
      if "return" in lines[line_idx]:
        test_end = line_idx
        break
    #print train transforms
    found_dataset = False
    for line_idx in range(train_start, train_end):
        if found_dataset and (":" in lines[line_idx] or "}" in lines[line_idx]):
            break
        if dataset_name in lines[line_idx]:
            found_dataset = True
        if found_dataset and not lines[line_idx].strip().startswith('#'):
            train_list.append(lines[line_idx])
    #print test transforms
    found_dataset = False
    for line_idx in range(test_start, test_end):
        if found_dataset and ":" in lines[line_idx]:
            break
        if dataset_name in lines[line_idx]:
            found_dataset = True
        if found_dataset and not lines[line_idx].strip().startswith('#'):
            test_list.append(lines[line_idx])
    train_list = ''.join(train_list)
    test_list = ''.join(test_list)
    return (train_list, test_list)