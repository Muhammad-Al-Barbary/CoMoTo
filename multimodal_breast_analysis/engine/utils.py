def prepare_batch(batch, device):
  image = [batch[i]["image"].to(device) for i in range(len(batch))]
  targets = [{"boxes":batch[i]["boxes"].to(device), "labels":batch[i]["labels"].to(device)} for i in range(len(batch))]
  return image, targets