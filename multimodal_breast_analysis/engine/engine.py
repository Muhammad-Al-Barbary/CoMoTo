# import nibabel as nib
from monai.transforms import ToTensor,Resize
from monai.metrics import DiceMetric
# from monai.networks.utils import one_hot
# from liver_imaging_analysis.models.liver_segmentation import segment_liver_3d
import os
# import gc
import torch
from shutil import rmtree
# import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import gc
import SimpleITK
import natsort
from monai.data import Dataset, DataLoader as MonaiLoader
import cv2
from multimodal_breast_analysis.models.unet import UNet
from monai.losses import DiceLoss
import torch
from monai.networks import one_hot
import numpy as np
from monai.inferers import sliding_window_inference
from multimodal_breast_analysis.data.dataloader import ct_trainloader, mri_trainloader, ct_testloader_3d, mri_testloader_3d, mri_test_transforms, test_transforms as ct_test_transforms
from monai.metrics import DiceMetric
from random import randint


device = torch.device(1 if torch.cuda.is_available() else "cpu")
ct_cp = 'ct_cp.pt'
mri_cp = 'mri_cp.pt'
mri_net = UNet(spatial_dims=2, in_channels=1, out_channels=2, channels=(64, 128, 256, 512), strides=(2, 2, 2), num_res_units=4, dropout = 0.5).to(device)
ct_net = UNet(spatial_dims=2, in_channels=1, out_channels=2, channels=(64, 128, 256, 512), strides=(2, 2, 2), num_res_units=4, dropout = 0.5).to(device)
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
ct_net.bottleneck.register_forward_hook(get_activation('ct'))
mri_net.bottleneck.register_forward_hook(get_activation('mri'))

for ct_batch, mri_batch in zip(ct_trainloader, mri_trainloader):
    print('image:', ct_batch['image'].shape, mri_batch['image'].shape)
    ct_pred = ct_net(ct_batch['image'].to(device))
    mri_pred = mri_net(mri_batch['image'].to(device))
    print('pred:', ct_pred.shape, mri_pred.shape)
    ct_feat, mri_feat = activation['ct'], activation['mri']
    print('feat:', ct_feat.shape, mri_feat.shape)
    break
flat = torch.nn.Flatten(start_dim=2).to(device)
project = torch.nn.Linear(torch.prod(torch.tensor(mri_feat.shape[2:])), torch.prod(torch.tensor(ct_feat.shape[2:]))).to(device)
ct_iter = iter(ct_trainloader)

loss_seg = DiceLoss(include_background=True, to_onehot_y=True, softmax=True, batch = True).to(device)
optimizer_mri = torch.optim.Adam(list(mri_net.parameters()) + list(project.parameters()), lr=0.001)
optimizer_ct = torch.optim.Adam(ct_net.parameters(), lr=0.001)



def loss_fn_kd(outputs, teacher_outputs, T, alpha):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(outputs/T, dim=1),
                             torch.nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return KD_loss


def predict_2dto3d(volume_path, network, transforms, batch_size = 8):
    # Read volume
    img_volume = SimpleITK.ReadImage(volume_path)
    img_volume_array = SimpleITK.GetArrayFromImage(img_volume)
    number_of_slices = img_volume_array.shape[0]
    # Create temporary folder to store 2d png files
    if os.path.exists("temp") == False:
      os.mkdir("temp")
    # Write volume slices as 2d png files
    for slice_number in range(number_of_slices):
        volume_slice = img_volume_array[slice_number, :, :]
        # Delete extension from filename
        volume_file_name = os.path.splitext(volume_path)[0].split("/")[-1]
        volume_png_path = os.path.join(
                                "temp",
                                volume_file_name + "_" + str(slice_number)
                                ) + ".png"
        cv2.imwrite(volume_png_path, volume_slice)
    # Predict slices individually then reconstruct 3D prediction
    network.eval()
    with torch.no_grad():
        volume_names = natsort.natsorted(os.listdir("temp"))
        volume_paths = [os.path.join("temp", file_name)
                        for file_name in volume_names]
        predict_files = [{'image': image_name}
                          for image_name in volume_paths]
        predict_set = Dataset(
            data=predict_files,
            transform=transforms
            )
        predict_loader = MonaiLoader(
            predict_set,
            batch_size = batch_size,
            num_workers = 0,
            pin_memory = False,
        )
        prediction_list = []
        for batch in predict_loader:
            batch['image'] = batch['image'].to(device)
            batch['pred'] = network(batch['image'])
            prediction_list.append(batch['pred'])
        prediction_list = torch.cat(prediction_list, dim=0)
    batch = {'pred' : prediction_list}
    # Transform shape from (batch,channel,length,width)
    # to (1,channel,length,width,batch)
    batch['pred'] = batch['pred'].permute(1,2,3,0).unsqueeze(dim = 0)
    # Apply post processing transforms
    # batch = self.post_process(batch)
    batch['pred'] = torch.argmax(batch['pred'], dim = 1, keepdim = True)
    # Delete temporary folder
    rmtree("temp")
    return batch['pred']




def test_3d_using_2d_network(dataset_path, network, transforms):
  volume_names = os.listdir(os.path.join(dataset_path, "volume"))
  mask_names = os.listdir(os.path.join(dataset_path, "mask"))
  volume_paths = [
      os.path.join(dataset_path, "volume/", file_name)
      for file_name in volume_names
  ]
  mask_paths = [
      os.path.join(dataset_path, "mask/", file_name)
      for file_name in mask_names
  ]
  volume_paths.sort()  # to be sure that the paths are sorted so every volume corresponds to the correct mask
  mask_paths.sort()
  average_liver_dice=0
  failed_counter=0
  for idx, (volume_path, mask_path) in enumerate(zip(volume_paths,mask_paths)):
      try:
          gc.collect()
          torch.cuda.empty_cache()
          # volume_idx=volume_path.split('-')[1].split('.')[0]
          volume=nib.load(volume_path).get_fdata() #read volume
          segmentation=ToTensor()((nib.load(mask_path).get_fdata()).reshape(1,1,volume.shape[0],volume.shape[1],volume.shape[2])) #read mask
          segmentation= (segmentation>0.5).float()
          prediction3d=predict_2dto3d(volume_path, network, transforms, batch_size=8)
          segmentation = (Resize(spatial_size = prediction3d.shape[2:], mode = 'nearest')(segmentation[0])).unsqueeze(dim = 0)
          dice=DiceMetric(reduction=None)( prediction3d.cpu(), segmentation )
        #   print(f"volume {idx} dice: {dice[0].item():.5f}");
          average_liver_dice+=dice[0].item()
      except RuntimeError:
          failed_counter+=1
          print("failed to predict volume")
          gc.collect()
          torch.cuda.empty_cache()
          rmtree("temp")
      # break
  average_liver_dice/= (len(volume_paths)-failed_counter)
  return average_liver_dice



def test3d(dataloader, net, modality):
    num_batches = len(dataloader)
    dice = DiceMetric(include_background = False)
    net.eval()
    with torch.no_grad():
        for batch_num, (batch) in enumerate(dataloader):
            pred = sliding_window_inference(batch['image'].to(device), (128,128,8), 2, net, 0.25)
            pred = torch.argmax(pred, dim = 1, keepdim = True)
            dice(pred.int(), batch['mask'].int().to(device))
        test_metric = dice.aggregate()
        dice.reset()
    print(modality, "Test Metric:", test_metric.item())
    return test_metric


def save_checkpoint(path, network, optimizer):
    checkpoint = {
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, path)

def load_checkpoint(path, network, optimizer):
    checkpoint = torch.load(path)
    network.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return network, optimizer

# print('Initial Loss:', test_3d_using_2d_network('lits/Test', ct_net, ct_test_transforms))




# print('WARMUP START')
# warmup_epochs = 10
# best_metric = 0
# for epoch in range(warmup_epochs):
#     print(f"\nEpoch {epoch+1}/{warmup_epochs}\n-------------------------------")
#     ct_training_loss = 0
#     ct_net.train()
#     for iteration, ct_batch in enumerate(ct_trainloader):
#         # print(f"     iteration {iteration+1}/{len(ct_trainloader)}")
#         ct_pred = ct_net(ct_batch['image'].to(device))
#         if not(iteration % 1000):
#           import matplotlib.pyplot as plt
#           rand_idx = randint(0, ct_batch['image'].shape[0]-1)
#           plt.subplot(1,3,1)
#           plt.imshow(ct_batch['image'][rand_idx,0,:,:], cmap = 'gray')
#           plt.subplot(1,3,2)
#           plt.imshow(ct_batch['mask'][rand_idx,0,:,:], cmap = 'gray')
#           plt.subplot(1,3,3)
#           plt.imshow(torch.argmax(ct_pred, dim = 1, keepdim = True)[rand_idx,0,:,:].cpu().detach(), cmap = 'gray')
#           plt.show()


#         ct_loss = loss_seg(ct_pred, ct_batch['mask'].to(device))
#         optimizer_ct.zero_grad()
#         ct_loss.backward()
#         optimizer_ct.step()
#         ct_training_loss += ct_loss.item()
#     ct_training_loss = ct_training_loss / len(ct_trainloader)
#     print("CT Warmup Loss:", ct_training_loss)
#     # current_metric = test(ct_testloader_3d, ct_net, 'CT')
#     current_metric = test_3d_using_2d_network('lits/Test', ct_net, ct_test_transforms)
#     print("CT Metric:", current_metric)
#     if current_metric >= best_metric:
#         best_metric = current_metric
#         print('saving CT checkpoint')
#         save_checkpoint(ct_cp, ct_net, optimizer_ct)
# print('WARMUP END')

ct_net, optimizer_ct =  load_checkpoint(ct_cp, ct_net, optimizer_ct)
# current_metric = test_3d_using_2d_network('lits/Test', ct_net, ct_test_transforms)
# print("CT Metric:", current_metric)



epochs = 50
distill_epoch = 1
mri_loss_distill = torch.tensor(0)
best_metric = 0

seg_loss_progress = []
distill_loss_progress = []
total_loss_progress = []
metric_progress = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
    ct_training_loss = 0
    mri_training_loss = 0
    mri_training_seg_loss = 0
    mri_training_distill_loss = 0
    mri_net.train()
    ct_net.eval()
    for iteration, mri_batch in enumerate( mri_trainloader):
        try:
            ct_batch = next(ct_iter)
        except StopIteration:
          ct_iter = iter(ct_trainloader)
          ct_batch = next(ct_iter)
        if (len(ct_batch['image']) != len(mri_batch['image'])):
          continue
        # print(f"     iteration {iteration+1}/{len(ct_trainloader)}")
        ct_pred = ct_net(ct_batch['image'].to(device))
        mri_pred = mri_net(mri_batch['image'].to(device))
        ct_feat, mri_feat = activation['ct'], activation['mri']
        ct_feat = flat(ct_feat).mean(dim = 0)#.mean(dim = 0).unsqueeze(0)
        mri_feat = project(flat(mri_feat).mean(dim = 0))#.mean(dim = 0).unsqueeze(0)
        # print('ct', ct_pred.shape, ct_feat.shape)
        # print('mri', mri_pred.shape, mri_feat.shape)
        if not(iteration % 500):
          import matplotlib.pyplot as plt
          rand_idx = randint(0, ct_batch['image'].shape[0]-1)
          plt.subplot(2,3,1)
          plt.imshow(ct_batch['image'][rand_idx,0,:,:], cmap = 'gray')
          plt.subplot(2,3,4)
          plt.imshow(mri_batch['image'][rand_idx,0,:,:], cmap = 'gray')
          plt.subplot(2,3,2)
          plt.imshow(ct_batch['mask'][rand_idx,0,:,:], cmap = 'gray')
          plt.subplot(2,3,5)
          plt.imshow(mri_batch['mask'][rand_idx,0,:,:], cmap = 'gray')
          plt.subplot(2,3,3)
          plt.imshow(torch.argmax(ct_pred, dim = 1, keepdim = True)[rand_idx,0,:,:].cpu().detach(), cmap = 'gray')
          plt.subplot(2,3,6)
          plt.imshow(torch.argmax(mri_pred, dim = 1, keepdim = True)[rand_idx,0,:,:].cpu().detach(), cmap = 'gray')
          plt.show()
        # ct_loss = loss_seg(ct_pred, ct_batch['mask'].to(device))
        mri_loss_seg = loss_seg(mri_pred, mri_batch['mask'].to(device))
        # mri_loss_seg = torch.tensor(0)
        if epoch >= distill_epoch: #if distill
            # mri_feat = torch.nn.functional.softmax(mri_feat, dim = 1)
            # ct_feat = torch.nn.functional.softmax(ct_feat, dim = 1) #keep
            mri_loss_distill = loss_fn_kd(mri_feat, ct_feat, T=1, alpha=2)
            mri_loss = mri_loss_seg + mri_loss_distill
        else:
          mri_loss = mri_loss_seg
        optimizer_mri.zero_grad()
        # optimizer_ct.zero_grad()
        # ct_loss.backward()
        mri_loss.backward()
        # optimizer_ct.step()
        optimizer_mri.step()
        # ct_training_loss += ct_loss.item()
        mri_training_loss += mri_loss.item()
        mri_training_seg_loss += mri_loss_seg.item()
        mri_training_distill_loss += mri_loss_distill.item()
        # print("     CT Loss:", ct_loss.item())
        print("     MRI Loss:", mri_loss.item(), "Seg:", mri_loss_seg.item(), "Distill:", mri_loss_distill.item())
        seg_loss_progress.append(mri_loss_seg.item())
        distill_loss_progress.append(mri_loss_distill.item())
        total_loss_progress.append(mri_loss.item())
    # ct_training_loss = ct_training_loss / len(mri_trainloader)
    mri_training_loss = mri_training_loss / len(mri_trainloader)
    mri_training_seg_loss = mri_training_seg_loss / len(mri_trainloader)
    mri_training_distill_loss = mri_training_distill_loss / len(mri_trainloader)
    # print("CT Loss:", ct_training_loss)
    print("MRI Loss:", mri_training_loss, "Seg:", mri_training_seg_loss, "Distill:", mri_training_distill_loss)
    # print("Training:")
    # test(ct_trainloader, mri_trainloader)
    # print("Testing:")
    current_metric =  test_3d_using_2d_network('duke/Test', mri_net, mri_test_transforms)
    metric_progress.append(current_metric)
    print("MRI Metric:", current_metric)
    if current_metric >= best_metric:
        best_metric = current_metric
        print('saving MRI checkpoint')
        save_checkpoint(mri_cp, mri_net, optimizer_mri)


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.title('dice loss per iteration')
plt.plot(seg_loss_progress)

plt.subplot(2,2,2)
plt.title('kl-divergence per iteration')
plt.plot(distill_loss_progress)

plt.subplot(2,2,3)
plt.title('total loss per iteration')
plt.plot(total_loss_progress)

plt.subplot(2,2,4)
plt.title('dice score per epoch')
plt.plot(metric_progress)

plt.show()