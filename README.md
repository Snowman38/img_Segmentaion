# Best Result
- Architecture: DeepLab_v3
- Data: Data augmentation with flip and random rotate
- Training parameters: Adam, lr=1e-3, batch_size=32
- Result:
    - val_dice = 0.85281
    - test_dice = 0.86096 


# Neural Computation Task

## About
- This is a basic frame. Not for functionality/submission.
    - Use it to verify whether your code(data augmentation/new models/new loss) is working or not.
    - The baseline model name is UNet

## Run The Code
1. generate unet_config.json
   ```
   python utils/configs.py
   ```
2. edit unet_config.json with your preference, for example
   - data paths
   - epochs and batch_size
3. train a model
   ```
   python train.py --config ./unet_config.json
   python train.py --config ./unet_config.json --use_val
   ```
4. eval the Dice Score on validation set
   ```
   python eval.py --config ./unet_config.json
   ```
5. predict the result
   - plot them
       ```
       python predict.py --result plot --max 1000 --config ./unet_config.json
       ```
   - save them as png
       ```
       python predict.py --result png --max 1000 --config ./unet_config.json
       ```
     By default, these two are the same
       ```
       python predict.py --config ./unet_config.json
       python predict.py --result plot --max 10 --config ./unet_config.json 
       ```
6. generate submission file
   ```
   python generate_csv.py --config ./unet_config.json
   ```
7. submit to kaggle https://www.kaggle.com/t/8ba9f4955d7c4d758a74cb9e52905873

## TODO
### Data
- [ ] Data augmentation
- [ ] Extra preprocessed data
- [ ] Pretty print mask with 4 kinds of colors, even with border
### Models
- [ ] FCN
- [ ] DeepLab
- [ ] SegNet
- [ ] …
### Optimiser 
- [ ] lr scheduler
- [ ] GradScaler
### Loss function (Pytorch Implementation)
- [ ] the soft Dice loss
- [ ] the Focal loss
- [ ] The boundary loss
- [ ] class balanced weights(TBC)
### Epochs Loop
- [ ] Val error is lower than train error. Bug or Data Leakage?
- [ ] Cross validation & Grid search. Sth like
  ```
  def pick_best_hyper_param():
    for hyper_param in lst
      for i = 1:folds_num
        shuffle training set
        split training set to (train_split,val_split)
        use train_split to train a model
        calculate error on the val_split
      end_for
      if avg_error < best_error
        update best_error and record hyper_param
      end_if
    end_for
  return best_hyper_param
  ```
- [ ] save checkpoints in every epochs/10 epochs, sth like
    ```
    saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)
    ```
    see https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/saver.py


- [ ] Record best models during training/early stopping
### Object-oriented programming
- [ ] build class for loss, training ...

## Rerferences
- https://github.com/milesial/Pytorch-UNet

## Final Report 
https://docs.google.com/document/d/1W6M6wwlEqXxAuD3OVt9RYNc-VghO7LQjXEwF3hehVp8/edit?usp=sharing

