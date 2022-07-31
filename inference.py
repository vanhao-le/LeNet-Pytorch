from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import KMNIST
from chip import config
import imutils
import torch
import cv2
import numpy as np


def main():
    tranform_lst = transforms.Compose([
        transforms.ToTensor()
    ])
    # load the KMNIST dataset
    print("[INFO] loading the KMNIST dataset...")
    test_ds = KMNIST(root="data", train=False, download=True, transform=tranform_lst)
    # get only 1 by 1 image for testing
    test_loader = DataLoader(test_ds, batch_size = 1)

    # load the model and set it to evaluation mode
    
    model = torch.load(config.MODEL_PATH)
    model.to(config.DEVICE)
    model.eval()

    # switch off autograd
    with torch.no_grad():
        # loop over the test set
        for (img, lb) in test_loader:
            # grab the original image and ground truth label
            orig_img= img.numpy().squeeze(axis=(0, 1))
            gt_label = test_loader.dataset.classes[lb.numpy()[0]]
            # send the input to the device and make predictions on it
            img = img.to(config.DEVICE)
            pred = model(img)
            # find the class label index with the maximum probability
            idx = pred.argmax(axis=1).cpu().numpy()[0]
            pred_label = test_loader.dataset.classes[idx]

            
            # convert the image from grayscale to RGB 
            orig_img = np.dstack([orig_img] * 3)
            
            orig_img = imutils.resize(orig_img, width=128)
            # draw the predicted class label on it
            color = (0, 255, 0) if gt_label == pred_label else (0, 0, 255)
            cv2.putText(orig_img, gt_label, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
            # display the result in terminal and show the input image
            print("[INFO] ground truth label: {}, predicted label: {}".format(gt_label, pred_label))
            cv2.imshow("image", orig_img)
            key = cv2.waitKey(0)
            # 27 is the esc key, 113 is the letter 'q' 
            if key == 27 or key == 113:
                break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
