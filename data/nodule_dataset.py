import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import pickle
import numpy as np
import cv2
import albumentations as A


class NoduleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.heatmaps_dir = os.path.join(opt.dataroot, "trainA")
        self.scans_dir = os.path.join(opt.dataroot, "trainB")

        self.A_path = sorted(make_dataset(self.heatmaps_dir))
        self.B_path = sorted(make_dataset(self.scans_dir))

        self.transform = get_transform(opt)


        self.patients_A, self.patients_B = self.parse_data(self.heatmaps_dir, self.scans_dir)

    def parse_data(self, image_path, label_path):

        patient_numbers_A = set()
        patient_numbers_B = set()

        for patient in range(1, int(len(self.A_path)/32)+1):
            patient_numbers_A.add(patient)
            patient_numbers_B.add(patient)

        patients_A = {patient_number: [] for patient_number in patient_numbers_A}
        patients_B = {patient_number: [] for patient_number in patient_numbers_B}

        slices_32_A = [self.A_path[n:n + 32] for n in range(0, len(self.A_path), 32)]
        slices_32_B = [self.B_path[n:n + 32] for n in range(0, len(self.B_path), 32)]

        for patient in range(1, len(patient_numbers_A)+1):
            line = slices_32_A[patient-1]
            patients_A[patient] = line

        for patient in range(1, len(patient_numbers_B)+1):
            line = slices_32_B[patient-1]
            patients_B[patient] = line

        self.patients_list_A = [files_this_patient for files_this_patient in patients_A.values()]
        self.patients_list_B = [files_this_patient for files_this_patient in patients_B.values()]

        flat_patients_list_A = []
        flat_patients_list_B = []

        for sublist in self.patients_list_A:
            for item in sublist:
                flat_patients_list_A.append(item)

        for sublist in self.patients_list_B:
            for item in sublist:
                flat_patients_list_B.append(item)

        return flat_patients_list_A, flat_patients_list_B


    def __getitem__(self, index):
        index = random.randint(0, len(self.A_path) - 1)
        A_path = self.A_path[index]
        B_path = self.B_path[index]
        #print("real: ", A_path)
        #print("label: ", B_path)

        stacked_A = []
        stacked_B = []

        for i, sublist in enumerate(self.patients_list_A):
            if A_path in sublist:
                for item in sublist:
                    A_img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
                    #A_img = cv2.resize(A_img, (128, 128), interpolation=cv2.INTER_NEAREST)
                    stacked_A.append(A_img)

        for j, sublistj in enumerate(self.patients_list_B):
            if B_path in sublistj:
                for itemj in sublistj:
                    B_img = cv2.imread(itemj, cv2.IMREAD_GRAYSCALE)
                    #B_img = cv2.resize(B_img, (128, 128), interpolation=cv2.INTER_NEAREST)
                    stacked_B.append(B_img)


        random_noise = random.uniform(0, 0.05)
        #random_rotate = random.uniform(0, 1)
        #random_probability_rotation = random.randint(0, 1)
        random_probability_noise = random.randint(0, 1)
        self.transform_da = A.Compose([
            #A.Rotate(limit=(random_rotate, random_rotate), p=random_probability_rotation,
            #         interpolation=cv2.INTER_NEAREST),
            A.GaussNoise(var_limit=(random_noise, random_noise), p=random_probability_noise)
        ])

        stacked_A_augmented = []
        stacked_B_augmented = []

        for image_a, image_b in zip(stacked_A, stacked_B):
            augmented = self.transform_da(image=image_b, mask=image_a)
            stacked_A_augmented.append(augmented['mask'])
            stacked_B_augmented.append(augmented['image'])

        imageA_tensor = torch.from_numpy(np.array(stacked_A_augmented)).unsqueeze(0)
        imageB_tensor = torch.from_numpy(np.array(stacked_B_augmented)).unsqueeze(0)

        imageA_tensor = self.transform(imageA_tensor.type(torch.float))
        imageB_tensor = self.transform(imageB_tensor.type(torch.float))
        imageA_tensor /= 255.
        imageB_tensor /= 255.


        return {
                'A' : imageA_tensor,
                'B' : imageB_tensor,
                #'A_paths': A_path, 'B_paths': B_path
                }

    def __len__(self):
        return len(self.A_path)

    def name(self):
        return 'NodulesDataset'
