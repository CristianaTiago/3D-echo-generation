import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
import torch
import numpy as np


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)

        self.transform = get_transform(opt)

        self.patients_A = self.parse_data(self.dir_A)


    def parse_data(self, image_path):

        patient_numbers_A = set()

        for patient in range(1, int(self.A_size/32)+1):
            patient_numbers_A.add(patient)

        patients_A = {patient_number: [] for patient_number in patient_numbers_A}

        slices_32_A = [self.A_paths[n:n + 32] for n in range(0, len(self.A_paths), 32)]

        for patient in range(1, len(patient_numbers_A)+1):
            line = slices_32_A[patient-1]
            patients_A[patient] = line

        self.patients_list_A = [files_this_patient for files_this_patient in patients_A.values()]

        flat_patients_list_A = []

        for sublist in self.patients_list_A:
            for item in sublist:
                flat_patients_list_A.append(item)


        return flat_patients_list_A

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        stacked_A = []

        for i, sublist in enumerate(self.patients_list_A):
            if A_path in sublist:
                for item in sublist:
                    A_img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
                    #A_img = cv2.resize(A_img, (128, 128), interpolation=cv2.INTER_NEAREST)
                    stacked_A.append(A_img)

        imageA_tensor = torch.from_numpy(np.array(stacked_A)).unsqueeze(0)
        imageA_tensor = self.transform(imageA_tensor.type(torch.float))
        imageA_tensor /= 255.

        return {'A': imageA_tensor, 'A_paths': A_path}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
