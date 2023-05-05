import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import torch
import numpy as np
import cv2

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

        self.patients_A, self.patients_B = self.parse_data(self.dir_A, self.dir_B)

    def parse_data(self, image_path, label_path):

        patient_numbers_A = set()
        patient_numbers_B = set()

        for patient in range(1, int(self.A_size/32)+1):
            patient_numbers_A.add(patient)

        for patient in range(1, int(self.B_size/32)+1):
            patient_numbers_B.add(patient)

        patients_A = {patient_number: [] for patient_number in patient_numbers_A}
        patients_B = {patient_number: [] for patient_number in patient_numbers_B}

        slices_32_A = [self.A_paths[n:n + 32] for n in range(0, len(self.A_paths), 32)]
        slices_32_B = [self.B_paths[n:n + 32] for n in range(0, len(self.B_paths), 32)]

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

        #patient_A = self.patients_A[index]
        #patient_B = self.patients_B[index]

        #index_A = index % self.A_size
        index_A = random.randint(0, self.A_size - 1)
        A_path = self.A_paths[index_A]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


        stacked_A=[]
        stacked_B=[]

        for i, sublist in enumerate(self.patients_list_A):
            if A_path in sublist:
                for item in sublist:
                    A_img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
                    A_img = cv2.resize(A_img, (128, 128), interpolation =cv2.INTER_NEAREST)
                    stacked_A.append(A_img)

        for i, sublist in enumerate(self.patients_list_B):
            if B_path in sublist:
                for item in sublist:
                    B_img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
                    B_img = cv2.resize(B_img, (128, 128), interpolation=cv2.INTER_NEAREST)
                    stacked_B.append(B_img)

        imageA_tensor = torch.from_numpy(np.array(stacked_A)).unsqueeze(0)
        imageB_tensor = torch.from_numpy(np.array(stacked_B)).unsqueeze(0)

        # imageA_tensor = self.transform(np.array(stacked_A)[np.newaxis,:])
        # imageB_tensor = self.transform(np.array(stacked_B)[np.newaxis,:])
        imageA_tensor = self.transform(imageA_tensor.type(torch.float))
        imageB_tensor = self.transform(imageB_tensor.type(torch.float))
        imageA_tensor /= 255.
        imageB_tensor /= 255.

        #mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        #std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        #for i in range(32):
        #    imageA_tensor[:,i,:,:] -= mean[i]
        #    imageA_tensor[:,i,:,:] /= std[i]

        return {'A': imageA_tensor, 'B': imageB_tensor,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
