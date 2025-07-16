import torch
import pickle
from cvbf.preprocessing.preprocessing import PreProcessing
from time import time as now

class BackgroundRemovalProcessor(PreProcessing):
    def __init__(self, input, PATCH_STORE_FILE=None) -> None:
        """
        Initialize the BackgroundRemovalProcessor class.

        Parameters:
        - input (PreProcessing): The input data for preprocessing.
        - PATCH_STORE_FILE (str, optional): The file path to load the patch store from. If not provided, initialize empty patch store.

        Attributes:
        - zeros (torch.Tensor): A tensor of zeros with a single element, moved to the device specified by self.DEVICE.
        - ones (torch.Tensor): A tensor of ones with a single element, moved to the device specified by self.DEVICE.
        - update (bool): A flag indicating whether to update the patch store.
        - patchStore (list, optional): The patch store containing cluster centers. If not provided, initialize as None.
        - patchStoreWeight (list, optional): The weights of the patches in the patch store. If not provided, initialize as None.
        - background (torch.Tensor, optional): The background image. If not provided, initialize as None.
        - foreground (torch.Tensor, optional): The foreground image. If not provided, initialize as None.
        - foreground_activity (torch.Tensor, optional): The activity of the foreground. If not provided, initialize as None.
        - foreground_activity_idx (int): The index of the current foreground activity.
        - mask (torch.Tensor, optional): The mask for the foreground. If not provided, initialize as None.
        - original (None): Placeholder for the original data.
        """
        super().__init__(input=input)
        self.zeros = torch.zeros((1)).to(self.DEVICE)
        self.ones = torch.ones((1)).to(self.DEVICE)
        self.update = True
        if PATCH_STORE_FILE is None:
            self.patchStore = None
            self.patchStoreWeight = None
            self.background = None
            self.foreground = None
            self.foreground_activity = None
            self.foreground_activity_idx = 0
            self.mask = None
            self.original = None
        else:
            file = open(PATCH_STORE_FILE, 'rb')
            data = pickle.load(file)
            self.patchStore = data[0]
            self.patchStoreWeight = data[1]
            self.background = data[2]
            self.foreground = data[3]
            self.foreground_activity = data[4]
            self.foreground_activity_idx = 0
            self.mask = data[5]


    def save_patch_store(self, PATCH_STORE_FILE: str) -> None:
        """
        Save the current state of the patch store to a file.

        This function opens a file in write binary mode, serializes the current state of the patch store,
        including the cluster centers, weights, background, foreground, foreground activity, foreground activity index,
        and mask, and writes them to the file.

        Parameters:
        - PATCH_STORE_FILE (str): The file path where the patch store will be saved.

        Returns:
        - None: This function does not return any value. It only saves the patch store to a file.
        """
        print('storing patch store ...')
        file = open(PATCH_STORE_FILE, 'wb')
        pickle.dump([self.patchStore, self.patchStoreWeight, self.background.cpu(), self.foreground.cpu(), self.foreground_activity.cpu(), self.foreground_activity_idx, self.mask.cpu()], file)



    def _check_similarity(self, store, patch, min_correlation) -> torch.Tensor:
        """
        Check the similarity between a given patch and a store of patches.

        This function computes the correlation between the given patch and each patch in the store,
        and returns a tensor containing the indices of the patches in the store that have a correlation
        greater than the specified minimum correlation threshold.

        Parameters:
        - store (torch.Tensor): A tensor containing the store of patches.
        - patch (torch.Tensor): A tensor representing the patch to be compared with the store.
        - min_correlation (float): The minimum correlation threshold for considering a patch as similar.

        Returns:
        - torch.Tensor: A tensor containing the indices of the patches in the store that have a correlation
          greater than the specified minimum correlation threshold.
        """
        # arXiv:1303.2465v1 [cs.CV] 11 Mar 2013
        def compute_correlation(element, patch):
            x = torch.reshape(element, (-1,))
            y = torch.reshape(patch, (-1,))
            return torch.corrcoef(torch.vstack((x,y)))[0,1]

        def compute_l1_distance(element, patch):
            return torch.sum(torch.abs(element-patch))
        return torch.nonzero(torch.Tensor([torch.where(compute_correlation(element, patch) > min_correlation, self.ones, self.zeros) for element in store]))


    def _get_background_patch(self, patch_index: int) -> torch.Tensor:
        """
        Retrieve the index of the background patch from the patch store based on the maximum weight.

        This function iterates through the weights of the patches in the specified patch store index,
        identifies the patch with the maximum weight, and returns its index.

        Parameters:
        - patch_index (int): The index of the patch store from which to retrieve the background patch.

        Returns:
        - torch.Tensor: A tensor containing the index of the background patch in the specified patch store.
        """
        return torch.nonzero(torch.Tensor([torch.where(weight == torch.max(self.patchStoreWeight[patch_index]), 1, 0) for weight in self.patchStoreWeight[patch_index]]))[0]

    def getWeights(self):
        """
        Retrieve the weights of the patches in the patch store.

        This function returns the weights of the patches in the patch store. The weights represent the
        number of times each patch has been considered a cluster center.

        Parameters:
        - None: This function does not take any parameters.

        Returns:
        - torch.Tensor: A tensor containing the weights of the patches in the patch store. The shape of the
          tensor is (number_of_patches, 1), where number_of_patches is the total number of patches in the
          patch store.
        """
        return self.patchStoreWeight

    def set_update(self, update):
        """
        Set the update flag for the BackgroundRemovalProcessor.

        This function sets the update flag to the specified value. The update flag determines whether
        the patch store should be updated with new patches or not.

        Parameters:
        - update (bool): The new value for the update flag. If True, the patch store will be updated.
                         If False, the patch store will not be updated.

        Returns:
        - None: This function does not return any value. It only sets the update flag.
        """
        self.update = update

    def run(self, min_correlation: float, number_of_clusters: int) -> None:
        """
        Run the background removal preprocessing algorithm.

        This function performs the background removal preprocessing algorithm on the input data.
        It updates the patch store, background, foreground, and other relevant attributes based on the
        input parameters and the current state of the object.

        Parameters:
        - min_correlation (float): The minimum correlation threshold for considering a patch as similar to a cluster center.
        - number_of_clusters (int): The maximum number of cluster centers to maintain in the patch store.

        Returns:
        - None: This function does not return any value. It updates the object's attributes directly.
        """
        start = now()
        self.foreground_activity_idx = self.foreground_activity_idx + 1
        if self.foreground_activity_idx > (number_of_clusters-1):
            self.foreground_activity_idx = 0
        self.output = self.input.output
        patches = self.input.output
        if self.patchStore == None:
            self.patchStore = [patch.unsqueeze(0) for patch in patches]
            self.patchStoreWeight = [torch.ones(1,1) for i in range(0, patches.shape[0])]
            self.background = torch.zeros(patches.shape).to(self.DEVICE)
            self.foreground = torch.zeros(patches.shape).to(self.DEVICE)
            self.foreground_activity = torch.zeros(patches.shape[0],number_of_clusters,1).to(self.DEVICE)
            self.mask = torch.zeros(patches.shape).to(self.DEVICE)
        else:
            for i in range(0, patches.shape[0]):
                # Stage 1: collection of label representatives
                idx = self._check_similarity(store=self.patchStore[i], patch=patches[i], min_correlation=min_correlation)
                if idx.shape[0] != 0:
                    # one of the cluster centers is similar to the patch
                    idx = idx[0,0]
                    if self.update == True:
                        self.patchStoreWeight[i][idx,0] = self.patchStoreWeight[i][idx,0] + 1
                        self.patchStore[i][idx,:] = self.patchStore[i][idx,:] + (1 / (self.patchStoreWeight[i][idx,0]+1)) * (patches[i] - self.patchStore[i][idx,:])
                    weights = [weights for weights in self.patchStoreWeight[i]]
                    idxs = sorted(range(len(weights)), key=lambda k: weights[k])
                    if len(self.patchStore[i]) != 1:
                        idx = idxs[len(weights)-1]
                    self.foreground[i] = torch.clamp(torch.abs(patches[i]-self.patchStore[i][idx,:]), min=0.0, max=1.0)
                else:
                    # non of the cluster centers is similar to the patch
                    if self.patchStore[i].shape[0] < (number_of_clusters-1):
                        # fewer than max clusters are in use, create a new cluster center
                        if self.update == True:
                            self.patchStore[i] = torch.vstack([self.patchStore[i], patches[i].unsqueeze(0)])
                            self.patchStoreWeight[i] = torch.vstack([self.patchStoreWeight[i], torch.ones(1,1)])
                        weights = [weights for weights in self.patchStoreWeight[i]]
                        idxs = sorted(range(len(weights)), key=lambda k: weights[k])
                    else:
                        # all available cluster centers are in use, find the one with the lowest weight and replace it with a new center
                        weights = [weights for weights in self.patchStoreWeight[i]]
                        idxs = sorted(range(len(weights)), key=lambda k: weights[k])
                        if self.update == True:
                            self.patchStore[i][idxs[0],:] = patches[i]
                            self.patchStoreWeight[i][idxs[0],:] = torch.ones(1,1)

                # store the current background image composed of the cluster centers with the highest weight.
                self.background[i] = self.patchStore[i][idxs[len(weights)-1],:]
