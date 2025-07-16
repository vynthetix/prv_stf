import torch
import torchvision.transforms.functional as F
from cvbf.postprocessing.postprocessing import PostProcessing
from time import time as now

class RigidTransformationPostProcessor(PostProcessing):
    """
    A class for performing overlay and underlay operations on input data.
    see https://github.com/nghiaho12/rigid_transform_3D/tree/master

    Attributes:
    -----------
    input : torch.Tensor
        The input data on which overlay and underlay operations will be performed.

    Methods:
    --------
    overlay(overlay, alpha, interpolationMode)
        Overlays the given overlay image on the input data.

    underlay(underlay, alpha, interpolationMode)
        Underlays the given underlay image on the input data.
    """
    def __init__(self, input):
        """
        Initializes the OverlayPostProcessor class with the given input data.

        Parameters:
        -----------
        input : torch.Tensor
            The input data on which overlay and underlay operations will be performed.
        """
        super().__init__(input=input)

    # Custom exceptions to make it easier to distinguish in the unit tests
    class SrcDstSizeMismatchError(Exception):
        pass


    class InvalidPointDimError(Exception):
        pass


    class NotEnoughPointsError(Exception):
        pass


    class RankDeficiencyError(Exception):
        pass


    def rigid_transform(self, src_pts, dst_pts, calc_scale=False):
        """Calculates the optimal rigid transform from src_pts to dst_pts.

        The returned transform minimizes the following least-squares problem
            r = dst_pts - (R @ src_pts + t)
            s = sum(r**2))

        If calc_scale is True, the similarity transform is solved, with the residual being
            r = dst_pts - (scale * R @ src_pts + t)
        where scale is a scalar.

        Parameters
        ----------
        src_pts: matrix of points stored as rows (e.g. Nx3)
        dst_pts: matrix of points stored as rows (e.g. Nx3)
        calc_scale: if True solve for scale

        Returns
        -------
        R: rotation matrix
        t: translation column vector
        scale: scalar, scale=1.0 if calc_scale=False
        """

        dim = src_pts.shape[1]

        if src_pts.shape != dst_pts.shape:
            raise SrcDstSizeMismatchError(
                f"src and dst points aren't the same matrix size {src_pts.shape=} != {dst_pts.shape=}"
            )

        if not (dim == 2 or dim == 3):
            raise InvalidPointDimError(f"Points must be 2D or 3D, src_pts.shape[1] = {dim}")

        if src_pts.shape[0] < dim:
            raise NotEnoughPointsError(f"Not enough points, expect >= {dim} points")

        # find mean/centroid
        centroid_src = torch.mean(src_pts, axis=0)
        centroid_dst = torch.mean(dst_pts, axis=0)

        centroid_src = centroid_src.reshape(-1, dim)
        centroid_dst = centroid_dst.reshape(-1, dim)

        # subtract mean
        # NOTE: doing src_pts -= centroid_src will modifiy input!
        src_pts = src_pts - centroid_src
        dst_pts = dst_pts - centroid_dst

        # the cross-covariance matrix minus the mean calculation for each element
        # https://en.wikipedia.org/wiki/Cross-covariance_matrix
        H = src_pts.T @ dst_pts

        rank = torch.linalg.matrix_rank(H)

        if dim == 2 and rank == 0:
            raise RankDeficiencyError(
                f"Insufficent matrix rank. For 2D points expect rank >= 1 but got {rank}. Maybe your points are all the same?"
            )
        elif dim == 3 and rank <= 1:
            raise RankDeficiencyError(
                f"Insufficent matrix rank. For 3D points expect rank >= 2 but got {rank}. Maybe your points are collinear?"
            )

        # find rotation
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        det = torch.linalg.det(R)
        if det < 0:
            print(f"det(R) = {det}, reflection detected!, correcting for it ...")
            S = torch.eye(dim)
            S[-1, -1] = -1
            R = Vt.T @ S @ U.T

        if calc_scale:
            scale = torch.sqrt(torch.mean(dst_pts**2) / torch.mean(src_pts**2))
        else:
            scale = 1.0

        # now --> with the affine transformation by torchvision the order of execution appears to be:
        # 1. center
        # 2. rotate
        # 3. relocate
        # 4. translate
        # 5. scale
        # --> if we keep the scale variable in trhe following equation, steps 4 and 5 are reversed, i.e, the translation variables are estimated after the scaling. We do not want this!!!!
        #t = -scale * R @ centroid_src.T + centroid_dst.T
        t = R @ centroid_src.T + centroid_dst.T

        return R, t, scale, centroid_src, centroid_dst

    def matchKeypoints(self, src, offset, dst, sigma):
        # transform from relative to absolut pixel coords
        src = src * torch.tensor(self.input.output[0].shape[1:3]).to(self.DEVICE)
        dst = dst * torch.tensor(self.input.output[0].shape[1:3]).to(self.DEVICE)

        lower_limit = torch.median(dst-src, axis=0)[0] - sigma
        upper_limit = torch.median(dst-src, axis=0)[0] + sigma
        filter = torch.sum(torch.add((dst-src < lower_limit), (dst-src > upper_limit)), axis=1)
        dst = dst[torch.where(filter == 0)]
        src = src[torch.where(filter == 0)]

        return src, dst
    
    def transformImage(self, src, dst):
        start = now()
        ret_r, ret_t, ret_s, centroid_src, centroid_dst = self.rigid_transform(src_pts=src, dst_pts=dst, calc_scale=True) # estimate rotation, translation and scale
        src = ((src-centroid_src[0]) @ ret_r) + centroid_src[0] # correct for predicted keypoints
        ret_t = torch.mean(dst-src, axis=0)
        src = src + ret_t
        angle = torch.rad2deg(torch.acos(ret_r[0,0])) * torch.sign(ret_r[1,0])
        self.output = [F.affine(img=self.input.output[0], center=centroid_src[0].flip(0).cpu().numpy().tolist(), translate=ret_t.flip(0).cpu().numpy().tolist(), angle=angle.cpu().numpy().tolist(), scale=ret_s.cpu().numpy().tolist(), shear=0), now()-start, src, dst]