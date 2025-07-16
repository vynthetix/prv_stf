from torch import tensor
import uuid

class QueueObjectClass:
    """
    A class to represent an object in a queue with image and metadata.

    Attributes
    ----------
    image : torch.tensor
        The image data.
    id : dict
        The metadata of the object. If not provided, a unique ID is generated.

    Methods
    -------
    get_image()
        Returns the image data.
    get_metadata()
        Returns the metadata of the object.
    """

    def __init__(self, image: tensor, metadata: dict = None) -> None:
        """
        Constructs all the necessary attributes for the QueueObjectClass object.

        Parameters
        ----------
        image : torch.tensor
            The image data.
        metadata : dict, optional
            The metadata of the object. If not provided, a unique ID is generated.
            (default is None)
        """
        self.image = image
        if metadata is None:
            self.id = {'id': uuid.uuid1()}
        else:
            self.id = metadata

    def get_image(self) -> tensor:
        """
        Returns the image data.

        Returns
        -------
        torch.tensor
            The image data.
        """
        return self.image

    def get_metadata(self) -> dict:
        """
        Returns the metadata of the object.

        Returns
        -------
        dict
            The metadata of the object.
        """
        return self.id
